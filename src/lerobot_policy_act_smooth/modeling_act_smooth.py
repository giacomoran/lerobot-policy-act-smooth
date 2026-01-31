#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACTSmooth Policy

Based on Action Chunking Transformer from Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
(https://huggingface.co/papers/2304.13705).

Adds delay conditioning for smoother real-time inference: instead of waiting for a full chunk to complete,
the policy can "continue" from partially-executed chunks by conditioning on an action prefix.

Note on delay semantics
-----------------------
At inference time, the action prefix may represent actions from the *previous* chunk that have been
committed to execution but are not yet executed due to inference/execution latency. These actions are
"future" relative to the current observation but are guaranteed to be executed, so conditioning on them
keeps the output chunk consistent with the queued prefix and yields smoother control.

Training vs Inference
---------------------
The model handles delays differently in training vs inference:

**Training:**
- Evaluates ALL delay values {0, 1, ..., max_delay} for each sample in parallel
- Input batch has B samples, internally expanded to B*(max_delay+1) virtual samples
- `delays` tensor has shape [B*(max_delay+1)] containing [0,1,...,max_delay, 0,1,...,max_delay, ...]
- `batch[ACTION]` has shape [B, max_delay + chunk_size, action_dim] (full action sequence)
- Loss is computed across all delays

**Inference:**
- Evaluates ONE delay value (you know exactly how many actions were already executed)
- `delays` tensor has shape [B] containing [delay, delay, ...] (same value repeated)
- `batch[ACTION]` has shape [B, max_delay, action_dim] (action prefix only)
- Returns action chunk predictions for the specified delay
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act_smooth import ACTSmoothConfig

# Batch key for delay values. Shape differs between training and inference (see module docstring).
DELAYS = "delays"


class ACTSmoothPolicy(PreTrainedPolicy):
    """
    ACTSmooth Policy based on Action Chunking Transformer
    (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)

    Adds delay conditioning for smoother real-time inference.
    """

    config_class = ACTSmoothConfig
    name = "act_smooth"

    def __init__(
        self,
        config: ACTSmoothConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTSmooth(config)

        # Precompute target indices for training loss computation
        # _indices_target[d, t] = d + t gives the action index for delay d at timestep t
        indices_delay = torch.arange(config.max_delay + 1, dtype=torch.long)
        indices_offset = torch.arange(config.chunk_size, dtype=torch.long)
        self.register_buffer("_indices_target", indices_delay.unsqueeze(1) + indices_offset.unsqueeze(0))

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.

        Note: This method does not support delay conditioning. For delay-conditioned inference,
        use predict_action_chunk directly with delay and action_prefix.
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        action_prefix: Tensor | None = None,
    ) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Dictionary of observation tensors.
        action_prefix: Tensor of shape (batch_size, delay, action_dim) containing
            the committed-but-not-yet-executed actions from the previous chunk. The delay is inferred
            from the tensor size. Pass None for delay=0 (no prefix).

        Returns:
            Tensor of shape (batch_size, chunk_size, action_dim) containing predicted actions.
        """
        self.eval()

        delay = 0 if action_prefix is None else action_prefix.shape[1]
        if delay > self.config.max_delay:
            raise ValueError(f"delay ({delay}) cannot exceed max_delay ({self.config.max_delay}).")

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Get batch size and device
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
            device = batch[OBS_IMAGES][0].device
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]
            device = batch[OBS_ENV_STATE].device

        max_delay = self.config.max_delay
        action_dim = self.config.action_feature.shape[0]

        # Build padded action prefix.
        # WARNING: Positions [delay:] are zero-padded here, but zeros are valid action values.
        # The model relies on batch[DELAYS] to mask these positions with action_prefix_pad_embed in ACTSmooth.forward.
        # If DELAYS is set incorrectly, zeros will be interpreted as real actions.
        if action_prefix is not None and delay > 0:
            action_prefix_padded = torch.zeros(
                (batch_size, max_delay, action_dim),
                dtype=action_prefix.dtype,
                device=action_prefix.device,
            )
            action_prefix_padded[:, :delay] = action_prefix
        else:
            action_prefix_padded = torch.zeros(
                (batch_size, max_delay, action_dim),
                dtype=torch.float32,
                device=device,
            )

        delays = torch.full((batch_size,), delay, dtype=torch.long, device=device)

        batch[ACTION] = action_prefix_padded
        batch[DELAYS] = delays

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        See module docstring for training vs inference differences.

        Expected batch keys:
        - OBS_STATE or OBS_ENV_STATE: observation state
        - ACTION: [B, max_delay + chunk_size, action_dim]
        - action_is_pad: [B, max_delay + chunk_size] padding mask
        """
        max_delay = self.config.max_delay
        chunk_size = self.config.chunk_size

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Get batch size
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        # batch[ACTION]: [B, max_delay + chunk_size, action_dim]
        expected_action_len = max_delay + chunk_size
        actual_action_len = batch[ACTION].shape[1]
        assert actual_action_len == expected_action_len, (
            f"batch[ACTION].shape[1]={actual_action_len} must equal "
            f"max_delay + chunk_size = {max_delay} + {chunk_size} = {expected_action_len}. "
            f"Check that action_delta_indices matches: {self.config.action_delta_indices}"
        )

        # Inner model returns [B*(max_delay+1), chunk_size, action_dim] during training
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # Reshape predictions: [B*(max_delay+1), chunk_size, action_dim] -> [B, max_delay+1, chunk_size, action_dim]
        actions_hat = actions_hat.reshape(batch_size, max_delay + 1, chunk_size, -1)

        # Extract targets for each delay: targets[b, d] = actions[b, d:d+chunk_size]
        # Uses precomputed _indices_target[d, t] = d + t for efficient indexing
        targets = batch[ACTION][:, self._indices_target]  # [B, D, chunk_size, action_dim]

        # Extract padding mask for each delay
        action_is_pad = batch["action_is_pad"][:, self._indices_target]  # [B, D, chunk_size]

        # Compute L1 loss with padding mask
        l1_loss = (F.l1_loss(targets, actions_hat, reduction="none") * ~action_is_pad.unsqueeze(-1)).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTSmooth(nn.Module):
    """ACTSmooth: The underlying neural network for ACTSmoothPolicy.

    Always uses delay conditioning (max_delay >= 1). For vanilla ACT without delay conditioning,
    use the standard ACT policy instead.
    """

    def __init__(self, config: ACTSmoothConfig):
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTSmoothEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTSmoothEncoder(config)
        self.decoder = ACTSmoothDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
        self.encoder_action_prefix_input_proj = nn.Linear(self.config.action_feature.shape[0], config.dim_model)
        n_1d_tokens = 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        n_1d_tokens += config.max_delay
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSmoothSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        self.action_prefix_pad_embed = nn.Parameter(torch.zeros(config.dim_model))

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through ACTSmooth with delay conditioning.

        See module docstring for training vs inference differences.

        The model distinguishes training vs inference by batch[ACTION] sequence length:
        - Training: max_delay + chunk_size (full sequence, delays generated internally)
        - Inference: max_delay (prefix only, delays from batch[DELAYS])
        """
        max_delay = self.config.max_delay

        actions = batch.get(ACTION)
        delays = batch.get(DELAYS)

        if actions is not None:
            action_seq_len = actions.shape[1]
            is_training_batch = action_seq_len == max_delay + self.config.chunk_size
        else:
            is_training_batch = False
            action_seq_len = 0

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        device = batch[OBS_IMAGES][0].device if OBS_IMAGES in batch else batch[OBS_ENV_STATE].device

        # VAE encoder (training only) - runs on original batch_size before expansion
        if self.config.use_vae and is_training_batch and self.training:
            action_targets = actions[:, : self.config.chunk_size]
            action_is_pad_targets = batch["action_is_pad"][:, : self.config.chunk_size]

            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(action_targets)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, action_is_pad_targets], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

        # Build encoder input tokens: images first (sinusoidal 2D pos embed), then 1D features
        encoder_in_tokens = []
        encoder_in_pos_embed = []

        # 1. Images first (sinusoidal 2D pos embed)
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 2. Latent token (learnable pos embed)
        pos_1d_idx = 0
        encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample))
        encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
        pos_1d_idx += 1

        # 3. Env state (if present)
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
            pos_1d_idx += 1

        # 4. Robot state (if present)
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
            pos_1d_idx += 1

        # Stack tokens (before training expansion)
        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)  # [n_tokens, B, dim]
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)  # [n_tokens, 1, dim]

        # Training: expand batch to evaluate all delays in parallel (see module docstring).
        # Expansion happens AFTER backbone (expensive CNN runs once per sample, not per delay).
        if is_training_batch and self.training:
            n_tokens = encoder_in_tokens.shape[0]

            # Expand encoder tokens: [n_tokens, B, dim] -> [n_tokens, B*(max_delay+1), dim]
            encoder_in_tokens = encoder_in_tokens.unsqueeze(2).expand(-1, -1, max_delay + 1, -1)
            encoder_in_tokens = encoder_in_tokens.reshape(n_tokens, batch_size * (max_delay + 1), -1)

            # Expand pos_embed: [n_tokens, 1, dim] -> [n_tokens, B*(max_delay+1), dim]
            encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size * (max_delay + 1), -1)

            # Create delays for all delay values: [max_delay+1] -> [B*(max_delay+1)]
            delays = torch.arange(max_delay + 1, dtype=torch.long, device=device)
            delays = delays.unsqueeze(0).expand(batch_size, -1).reshape(-1)

            # Expand actions: [B, seq_len, action_dim] -> [B*(max_delay+1), seq_len, action_dim]
            actions = (
                actions.unsqueeze(1)
                .expand(-1, max_delay + 1, -1, -1)
                .reshape(batch_size * (max_delay + 1), -1, actions.shape[-1])
            )

            batch_size_effective = batch_size * (max_delay + 1)
        else:
            batch_size_effective = batch_size
            # Expand pos_embed to match batch size for inference
            encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size_effective, -1)
            # Default to delay=0 if not provided
            if delays is None:
                delays = torch.zeros(batch_size_effective, dtype=torch.long, device=device)

        # 5. Action prefix (learnable pos embed, after expansion)
        action_prefix = (
            actions[:, :max_delay]
            if actions is not None
            else torch.zeros(batch_size_effective, max_delay, self.config.action_feature.shape[0], device=device)
        )

        # Project all positions, then mask invalid ones (simpler than variable-length handling)
        action_prefix_embed = self.encoder_action_prefix_input_proj(action_prefix)

        # Replace positions >= delay or padded positions with learnable action_prefix_pad_embed
        if "action_is_pad" in batch:
            prefix_pad = batch["action_is_pad"][:, :max_delay]
            if is_training_batch and self.training:
                prefix_pad = (
                    prefix_pad.unsqueeze(1)
                    .expand(-1, max_delay + 1, -1)
                    .reshape(batch_size_effective, max_delay)
                )
            prefix_pad = prefix_pad.to(device=device)
        else:
            prefix_pad = torch.zeros(
                (batch_size_effective, max_delay),
                dtype=torch.bool,
                device=device,
            )

        mask = (torch.arange(max_delay, device=device)[None, :] >= delays[:, None]) | prefix_pad
        action_prefix_embed = torch.where(
            mask.unsqueeze(-1),
            self.action_prefix_pad_embed[None, None, :].expand_as(action_prefix_embed),
            action_prefix_embed,
        )

        prefix_tokens = action_prefix_embed.permute(1, 0, 2)  # [max_delay, B_eff, dim]
        encoder_in_tokens = torch.cat([encoder_in_tokens, prefix_tokens], dim=0)

        # Use learnable positional embeddings for action prefix
        prefix_pos_embed = self.encoder_1d_feature_pos_embed.weight[pos_1d_idx : pos_1d_idx + max_delay]
        prefix_pos_embed = prefix_pos_embed.unsqueeze(1).expand(-1, batch_size_effective, -1)
        encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, prefix_pos_embed], dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size_effective, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions_out = self.action_head(decoder_out)

        return actions_out, (mu, log_sigma_x2)


class ACTSmoothEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTSmoothConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTSmoothEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTSmoothEncoderLayer(nn.Module):
    def __init__(self, config: ACTSmoothConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTSmoothDecoder(nn.Module):
    def __init__(self, config: ACTSmoothConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTSmoothDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTSmoothDecoderLayer(nn.Module):
    def __init__(self, config: ACTSmoothConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSmoothSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
