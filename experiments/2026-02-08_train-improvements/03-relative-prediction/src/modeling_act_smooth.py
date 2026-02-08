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
"""ACTSmooth Policy with relative action prediction.

Identical to the original ACTSmooth except: all actions (prefix and targets)
are transformed to deltas relative to a_{t_0} (the action at the observation
timestep, always at index length_prefix_past in the action sequence).

This architecturally enforces C0 continuity: predictions are anchored to the
known action at the observation timestep. The model predicts "how much to
change" rather than "where to go".

Transform:
  a'_i = a_i - a_{t_0}

At inference, predictions are converted back:
  pred_abs_i = pred_delta_i + a_{t_0}
"""

import math
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

# Batch keys for training (full action sequence with targets)
# Other training keys (ACTION, OBS_STATE, OBS_ENV_STATE, OBS_IMAGES) are imported from lerobot.utils.constants
ACTION_IS_PAD = "action_is_pad"

# Batch keys for inference (history + committed prefix only)
INFERENCE_ACTION = "inference_action"
INFERENCE_ACTION_IS_PAD = "inference_action_is_pad"


class ACTSmoothPolicy(PreTrainedPolicy):
    """ACTSmooth Policy with relative action prediction."""

    config_class = ACTSmoothConfig
    name = "act_smooth"

    def __init__(self, config: ACTSmoothConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTSmooth(config)

        delays = torch.arange(1, config.length_prefix_future + 1, dtype=torch.long)
        offsets_chunk = torch.arange(config.chunk_size, dtype=torch.long)
        self.register_buffer(
            "_indices_target", config.length_prefix_past + delays.unsqueeze(1) + offsets_chunk.unsqueeze(0)
        )

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
        pass

    def build_inference_action_prefix(
        self, action_prefix_future: Tensor, action_prefix_past: Tensor
    ) -> tuple[Tensor, Tensor]:
        length_prefix_future = self.config.length_prefix_future
        length_prefix_past = self.config.length_prefix_past
        length_prefix_future_effective = action_prefix_future.shape[1]
        length_prefix_past_effective = action_prefix_past.shape[1]
        batch_size = action_prefix_future.shape[0]
        device = action_prefix_future.device
        dtype = action_prefix_future.dtype

        assert 1 <= length_prefix_future_effective <= length_prefix_future
        assert length_prefix_past_effective <= length_prefix_past

        length_pad_past = length_prefix_past - length_prefix_past_effective
        if length_pad_past > 0:
            pad_past = torch.zeros(batch_size, length_pad_past, action_prefix_past.shape[2], device=device, dtype=dtype)
            inference_action_past = torch.cat([pad_past, action_prefix_past], dim=1)
        else:
            inference_action_past = action_prefix_past
        inference_action_is_pad_past = torch.arange(length_prefix_past, device=device)[None, :] < length_pad_past

        length_pad_future = length_prefix_future - length_prefix_future_effective
        if length_pad_future > 0:
            pad_future = torch.zeros(
                batch_size, length_pad_future, action_prefix_future.shape[2], device=device, dtype=dtype
            )
            inference_action_future = torch.cat([action_prefix_future, pad_future], dim=1)
        else:
            inference_action_future = action_prefix_future
        inference_action_is_pad_future = (
            torch.arange(length_prefix_future, device=device)[None, :] >= length_prefix_future_effective
        )

        inference_action = torch.cat([inference_action_past, inference_action_future], dim=1)
        inference_action_is_pad = torch.cat(
            [
                inference_action_is_pad_past.expand(batch_size, -1),
                inference_action_is_pad_future.expand(batch_size, -1),
            ],
            dim=1,
        )
        return inference_action, inference_action_is_pad

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("ACTSmoothPolicy does not implement select_action. Use predict_action_chunk instead.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        assert INFERENCE_ACTION in batch
        assert INFERENCE_ACTION_IS_PAD in batch
        self._set_eval_mode()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # === RELATIVE PREDICTION: transform prefix to deltas ===
        # a_{t_0} is the first action in the future prefix (at index length_prefix_past)
        length_prefix_past = self.config.length_prefix_past
        action_anchor = batch[INFERENCE_ACTION][:, length_prefix_past : length_prefix_past + 1]  # [B, 1, action_dim]
        batch = dict(batch)
        batch[INFERENCE_ACTION] = batch[INFERENCE_ACTION] - action_anchor

        # Model predicts relative actions
        actions_relative = self.model(batch)[0]  # [B, chunk_size, action_dim]

        # Convert back to absolute
        actions = actions_relative + action_anchor
        return actions

    def _set_eval_mode(self):
        """Set the model to evaluation mode (PyTorch standard)."""
        super().train(False)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        length_prefix_past = self.config.length_prefix_past
        length_prefix_future = self.config.length_prefix_future
        chunk_size = self.config.chunk_size

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        expected_length_action = length_prefix_past + length_prefix_future + chunk_size
        actual_length_action = batch[ACTION].shape[1]
        assert actual_length_action == expected_length_action

        # === RELATIVE PREDICTION: transform all actions to deltas ===
        # a_{t_0} is at index length_prefix_past in the action sequence
        action_anchor = batch[ACTION][:, length_prefix_past : length_prefix_past + 1]  # [B, 1, action_dim]
        batch = dict(batch)
        batch[ACTION] = batch[ACTION] - action_anchor

        # Inner model returns [B*length_prefix_future, chunk_size, action_dim] during training
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # Reshape predictions: [B*length_prefix_future, chunk_size, action_dim] -> [B, length_prefix_future, chunk_size, action_dim]
        actions_hat = actions_hat.reshape(batch_size, length_prefix_future, chunk_size, -1)

        # Extract targets for each delay using precomputed indices
        # Targets are already relative (we transformed batch[ACTION] above)
        targets = batch[ACTION][:, self._indices_target]  # [B, length_prefix_future, chunk_size, action_dim]

        # Extract padding mask for each delay
        action_is_pad = batch[ACTION_IS_PAD][:, self._indices_target]  # [B, length_prefix_future, chunk_size]

        # Compute L1 loss with padding mask (on relative actions)
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
    """ACTSmooth: The underlying neural network for ACTSmoothPolicy."""

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
            self.vae_encoder_action_input_proj = nn.Linear(self.config.action_feature.shape[0], config.dim_model)
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
        n_1d_tokens += config.length_prefix_past
        n_1d_tokens += config.length_prefix_future
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSmoothSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        self.action_prefix_pad_embed = nn.Parameter(torch.zeros(config.dim_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        length_prefix_past = self.config.length_prefix_past
        length_prefix_future = self.config.length_prefix_future

        has_action = batch.get(ACTION) is not None
        has_action_is_pad = batch.get(ACTION_IS_PAD) is not None
        has_inference_action = batch.get(INFERENCE_ACTION) is not None
        has_inference_action_is_pad = batch.get(INFERENCE_ACTION_IS_PAD) is not None

        assert has_action == has_action_is_pad
        assert has_inference_action == has_inference_action_is_pad
        assert not (has_action and has_inference_action)
        assert has_action or has_inference_action
        is_training_batch = has_action

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        device = batch[OBS_IMAGES][0].device if OBS_IMAGES in batch else batch[OBS_ENV_STATE].device

        encoder_in_tokens = []
        encoder_in_pos_embed = []

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        if is_training_batch:
            actions = batch[ACTION]

            if self.config.use_vae:
                target_start_idx = length_prefix_past + 1
                action_targets = actions[:, target_start_idx : target_start_idx + self.config.chunk_size]
                action_is_pad_targets = batch[ACTION_IS_PAD][
                    :, target_start_idx : target_start_idx + self.config.chunk_size
                ]
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
                    (batch_size, 2 if self.config.robot_state_feature else 1), False, device=device
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

            pos_1d_idx = 0
            encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
            pos_1d_idx += 1

            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
                pos_1d_idx += 1

            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
                pos_1d_idx += 1

            encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

            n_tokens = encoder_in_tokens.shape[0]
            batch_size_effective = batch_size * length_prefix_future
            encoder_in_tokens = encoder_in_tokens.unsqueeze(2).expand(-1, -1, length_prefix_future, -1)
            encoder_in_tokens = encoder_in_tokens.reshape(n_tokens, batch_size_effective, -1)
            encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size_effective, -1)

            delays = torch.arange(1, length_prefix_future + 1, dtype=torch.long, device=device)
            delays = delays.unsqueeze(0).expand(batch_size, -1).reshape(-1)

            actions = (
                actions.unsqueeze(1)
                .expand(-1, length_prefix_future, -1, -1)
                .reshape(batch_size_effective, -1, actions.shape[-1])
            )

            if length_prefix_past > 0:
                action_prefix_past = actions[:, :length_prefix_past]
                is_pad_prefix_past = batch[ACTION_IS_PAD][:, :length_prefix_past]
                is_pad_prefix_past = (
                    is_pad_prefix_past.unsqueeze(1)
                    .expand(-1, length_prefix_future, -1)
                    .reshape(batch_size_effective, length_prefix_past)
                )
                embed_prefix_past = self.encoder_action_prefix_input_proj(action_prefix_past)
                embed_prefix_past = torch.where(
                    is_pad_prefix_past.unsqueeze(-1),
                    self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_past),
                    embed_prefix_past,
                )
                tokens_prefix_past = embed_prefix_past.permute(1, 0, 2)
                encoder_in_tokens = torch.cat([encoder_in_tokens, tokens_prefix_past], dim=0)
                pos_embed_prefix_past = self.encoder_1d_feature_pos_embed.weight[
                    pos_1d_idx : pos_1d_idx + length_prefix_past
                ]
                pos_embed_prefix_past = pos_embed_prefix_past.unsqueeze(1).expand(-1, batch_size_effective, -1)
                encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, pos_embed_prefix_past], dim=0)
                pos_1d_idx += length_prefix_past

            action_prefix_future = actions[:, length_prefix_past : length_prefix_past + length_prefix_future]
            pad_prefix_future = batch[ACTION_IS_PAD][:, length_prefix_past : length_prefix_past + length_prefix_future]
            pad_prefix_future = (
                pad_prefix_future.unsqueeze(1)
                .expand(-1, length_prefix_future, -1)
                .reshape(batch_size_effective, length_prefix_future)
            )
            is_pad_prefix_future = (
                torch.arange(length_prefix_future, device=device)[None, :] >= delays[:, None]
            ) | pad_prefix_future
            embed_prefix_future = self.encoder_action_prefix_input_proj(action_prefix_future)
            embed_prefix_future = torch.where(
                is_pad_prefix_future.unsqueeze(-1),
                self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_future),
                embed_prefix_future,
            )
            tokens_prefix_future = embed_prefix_future.permute(1, 0, 2)
            encoder_in_tokens = torch.cat([encoder_in_tokens, tokens_prefix_future], dim=0)
            pos_embed_prefix_future = self.encoder_1d_feature_pos_embed.weight[
                pos_1d_idx : pos_1d_idx + length_prefix_future
            ]
            pos_embed_prefix_future = pos_embed_prefix_future.unsqueeze(1).expand(-1, batch_size_effective, -1)
            encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, pos_embed_prefix_future], dim=0)

        else:
            inference_action = batch[INFERENCE_ACTION]
            inference_action_is_pad = batch[INFERENCE_ACTION_IS_PAD]

            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

            pos_1d_idx = 0
            encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
            pos_1d_idx += 1

            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
                pos_1d_idx += 1

            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].unsqueeze(0))
                pos_1d_idx += 1

            encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)
            batch_size_effective = batch_size
            encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size_effective, -1)

            if length_prefix_past > 0:
                action_prefix_past = inference_action[:, :length_prefix_past]
                is_pad_prefix_past = inference_action_is_pad[:, :length_prefix_past]
                embed_prefix_past = self.encoder_action_prefix_input_proj(action_prefix_past)
                embed_prefix_past = torch.where(
                    is_pad_prefix_past.unsqueeze(-1),
                    self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_past),
                    embed_prefix_past,
                )
                tokens_prefix_past = embed_prefix_past.permute(1, 0, 2)
                encoder_in_tokens = torch.cat([encoder_in_tokens, tokens_prefix_past], dim=0)
                pos_embed_prefix_past = self.encoder_1d_feature_pos_embed.weight[
                    pos_1d_idx : pos_1d_idx + length_prefix_past
                ]
                pos_embed_prefix_past = pos_embed_prefix_past.unsqueeze(1).expand(-1, batch_size_effective, -1)
                encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, pos_embed_prefix_past], dim=0)
                pos_1d_idx += length_prefix_past

            action_prefix_future = inference_action[:, length_prefix_past:]
            is_pad_prefix_future = inference_action_is_pad[:, length_prefix_past:]
            embed_prefix_future = self.encoder_action_prefix_input_proj(action_prefix_future)
            embed_prefix_future = torch.where(
                is_pad_prefix_future.unsqueeze(-1),
                self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_future),
                embed_prefix_future,
            )
            tokens_prefix_future = embed_prefix_future.permute(1, 0, 2)
            encoder_in_tokens = torch.cat([encoder_in_tokens, tokens_prefix_future], dim=0)
            pos_embed_prefix_future = self.encoder_1d_feature_pos_embed.weight[
                pos_1d_idx : pos_1d_idx + length_prefix_future
            ]
            pos_embed_prefix_future = pos_embed_prefix_future.unsqueeze(1).expand(-1, batch_size_effective, -1)
            encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, pos_embed_prefix_future], dim=0)

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
        super().__init__()
        self.layers = nn.ModuleList([ACTSmoothDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None) -> Tensor:
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

    def maybe_add_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None) -> Tensor:
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
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSmoothSinusoidalPositionEmbedding2d(nn.Module):
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
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
