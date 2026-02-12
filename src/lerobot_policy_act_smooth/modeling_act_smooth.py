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

Adds action prefix conditioning for smoother real-time inference: instead of waiting for a full chunk
to complete, the policy can "continue" from partially-executed chunks by conditioning on an action prefix.

Action Prefix Structure
-----------------------
The prefix consists of two parts:
  Prefix = [past completed actions] + [committed pending actions]
           |---- k actions (fixed) ----|  |---- d actions (variable, d >= 1) ---|

- History (past completed): Actions that have already been executed. Provides context for continuity.
- Committed (pending): Actions that will execute during inference. Must include at least a_{t_0}
  because inference has non-zero latency.

This decouples context (history + committed) from delay (only committed), allowing rich context
for smooth continuity while maintaining fast reactivity.

Training vs Inference
---------------------
The model handles delays differently in training vs inference:

**Training:**
- Evaluates ALL delay values {1, 2, ..., length_prefix_future} for each sample in parallel
- Input batch has B samples, internally expanded to B*length_prefix_future virtual samples
- `batch[ACTION]` has shape [B, length_prefix_past + length_prefix_future + chunk_size, action_dim] (full action sequence)
- `batch[ACTION_IS_PAD]` has shape [B, length_prefix_past + length_prefix_future + chunk_size] (padding mask)
- Loss is computed across all delays

**Inference:**
- Evaluates ONE delay value (determined by the padding mask)
- `batch[INFERENCE_ACTION]` has shape [B, length_prefix_past + length_prefix_future, action_dim] (history + committed prefix)
- `batch[INFERENCE_ACTION_IS_PAD]` has shape [B, length_prefix_past + length_prefix_future] (padding mask encodes both history padding and which prefix positions are valid)
- Returns action chunk predictions for the specified delay

Relative Action Representation
------------------------------
When `use_action_relative=True`, all actions (prefix + targets) are transformed to deltas relative
to an anchor before being fed to the model. At inference, the inverse transform recovers absolute actions.

**Why relative?**
Action chunking policies predict a sequence of future actions. At chunk boundaries, consecutive
predictions must join smoothly. In absolute action space, the model must predict exact positions
across the entire workspace — a hard function to learn — and two independently predicted chunks
can disagree on absolute position, causing discontinuities. In relative space, the model only
needs to predict motion offsets from a known anchor. This is a simpler function (the shape of
the trajectory, not its absolute placement), and prediction errors near the boundary translate to
smaller physical discontinuities since the offsets there are small.

**Anchor: action at t_0**
The anchor is the action at position t_0 in the prefix — i.e. `batch[ACTION][:, length_prefix_past]`
during training and `batch[INFERENCE_ACTION][:, length_prefix_past]` during inference. All actions
(prefix + targets) become deltas relative to this anchor. This means the first future prefix
element is always zero, but the model still sees non-trivial past prefix values and the relative
offsets of future targets.

We tried anchoring on observation.state (s_{t_0}) instead, but with teleop data the observation
states are noisy (servo noise, communication jitter), making the relative representation
inconsistent across chunks. The model trains fine (low L1 loss) but doesn't generalize to
real inference where the noise pattern differs.
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
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics dict (feature_name -> {stat_name -> tensor}).
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTSmooth(config)

        # Precompute indices to extract target actions from batch[ACTION] for each delay.
        # During training, we evaluate all delays in parallel. For a given delay d, the model
        # predicts a chunk starting d steps into the future, so targets start at index
        # (length_prefix_past + d) in batch[ACTION].
        # Shape: [length_prefix_future, chunk_size] - one row per delay value (1, 2, ..., length_prefix_future)
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
        """This should be called whenever the environment is reset."""
        pass

    def build_inference_action_prefix(
        self,
        action_prefix_future: Tensor,
        action_prefix_past: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build INFERENCE_ACTION and INFERENCE_ACTION_IS_PAD tensors from action prefix tensors.

        Args:
            action_prefix_future: Tensor [B, length_prefix_future_effective, action_dim]
                where 1 <= length_prefix_future_effective <= length_prefix_future.
                Must have at least 1 action (d >= 1) since training uses delays {1..D}.
            action_prefix_past: Tensor [B, length_prefix_past_effective, action_dim]
                where length_prefix_past_effective <= length_prefix_past.
                Use [B, 0, action_dim] for no past prefix.

        Returns:
            Tuple of (inference_action, inference_action_is_pad):
            - inference_action: [B, length_prefix_past + length_prefix_future, action_dim]
            - inference_action_is_pad: [B, length_prefix_past + length_prefix_future]
        """
        length_prefix_future = self.config.length_prefix_future
        length_prefix_past = self.config.length_prefix_past
        length_prefix_future_effective = action_prefix_future.shape[1]
        length_prefix_past_effective = action_prefix_past.shape[1]
        batch_size = action_prefix_future.shape[0]
        device = action_prefix_future.device
        dtype = action_prefix_future.dtype

        # WARNING: Inference code must provide at least 1 action in action_prefix_future.
        # The model was trained on delays {1..length_prefix_future}, so delay=0 is out-of-distribution.
        # For the first inference (no prior chunk), use the current robot state as a 1-action prefix.
        assert 1 <= length_prefix_future_effective <= length_prefix_future, (
            f"action_prefix_future has {length_prefix_future_effective} steps, but must have 1 <= d <= {length_prefix_future} "
            f"(training uses delays {{1..{length_prefix_future}}}, d=0 is out-of-distribution)"
        )
        assert length_prefix_past_effective <= length_prefix_past, (
            f"action_prefix_past has {length_prefix_past_effective} steps, but length_prefix_past is {length_prefix_past}"
        )

        # 1. Build past portion (past completed actions)
        # Pad at the START (older positions are missing first)
        length_pad_past = length_prefix_past - length_prefix_past_effective
        if length_pad_past > 0:
            pad_past = torch.zeros(batch_size, length_pad_past, action_prefix_past.shape[2], device=device, dtype=dtype)
            inference_action_past = torch.cat([pad_past, action_prefix_past], dim=1)
        else:
            inference_action_past = action_prefix_past

        # Past mask: first length_pad_past positions are padded
        inference_action_is_pad_past = torch.arange(length_prefix_past, device=device)[None, :] < length_pad_past

        # 2. Build future portion (committed pending actions)
        # Pad at the END (positions beyond length_prefix_future_effective are padded)
        length_pad_future = length_prefix_future - length_prefix_future_effective
        if length_pad_future > 0:
            pad_future = torch.zeros(
                batch_size, length_pad_future, action_prefix_future.shape[2], device=device, dtype=dtype
            )
            inference_action_future = torch.cat([action_prefix_future, pad_future], dim=1)
        else:
            inference_action_future = action_prefix_future

        # Future mask: positions >= length_prefix_future_effective are padded
        inference_action_is_pad_future = (
            torch.arange(length_prefix_future, device=device)[None, :] >= length_prefix_future_effective
        )

        # 3. Concatenate into final tensors
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
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Dictionary containing observation tensors and inference action prefix keys:
                - INFERENCE_ACTION: [B, length_prefix_past + length_prefix_future, action_dim]
                - INFERENCE_ACTION_IS_PAD: [B, length_prefix_past + length_prefix_future]

        Returns:
            Tensor of shape (batch_size, chunk_size, action_dim) containing predicted actions.
        """
        assert INFERENCE_ACTION in batch, f"batch must contain {INFERENCE_ACTION}"
        assert INFERENCE_ACTION_IS_PAD in batch, f"batch must contain {INFERENCE_ACTION_IS_PAD}"

        self.eval()

        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Relative action representation: anchor on action at t_0 (index length_prefix_past
        # in the prefix). Subtract anchor from prefix, predict relative chunk, add anchor back.
        if self.config.use_action_relative:
            k = self.config.length_prefix_past
            anchor = batch[INFERENCE_ACTION][:, k : k + 1]  # [B, 1, action_dim]
            batch[INFERENCE_ACTION] = batch[INFERENCE_ACTION] - anchor
            actions = self.model(batch)[0]
            actions = actions + anchor
            return actions

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        See module docstring for training vs inference differences.

        Expected batch keys:
        - OBS_STATE or OBS_ENV_STATE: observation state
        - ACTION: [B, length_prefix_past + length_prefix_future + chunk_size, action_dim]
        - ACTION_IS_PAD: [B, length_prefix_past + length_prefix_future + chunk_size] padding mask
        """
        length_prefix_past = self.config.length_prefix_past
        length_prefix_future = self.config.length_prefix_future
        chunk_size = self.config.chunk_size

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Get batch size
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        # batch[ACTION]: [B, length_prefix_past + length_prefix_future + chunk_size, action_dim]
        expected_length_action = length_prefix_past + length_prefix_future + chunk_size
        actual_length_action = batch[ACTION].shape[1]
        assert actual_length_action == expected_length_action, (
            f"batch[ACTION].shape[1]={actual_length_action} must equal "
            f"length_prefix_past + length_prefix_future + chunk_size = {length_prefix_past} + {length_prefix_future} + {chunk_size} = {expected_length_action}. "
            f"Check that action_delta_indices matches: {self.config.action_delta_indices}"
        )

        # Relative action representation: anchor on action at t_0 (index length_prefix_past),
        # transform all actions (prefix + targets) to deltas relative to this anchor.
        if self.config.use_action_relative:
            anchor = batch[ACTION][:, length_prefix_past : length_prefix_past + 1]  # [B, 1, action_dim]
            batch = dict(batch)
            batch[ACTION] = batch[ACTION] - anchor

        # Inner model returns [B*length_prefix_future, chunk_size, action_dim] during training
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # Reshape predictions: [B*length_prefix_future, chunk_size, action_dim] -> [B, length_prefix_future, chunk_size, action_dim]
        actions_hat = actions_hat.reshape(batch_size, length_prefix_future, chunk_size, -1)

        # Extract targets for each delay using precomputed indices
        targets = batch[ACTION][:, self._indices_target]  # [B, length_prefix_future, chunk_size, action_dim]

        # Extract padding mask for each delay
        action_is_pad = batch[ACTION_IS_PAD][:, self._indices_target]  # [B, length_prefix_future, chunk_size]

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


def _build_block_attn_masks(
    n_shared: int,
    n_branches: int,
    n_branch_tokens: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build VLASH-style block attention masks for parallel delay training.

    Packs D branches into the sequence dimension instead of expanding batch.
    Shared tokens (images, latent, state, prefix_past) attend to all shared tokens.
    Each branch's tokens attend to shared tokens + own branch only.

    Args:
        n_shared: Number of shared encoder tokens (images + latent + env_state + robot_state + prefix_past)
        n_branches: D = length_prefix_future (number of delay branches)
        n_branch_tokens: Number of tokens per branch in the encoder (= length_prefix_future, the prefix_future positions)
        chunk_size: Number of decoder queries per branch
        device: Target device

    Returns:
        (mask_enc, mask_dec_self, mask_dec_cross) — bool masks where True = blocked.
    """
    S_total_enc = n_shared + n_branches * n_branch_tokens

    # Encoder self-attention mask [S_total_enc, S_total_enc]
    mask_enc = torch.ones(S_total_enc, S_total_enc, dtype=torch.bool, device=device)

    # Shared-to-shared: allowed
    mask_enc[:n_shared, :n_shared] = False

    for i in range(n_branches):
        start = n_shared + i * n_branch_tokens
        end = start + n_branch_tokens
        # Branch i attends to shared: allowed
        mask_enc[start:end, :n_shared] = False
        # Branch i attends to itself: allowed
        mask_enc[start:end, start:end] = False
        # Shared attends to branch i: blocked (already True by default)

    # Decoder self-attention mask [D*C, D*C] — block-diagonal
    S_dec = n_branches * chunk_size
    mask_dec_self = torch.ones(S_dec, S_dec, dtype=torch.bool, device=device)
    for i in range(n_branches):
        start = i * chunk_size
        end = start + chunk_size
        mask_dec_self[start:end, start:end] = False

    # Decoder cross-attention mask [D*C, S_total_enc]
    mask_dec_cross = torch.ones(S_dec, S_total_enc, dtype=torch.bool, device=device)
    for i in range(n_branches):
        dec_start = i * chunk_size
        dec_end = dec_start + chunk_size
        enc_start = n_shared + i * n_branch_tokens
        enc_end = enc_start + n_branch_tokens
        # Decoder branch i attends to shared encoder tokens
        mask_dec_cross[dec_start:dec_end, :n_shared] = False
        # Decoder branch i attends to its own encoder branch tokens
        mask_dec_cross[dec_start:dec_end, enc_start:enc_end] = False

    return mask_enc, mask_dec_self, mask_dec_cross


class ACTSmooth(nn.Module):
    """ACTSmooth: The underlying neural network for ACTSmoothPolicy.

    Always uses delay conditioning (length_prefix_future >= 1). For vanilla ACT without delay conditioning,
    use the standard ACT policy instead.

    Token ordering in encoder: [images, latent, env_state, robot_state, prefix_past, prefix_future]
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
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through ACTSmooth with delay conditioning.

        See module docstring for training vs inference differences.

        Mode is determined by batch keys:
        - Training batch: ACTION and ACTION_IS_PAD
        - Inference batch: INFERENCE_ACTION and INFERENCE_ACTION_IS_PAD
        """
        length_prefix_past = self.config.length_prefix_past
        length_prefix_future = self.config.length_prefix_future

        # Infer mode from keys so model.eval() can still run loss computation on training batches.
        # Use .get() to check for actual values, not just key existence (preprocessor adds ACTION: None).
        has_action = batch.get(ACTION) is not None
        has_action_is_pad = batch.get(ACTION_IS_PAD) is not None
        has_inference_action = batch.get(INFERENCE_ACTION) is not None
        has_inference_action_is_pad = batch.get(INFERENCE_ACTION_IS_PAD) is not None

        assert has_action == has_action_is_pad, (
            f"Batch must provide both {ACTION} and {ACTION_IS_PAD}, or neither. "
            f"Got {ACTION}={has_action}, {ACTION_IS_PAD}={has_action_is_pad}."
        )
        assert has_inference_action == has_inference_action_is_pad, (
            f"Batch must provide both {INFERENCE_ACTION} and {INFERENCE_ACTION_IS_PAD}, or neither. "
            f"Got {INFERENCE_ACTION}={has_inference_action}, {INFERENCE_ACTION_IS_PAD}={has_inference_action_is_pad}."
        )
        assert not (has_action and has_inference_action), (
            f"Batch is ambiguous: contains both training keys ({ACTION}, {ACTION_IS_PAD}) "
            f"and inference keys ({INFERENCE_ACTION}, {INFERENCE_ACTION_IS_PAD})."
        )
        assert has_action or has_inference_action, (
            f"Batch must contain either training keys ({ACTION}, {ACTION_IS_PAD}) "
            f"or inference keys ({INFERENCE_ACTION}, {INFERENCE_ACTION_IS_PAD})."
        )
        is_training_batch = has_action

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        device = batch[OBS_IMAGES][0].device if OBS_IMAGES in batch else batch[OBS_ENV_STATE].device

        # Build encoder input tokens: images first (sinusoidal 2D pos embed), then 1D features
        # This is shared between training and inference (backbone runs once per sample)
        encoder_in_tokens = []
        encoder_in_pos_embed = []

        # 1. Images (sinusoidal 2D pos embed)
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.append(cam_features)  # [h*w, B, dim]
                encoder_in_pos_embed.append(cam_pos_embed)  # [h*w, 1, dim]

        # ===== TRAINING-BATCH PATH =====
        if is_training_batch:
            actions = batch[ACTION]

            # VAE encoder - runs on original batch_size before expansion
            # VAE targets are for d=1: actions starting at index length_prefix_past + 1
            if self.config.use_vae:
                target_start_idx = length_prefix_past + 1  # d=1 target starts at t_1
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

            # 2. Latent token (learnable pos embed)
            pos_1d_idx = 0
            encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample).unsqueeze(0))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].reshape(1, 1, -1))
            pos_1d_idx += 1

            # 3. Env state (if present)
            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]).unsqueeze(0))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].reshape(1, 1, -1))
                pos_1d_idx += 1

            # 4. Robot state (if present)
            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(0))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].reshape(1, 1, -1))
                pos_1d_idx += 1

            # 5. Action history (past completed actions) — shared, at batch B (no expand)
            if length_prefix_past > 0:
                action_prefix_past = actions[:, :length_prefix_past]  # [B, K, action_dim]
                is_pad_prefix_past = batch[ACTION_IS_PAD][:, :length_prefix_past]  # [B, K]

                embed_prefix_past = self.encoder_action_prefix_input_proj(action_prefix_past)

                embed_prefix_past = torch.where(
                    is_pad_prefix_past.unsqueeze(-1),
                    self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_past),
                    embed_prefix_past,
                )

                encoder_in_tokens.append(embed_prefix_past.permute(1, 0, 2))  # [K, B, dim]
                encoder_in_pos_embed.append(
                    self.encoder_1d_feature_pos_embed.weight[pos_1d_idx : pos_1d_idx + length_prefix_past].unsqueeze(1)
                )  # [K, 1, dim]
                pos_1d_idx += length_prefix_past

            # Record n_shared before adding branch tokens
            n_shared = sum(t.shape[0] for t in encoder_in_tokens)

            # 6. length_prefix_future branches of prefix_future — packed into sequence dim with block attention
            # Each branch has length_prefix_future positions, total length_prefix_future^2 tokens appended.
            # For branch d (delay=d), positions 0..d-1 are valid, positions d..length_prefix_future-1 are padded.
            action_prefix_future = actions[
                :, length_prefix_past : length_prefix_past + length_prefix_future
            ]  # [B, length_prefix_future, action_dim]
            pad_prefix_future = batch[ACTION_IS_PAD][
                :, length_prefix_past : length_prefix_past + length_prefix_future
            ]  # [B, length_prefix_future]

            # Project all future prefix positions at once: [B, length_prefix_future, dim]
            embed_prefix_future_all = self.encoder_action_prefix_input_proj(action_prefix_future)

            # Build length_prefix_future branches, each with length_prefix_future positions
            # For branch d (0-indexed: d=0 means delay=1), positions >= delay are padded
            delays = torch.arange(1, length_prefix_future + 1, dtype=torch.long, device=device)
            positions = torch.arange(length_prefix_future, dtype=torch.long, device=device)

            # is_pad_branch[d, pos] = True if pos >= delay_d, for delay masking
            is_pad_delay = positions[None, :] >= delays[:, None]  # [length_prefix_future, length_prefix_future]

            # Combine with data padding
            is_pad_branch = (
                is_pad_delay[None, :, :] | pad_prefix_future[:, None, :]
            )  # [B, length_prefix_future, length_prefix_future]

            # Replicate embeddings for each branch: [B, length_prefix_future, dim] -> [B, length_prefix_future, length_prefix_future, dim]
            embed_branches = embed_prefix_future_all.unsqueeze(1).expand(-1, length_prefix_future, -1, -1)

            # Apply pad_embed to padded positions
            embed_branches = torch.where(
                is_pad_branch.unsqueeze(-1),
                self.action_prefix_pad_embed[None, None, None, :].expand_as(embed_branches),
                embed_branches,
            )

            # Reshape to sequence: [B, length_prefix_future, length_prefix_future, dim] -> [length_prefix_future^2, B, dim]
            encoder_in_tokens.append(
                embed_branches.reshape(batch_size, length_prefix_future * length_prefix_future, -1).permute(1, 0, 2)
            )

            # Positional embeddings: same positions repeated for each branch
            pos_embed_prefix_future = self.encoder_1d_feature_pos_embed.weight[
                pos_1d_idx : pos_1d_idx + length_prefix_future
            ]  # [length_prefix_future, dim]
            encoder_in_pos_embed.append(
                pos_embed_prefix_future.repeat(length_prefix_future, 1).unsqueeze(1)
            )  # [length_prefix_future^2, 1, dim]

            # Assemble encoder input
            encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)  # [S_total, B, dim]
            encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)  # [S_total, 1, dim]
            encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size, -1)  # [S_total, B, dim]

            # Build block attention masks (cached)
            if not hasattr(self, "_cached_block_attn_masks") or self._cached_block_attn_masks[0] != n_shared:
                self._cached_block_attn_masks = (
                    n_shared,
                    *_build_block_attn_masks(
                        n_shared, length_prefix_future, length_prefix_future, self.config.chunk_size, device
                    ),
                )
            _, mask_enc, mask_dec_self, mask_dec_cross = self._cached_block_attn_masks

            # Encoder (batch B, block attention)
            encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed, attn_mask=mask_enc)

            # Decoder: length_prefix_future branches of chunk_size zero-queries packed in sequence dim
            decoder_in = torch.zeros(
                (length_prefix_future * self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=device,
            )
            # Same chunk pos embed repeated for each branch
            decoder_pos_embed = self.decoder_pos_embed.weight.repeat(length_prefix_future, 1).unsqueeze(1)

            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=decoder_pos_embed,
                self_attn_mask=mask_dec_self,
                cross_attn_mask=mask_dec_cross,
            )

            # [length_prefix_future*C, B, dim] -> [B*length_prefix_future, C, dim] for loss computation
            decoder_out = decoder_out.reshape(
                length_prefix_future, self.config.chunk_size, batch_size, self.config.dim_model
            )
            decoder_out = decoder_out.permute(2, 0, 1, 3).reshape(
                batch_size * length_prefix_future, self.config.chunk_size, -1
            )
            actions_out = self.action_head(decoder_out)

            return actions_out, (mu, log_sigma_x2)

        # ===== INFERENCE-BATCH PATH =====
        else:
            inference_action = batch[INFERENCE_ACTION]
            inference_action_is_pad = batch[INFERENCE_ACTION_IS_PAD]

            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

            # 2. Latent token (learnable pos embed)
            pos_1d_idx = 0
            encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample).unsqueeze(0))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].reshape(1, 1, -1))
            pos_1d_idx += 1

            # 3. Env state (if present)
            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]).unsqueeze(0))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].reshape(1, 1, -1))
                pos_1d_idx += 1

            # 4. Robot state (if present)
            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(0))
                encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[pos_1d_idx].reshape(1, 1, -1))
                pos_1d_idx += 1

            # 5. Action history (past completed actions) - learnable pos embed
            if length_prefix_past > 0:
                action_prefix_past = inference_action[:, :length_prefix_past]
                is_pad_prefix_past = inference_action_is_pad[:, :length_prefix_past]

                embed_prefix_past = self.encoder_action_prefix_input_proj(action_prefix_past)

                # Replace padded positions with learnable action_prefix_pad_embed
                embed_prefix_past = torch.where(
                    is_pad_prefix_past.unsqueeze(-1),
                    self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_past),
                    embed_prefix_past,
                )

                encoder_in_tokens.append(embed_prefix_past.permute(1, 0, 2))  # [K, B, dim]
                encoder_in_pos_embed.append(
                    self.encoder_1d_feature_pos_embed.weight[pos_1d_idx : pos_1d_idx + length_prefix_past].unsqueeze(1)
                )  # [K, 1, dim]
                pos_1d_idx += length_prefix_past

            # 6. Committed pending actions - learnable pos embed
            action_prefix_future = inference_action[:, length_prefix_past:]
            # INFERENCE_ACTION_IS_PAD already encodes which positions are valid
            is_pad_prefix_future = inference_action_is_pad[:, length_prefix_past:]

            embed_prefix_future = self.encoder_action_prefix_input_proj(action_prefix_future)

            # Replace invalid positions with learnable action_prefix_pad_embed
            embed_prefix_future = torch.where(
                is_pad_prefix_future.unsqueeze(-1),
                self.action_prefix_pad_embed[None, None, :].expand_as(embed_prefix_future),
                embed_prefix_future,
            )

            encoder_in_tokens.append(embed_prefix_future.permute(1, 0, 2))  # [D, B, dim]
            encoder_in_pos_embed.append(
                self.encoder_1d_feature_pos_embed.weight[pos_1d_idx : pos_1d_idx + length_prefix_future].unsqueeze(1)
            )  # [D, 1, dim]

        # Assemble encoder input
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)  # [S_total, B, dim]
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)  # [S_total, 1, dim]
        encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size, -1)  # [S_total, B, dim]

        # ===== Inference: Encoder-Decoder forward (no masks, single delay) =====
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=device,
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

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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

    def forward(
        self,
        x,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
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
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x, attn_mask=self_attn_mask)[0]
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
            attn_mask=cross_attn_mask,
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
