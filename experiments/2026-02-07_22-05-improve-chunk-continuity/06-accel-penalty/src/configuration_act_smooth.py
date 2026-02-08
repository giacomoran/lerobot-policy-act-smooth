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
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_smooth")
@dataclass
class ACTSmoothConfig(PreTrainedConfig):
    """Configuration class for the ACTSmooth policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    # Action prefix conditioning for smooth chunk transitions.
    # The model learns to predict action chunks given a prefix of committed actions.
    # This enables smoother real-time inference: instead of waiting for a full chunk to complete,
    # the policy can "continue" from partially-executed chunks.
    #
    # Prefix = [past completed actions] + [committed pending actions]
    #          |---- k actions (fixed) ----|  |---- d actions (variable, d >= 1) ---|
    #
    # - length_prefix_past (k): Number of past (completed) actions to include in prefix.
    #   These provide historical context for continuity without adding delay.
    # - length_prefix_future (D): Maximum number of committed (pending) actions.
    #   During training, d is sampled from {1, ..., D}.
    #
    # Choosing length_prefix_future (D):
    #   The committed actions are t_0, t_1, ..., t_{d-1} where t_0 is at the observation
    #   timestep. t_0 is already being executed when inference starts, so only the remaining
    #   d-1 actions absorb inference latency. You need:
    #
    #       (d - 1) / fps > inference_latency
    #       D >= ceil(inference_latency * fps) + 1
    #
    #   Examples (assuming 35ms inference latency):
    #     10 FPS: D=2 → (2-1)/10 = 100ms > 35ms ✓
    #     30 FPS: D=3 → (3-1)/30 =  67ms > 35ms ✓
    #
    # Example at 30fps with k=4, D=2:
    #   Prefix = [4 past actions, 2 committed] = 6 total (200ms context)
    #   But delay = only 2 (67ms until new chunk starts)
    #   Best of both: rich context for continuity + fast reactivity
    length_prefix_past: int = 0
    length_prefix_future: int = 1
    # Deprecated: kept for backward compatibility with older configs/CLI.
    max_delay: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0
    weight_loss_accel: float = 0.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.max_delay is not None and self.length_prefix_future == 1:
            self.length_prefix_future = self.max_delay
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`")
        if self.length_prefix_future < 1:
            raise ValueError(
                f"length_prefix_future must be >= 1 (t_0 is always committed due to inference latency). "
                f"Got {self.length_prefix_future}. For no delay conditioning, use vanilla ACT instead."
            )
        if self.length_prefix_past < 0:
            raise ValueError(f"length_prefix_past must be >= 0. Got {self.length_prefix_past}.")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ):
        """Load config with backward compatibility for legacy ACTSmooth fields.

        Supports loading older configs that still use `max_delay`.
        """
        # Import here to avoid adding hard dependencies at module import time.
        import draccus
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import CONFIG_NAME
        from huggingface_hub.errors import HfHubHTTPError

        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}") from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")

        with open(config_file) as f:
            config = json.load(f)

        # Remove legacy/registry fields and map deprecated keys.
        config.pop("type", None)
        max_delay = config.pop("max_delay", None)
        if max_delay is not None and "length_prefix_future" not in config:
            config["length_prefix_future"] = max_delay

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_overrides)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        """Indices for loading action sequences relative to observation timestep.

        batch[ACTION] layout (k = length_prefix_past, D = length_prefix_future, C = chunk_size):

            [t_{-k}, ..., t_{-1}, t_0, t_1, ..., t_{D-1}, t_D, ..., t_{D+C-1}]
            |------ k past ------|  |---- D committed ----|  |--- C target ---|

        Indices relative to observation (t_0 = index 0): [-k, ..., -1, 0, 1, ..., D + C - 1]

        Note: t_0 is always committed (delay d >= 1), so prediction target starts at t_d.
        """
        start = -self.length_prefix_past
        end = self.length_prefix_future + self.chunk_size
        return list(range(start, end))

    @property
    def reward_delta_indices(self) -> None:
        return None
