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
    """Configuration class for the ACTSmooth policy with observation dropout support."""

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    length_prefix_past: int = 0
    length_prefix_future: int = 1
    max_delay: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Observation dropout: probability of dropping all observation tokens during training.
    # When dropped, image and state tokens are zeroed out, forcing the model to rely on the prefix.
    p_drop_obs: float = 0.0

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
        if not 0.0 <= self.p_drop_obs <= 1.0:
            raise ValueError(f"p_drop_obs must be in [0, 1]. Got {self.p_drop_obs}.")

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
        """Load config with backward compatibility for legacy ACTSmooth fields."""
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
        start = -self.length_prefix_past
        end = self.length_prefix_future + self.chunk_size
        return list(range(start, end))

    @property
    def reward_delta_indices(self) -> None:
        return None
