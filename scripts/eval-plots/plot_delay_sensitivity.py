#!/usr/bin/env python
"""Plot ACTSmooth sensitivity to action prefix delay and translation offsets.

This script tests how the policy responds to:
1. Different delay values (0 to max_delay) with ground truth action prefix
2. Translated/offset action prefixes at max delay

Usage:
    python scripts/eval-plots/plot_delay_sensitivity.py \
        --path_policy=outputs/model/pretrained_model \
        --id_repo_dataset=giacomoran/cube_hand_guided \
        --idx_episode=0 \
        --idx_joint=0 \
        --n_observations=5
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.utils import init_logging
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    ggtitle,
    labs,
    scale_color_cmap,
    scale_color_gradient2,
)

from utils_plotting import save_plot, theme_publication

# Import ACTSmooth for type registration and loading
from lerobot_policy_act_smooth import ACTSmoothConfig, ACTSmoothPolicy  # noqa: F401


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ConfigDelaySensitivity:
    """Configuration for delay sensitivity plotting."""

    path_policy: str = ""
    id_repo_dataset: str = "giacomoran/cube_hand_guided"
    idx_episode: int = 0
    idx_joint: int = 0
    n_observations: int = 5
    seed: int = 42

    # Explicit observation indices (comma-separated). Overrides n_observations/seed.
    indices_observation: str = ""

    # Translation experiment parameters
    n_translations: int = 5
    max_translation: float = 0.3

    # Output
    path_output: str = "outputs/plots/delay_sensitivity.png"

    # Device and data loading
    device: str | None = None
    backend_video: str = "pyav"

    def __post_init__(self):
        if not self.path_policy:
            raise ValueError("--path_policy must be provided")


# ============================================================================
# Dataset Loading
# ============================================================================


def load_dataset_for_episode(
    id_repo: str,
    idx_episode: int,
    max_delay: int,
    chunk_size: int,
    fps: int,
    backend_video: str,
    meta_dataset: LeRobotDatasetMetadata,
) -> LeRobotDataset:
    """Load dataset with extended delta_timestamps for analysis.

    For delay=d, we need:
    - Prefix: actions at t, ..., t+d-1 (indices 0 to d-1 in loaded data)
    - Ground truth: actions at t+d, ..., t+d+chunk_size-1 (indices d to d+chunk_size-1)

    So we need indices 0 to max_delay+chunk_size-1 (total: max_delay+chunk_size actions).

    Args:
        id_repo: HuggingFace dataset repo ID.
        idx_episode: Episode index to load.
        max_delay: Maximum delay from policy config.
        chunk_size: Chunk size from policy config.
        fps: Dataset FPS.
        backend_video: Video backend to use.
        meta_dataset: Dataset metadata.

    Returns:
        LeRobotDataset with extended action delta_timestamps.
    """
    delta_timestamps = {
        "action": [i / fps for i in range(0, max_delay + chunk_size)],
        "observation.state": [0],
    }

    for key in meta_dataset.features:
        if key.startswith("observation.images."):
            delta_timestamps[key] = [0]

    dataset = LeRobotDataset(
        id_repo,
        episodes=[idx_episode],
        delta_timestamps=delta_timestamps,
        video_backend=backend_video,
    )

    return dataset


# ============================================================================
# Sampling
# ============================================================================


def sample_indices_observation(
    idx_from_episode: int,
    idx_to_episode: int,
    max_delay: int,
    chunk_size: int,
    n_samples: int,
    seed: int,
) -> list[int]:
    """Sample observation indices from valid range in episode.

    Valid range ensures enough future actions for all delays up to max_delay + chunk_size.

    Args:
        idx_from_episode: Episode start index in dataset.
        idx_to_episode: Episode end index in dataset.
        max_delay: Maximum delay.
        chunk_size: Action chunk size.
        n_samples: Number of samples to draw.
        seed: Random seed.

    Returns:
        List of sampled observation indices (global dataset indices).
    """
    idx_valid_from = idx_from_episode
    idx_valid_to = idx_to_episode - max_delay - chunk_size

    if idx_valid_to <= idx_valid_from:
        raise ValueError(f"Episode too short for sampling: valid range [{idx_valid_from}, {idx_valid_to}) is empty")

    rng = np.random.RandomState(seed)
    indices_sampled = rng.choice(
        range(idx_valid_from, idx_valid_to),
        size=min(n_samples, idx_valid_to - idx_valid_from),
        replace=False,
    )

    return sorted(indices_sampled.tolist())


# ============================================================================
# Prefix Computation
# ============================================================================


def compute_prefix(
    sample: dict,
    delay: int,
    device: torch.device,
    offset: float = 0.0,
) -> torch.Tensor | None:
    """Compute action prefix from episode data.

    The prefix consists of the FIRST `delay` actions (t to t+delay-1).

    Args:
        sample: Dataset sample with action delta_timestamps starting at 0/fps.
        delay: Current delay value.
        device: Device to use.
        offset: Offset to add to prefix values (applied proportionally).

    Returns:
        Absolute action prefix [1, delay, action_dim] or None if delay=0.
    """
    if delay == 0:
        return None

    actions_all = sample["action"]  # [max_delay + chunk_size, action_dim]
    prefix_absolute = actions_all[:delay]  # [delay, action_dim]

    if offset != 0.0:
        offsets_proportional = torch.linspace(1.0 / delay, 1.0, delay).unsqueeze(1).to(prefix_absolute.device)
        prefix_absolute = prefix_absolute + offset * offsets_proportional

    return prefix_absolute.unsqueeze(0).to(device)  # [1, delay, action_dim]


# ============================================================================
# Inference
# ============================================================================


def predict_with_prefix(
    policy: ACTSmoothPolicy,
    preprocessor,
    postprocessor,
    sample: dict,
    device: torch.device,
    action_prefix: torch.Tensor | None,
) -> np.ndarray:
    """Predict action chunk with given prefix.

    Args:
        policy: Policy instance.
        preprocessor: Preprocessor pipeline.
        postprocessor: Postprocessor pipeline for unnormalization.
        sample: Dataset sample.
        device: Device to use.
        action_prefix: Absolute action prefix [1, delay, action_dim] or None.

    Returns:
        Predicted absolute action chunk [chunk_size, action_dim].
    """
    state_t = sample["observation.state"].squeeze(0)  # [state_dim]

    batch = {"observation.state": state_t.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[-1]  # Get image at t (last frame)
            batch[key] = img.unsqueeze(0).to(device)

    batch = preprocessor(batch)

    # IMPORTANT: Normalize action_prefix to match training format.
    # During training, batch[ACTION] is normalized by the preprocessor.
    # At inference, we must normalize the action_prefix the same way.
    # We use the normalizer step directly since process_action goes through
    # all pipeline steps (including observation-required ones).
    if action_prefix is not None:
        for step in preprocessor.steps:
            if isinstance(step, NormalizerProcessorStep):
                action_prefix = step._normalize_action(action_prefix, inverse=False)
                break

    with torch.no_grad():
        actions_normalized = policy.predict_action_chunk(batch, action_prefix=action_prefix)
        # Unnormalize: postprocessor expects [B, chunk_size, action_dim]
        actions = postprocessor(actions_normalized)

    return actions[0].cpu().numpy()


# ============================================================================
# Experiments
# ============================================================================


def run_experiment_delay(
    policy: ACTSmoothPolicy,
    preprocessor,
    postprocessor,
    sample: dict,
    max_delay: int,
    chunk_size: int,
    device: torch.device,
) -> dict:
    """Run delay experiment for a single observation.

    Args:
        policy: Policy instance.
        preprocessor: Preprocessor pipeline.
        postprocessor: Postprocessor pipeline for unnormalization.
        sample: Dataset sample.
        max_delay: Maximum delay.
        chunk_size: Action chunk size.
        device: Device to use.

    Returns:
        Dictionary with delays, predictions, prefixes, and ground truths.
    """
    delays = list(range(max_delay + 1))
    predictions = []
    prefixes = []
    truths_ground = []

    actions_all = sample["action"]  # [max_delay + chunk_size, action_dim]

    for delay in delays:
        action_prefix = compute_prefix(sample, delay, device)

        if delay > 0:
            prefix_absolute = actions_all[:delay]
            prefixes.append(prefix_absolute.cpu().numpy())
        else:
            prefixes.append(None)

        # Ground truth for delay=d is actions_all[d:d+chunk_size]
        truth_ground = actions_all[delay : delay + chunk_size]
        truths_ground.append(truth_ground.cpu().numpy())

        pred = predict_with_prefix(policy, preprocessor, postprocessor, sample, device, action_prefix)
        predictions.append(pred)

    return {
        "delays": delays,
        "predictions": predictions,
        "prefixes": prefixes,
        "truths_ground": truths_ground,
    }


def run_experiment_translation(
    policy: ACTSmoothPolicy,
    preprocessor,
    postprocessor,
    sample: dict,
    max_delay: int,
    chunk_size: int,
    n_translations: int,
    max_translation: float,
    device: torch.device,
) -> dict:
    """Run translation experiment for a single observation.

    Uses max_delay as the fixed delay and varies the prefix offset.

    Args:
        policy: Policy instance.
        preprocessor: Preprocessor pipeline.
        postprocessor: Postprocessor pipeline for unnormalization.
        sample: Dataset sample.
        max_delay: Maximum delay (used as fixed delay).
        chunk_size: Action chunk size.
        n_translations: Number of translation levels.
        max_translation: Maximum translation offset.
        device: Device to use.

    Returns:
        Dictionary with offsets, predictions, prefixes, and ground truth.
    """
    offsets = np.linspace(-max_translation, max_translation, n_translations).tolist()
    predictions = []
    prefixes = []
    delay = max_delay

    actions_all = sample["action"]  # [max_delay + chunk_size, action_dim]
    truth_ground = actions_all[delay : delay + chunk_size].cpu().numpy()

    for offset in offsets:
        action_prefix = compute_prefix(sample, delay, device, offset=offset)

        prefix_absolute = actions_all[:delay].cpu().numpy()
        offsets_proportional = np.linspace(1.0 / delay, 1.0, delay).reshape(-1, 1)
        prefix_with_offset = prefix_absolute + offset * offsets_proportional
        prefixes.append(prefix_with_offset)

        pred = predict_with_prefix(policy, preprocessor, postprocessor, sample, device, action_prefix)
        predictions.append(pred)

    return {
        "offsets": offsets,
        "predictions": predictions,
        "prefixes": prefixes,
        "truth_ground": truth_ground,
    }


# ============================================================================
# DataFrame Building
# ============================================================================


def build_dataframe_experiments(
    indices_obs: list[int],
    results_delay: list[dict],
    results_translation: list[dict],
    samples: list[dict],
    max_delay: int,
    chunk_size: int,
    idx_joint: int,
    fps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transform experiment results into tidy DataFrames for plotnine.

    Args:
        indices_obs: List of observation indices.
        results_delay: List of delay experiment results.
        results_translation: List of translation experiment results.
        samples: List of dataset samples.
        max_delay: Maximum delay.
        chunk_size: Action chunk size.
        idx_joint: Which joint to plot.
        fps: Dataset FPS.

    Returns:
        Tuple of (df_delay, df_truth_delay, df_trans, df_truth_trans).
    """
    rows_delay = []
    rows_truth_delay = []
    rows_trans = []
    rows_truth_trans = []

    for idx_row, (idx_obs, sample, res_delay, res_trans) in enumerate(
        zip(indices_obs, samples, results_delay, results_translation)
    ):
        state_t = sample["observation.state"]
        if state_t.dim() > 1:
            state_t = state_t.squeeze(0)
        value_state = state_t[idx_joint].item()

        # Delay experiment
        for delay, pred, truth in zip(res_delay["delays"], res_delay["predictions"], res_delay["truths_ground"]):
            time_axis = np.arange(chunk_size) / fps + delay / fps

            for t, val in zip(time_axis, pred[:, idx_joint]):
                rows_delay.append(
                    {
                        "idx_obs": idx_obs,
                        "delay": delay,
                        "time_s": t,
                        "value_position": val,
                        "value_state": value_state,
                    }
                )

            for t, val in zip(time_axis, truth[:, idx_joint]):
                rows_truth_delay.append(
                    {
                        "idx_obs": idx_obs,
                        "delay": delay,
                        "time_s": t,
                        "value_position": val,
                    }
                )

        # Translation experiment
        truth = res_trans["truth_ground"]
        time_axis = np.arange(chunk_size) / fps + max_delay / fps

        for t, val in zip(time_axis, truth[:, idx_joint]):
            rows_truth_trans.append(
                {
                    "idx_obs": idx_obs,
                    "time_s": t,
                    "value_position": val,
                    "value_state": value_state,
                }
            )

        for offset, pred, prefix in zip(res_trans["offsets"], res_trans["predictions"], res_trans["prefixes"]):
            for t, val in zip(time_axis, pred[:, idx_joint]):
                rows_trans.append(
                    {
                        "idx_obs": idx_obs,
                        "offset": offset,
                        "time_s": t,
                        "value_position": val,
                        "value_state": value_state,
                        "is_prefix": False,
                    }
                )

            # Add prefix points
            time_prefix = np.arange(0, max_delay) / fps
            for t, val in zip(time_prefix, prefix[:, idx_joint]):
                rows_trans.append(
                    {
                        "idx_obs": idx_obs,
                        "offset": offset,
                        "time_s": t,
                        "value_position": val,
                        "value_state": value_state,
                        "is_prefix": True,
                    }
                )

    return (
        pd.DataFrame(rows_delay),
        pd.DataFrame(rows_truth_delay),
        pd.DataFrame(rows_trans),
        pd.DataFrame(rows_truth_trans),
    )


# ============================================================================
# Plotting
# ============================================================================


def create_plot(
    df_delay: pd.DataFrame,
    df_truth_delay: pd.DataFrame,
    df_trans: pd.DataFrame,
    df_truth_trans: pd.DataFrame,
    idx_joint: int,
    max_delay: int,
    path_output: Path,
) -> None:
    """Create and save the delay sensitivity plot using plotnine.

    Args:
        df_delay: DataFrame with delay experiment predictions.
        df_truth_delay: DataFrame with delay experiment ground truth.
        df_trans: DataFrame with translation experiment data.
        df_truth_trans: DataFrame with translation experiment ground truth.
        idx_joint: Which joint was plotted.
        max_delay: Maximum delay value.
        path_output: Path to save the plot.
    """
    # Get unique observations for state points
    df_state = df_delay[["idx_obs", "value_state"]].drop_duplicates()
    df_state["time_s"] = 0

    # Delay experiment plot
    plot_delay = (
        ggplot()
        + geom_line(
            data=df_truth_delay[df_truth_delay["delay"] == 0],
            mapping=aes(x="time_s", y="value_position"),
            linetype="dashed",
            color="black",
            alpha=0.6,
            size=0.8,
        )
        + geom_line(
            data=df_delay,
            mapping=aes(x="time_s", y="value_position", color="factor(delay)", group="delay"),
            size=0.6,
            alpha=0.8,
        )
        + geom_point(data=df_state, mapping=aes(x="time_s", y="value_state"), size=3, color="black")
        + facet_wrap("~idx_obs", ncol=1, scales="free_y")
        + scale_color_cmap("viridis")
        + labs(x="Time from observation (s)", y=f"Joint {idx_joint} (rad)", color="Delay")
        + ggtitle("Delay Experiment")
        + theme_publication()
    )

    # Translation experiment plot
    df_trans_pred = df_trans[~df_trans["is_prefix"]]
    df_trans_prefix = df_trans[df_trans["is_prefix"]]

    plot_trans = (
        ggplot()
        + geom_line(
            data=df_truth_trans,
            mapping=aes(x="time_s", y="value_position"),
            linetype="dashed",
            color="black",
            alpha=0.6,
            size=0.8,
        )
        + geom_line(
            data=df_trans_pred,
            mapping=aes(x="time_s", y="value_position", color="offset", group="offset"),
            size=0.6,
            alpha=0.8,
        )
        + geom_line(
            data=df_trans_prefix,
            mapping=aes(x="time_s", y="value_position", color="offset", group="offset"),
            linetype="dotted",
            size=0.6,
            alpha=0.6,
        )
        + geom_point(data=df_state, mapping=aes(x="time_s", y="value_state"), size=3, color="black")
        + facet_wrap("~idx_obs", ncol=1, scales="free_y")
        + scale_color_gradient2(low="blue", mid="gray", high="red", midpoint=0)
        + labs(x="Time from observation (s)", y=f"Joint {idx_joint} (rad)", color="Offset")
        + ggtitle(f"Translation Experiment (delay={max_delay})")
        + theme_publication()
    )

    # Save both plots
    n_obs = df_delay["idx_obs"].nunique()
    height_per_obs = 3
    height = n_obs * height_per_obs + 1

    path_delay = path_output.parent / f"{path_output.stem}_delay{path_output.suffix}"
    path_trans = path_output.parent / f"{path_output.stem}_translation{path_output.suffix}"

    save_plot(plot_delay, path_delay, width=8, height=height)
    save_plot(plot_trans, path_trans, width=8, height=height)

    logging.info(f"Saved delay plot to {path_delay}")
    logging.info(f"Saved translation plot to {path_trans}")


# ============================================================================
# Main
# ============================================================================


@draccus.wrap()
def main(cfg: ConfigDelaySensitivity):
    """Main entry point for delay sensitivity plotting."""
    init_logging()

    logging.info(f"Setting seed: {cfg.seed}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    logging.info(f"Using device: {device}")

    logging.info(f"Loading policy from {cfg.path_policy}")
    policy = ACTSmoothPolicy.from_pretrained(cfg.path_policy)
    policy.to(device)
    policy.eval()

    max_delay = policy.config.max_delay
    chunk_size = policy.config.chunk_size
    logging.info(f"Policy config: max_delay={max_delay}, chunk_size={chunk_size}")

    overrides_device = {"device_processor": {"device": str(device)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.path_policy,
        preprocessor_overrides=overrides_device,
        postprocessor_overrides=overrides_device,
    )

    logging.info(f"Loading dataset metadata from {cfg.id_repo_dataset}")
    meta_dataset = LeRobotDatasetMetadata(cfg.id_repo_dataset)
    fps = meta_dataset.fps

    idx_from = meta_dataset.episodes["dataset_from_index"][cfg.idx_episode]
    idx_to = meta_dataset.episodes["dataset_to_index"][cfg.idx_episode]
    idx_from = int(idx_from.item() if hasattr(idx_from, "item") else idx_from)
    idx_to = int(idx_to.item() if hasattr(idx_to, "item") else idx_to)
    logging.info(f"Episode {cfg.idx_episode}: indices [{idx_from}, {idx_to})")

    logging.info("Loading dataset with extended delta_timestamps")
    dataset = load_dataset_for_episode(
        cfg.id_repo_dataset,
        cfg.idx_episode,
        max_delay,
        chunk_size,
        fps,
        cfg.backend_video,
        meta_dataset,
    )

    if cfg.indices_observation:
        indices_obs = [int(x.strip()) for x in cfg.indices_observation.split(",")]
        logging.info(f"Using explicit observation indices: {indices_obs}")
    else:
        logging.info(f"Sampling {cfg.n_observations} observations")
        indices_obs = sample_indices_observation(
            idx_from,
            idx_to,
            max_delay,
            chunk_size,
            cfg.n_observations,
            cfg.seed,
        )
        logging.info(f"Sampled indices: {indices_obs}")

    logging.info("Running experiments...")
    results_delay = []
    results_translation = []
    samples = []

    for idx_obs in indices_obs:
        logging.info(f"  Processing observation {idx_obs}")
        sample = dataset[idx_obs]
        samples.append(sample)

        res_delay = run_experiment_delay(policy, preprocessor, postprocessor, sample, max_delay, chunk_size, device)
        results_delay.append(res_delay)

        res_trans = run_experiment_translation(
            policy,
            preprocessor,
            postprocessor,
            sample,
            max_delay,
            chunk_size,
            cfg.n_translations,
            cfg.max_translation,
            device,
        )
        results_translation.append(res_trans)

    logging.info("Building DataFrames...")
    df_delay, df_truth_delay, df_trans, df_truth_trans = build_dataframe_experiments(
        indices_obs,
        results_delay,
        results_translation,
        samples,
        max_delay,
        chunk_size,
        cfg.idx_joint,
        fps,
    )

    path_output = Path(cfg.path_output)
    logging.info("Creating plots...")
    create_plot(
        df_delay,
        df_truth_delay,
        df_trans,
        df_truth_trans,
        cfg.idx_joint,
        max_delay,
        path_output,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
