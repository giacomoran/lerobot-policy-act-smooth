#!/usr/bin/env python
"""Plot ACTSmooth sensitivity to action prefix.

Experiments:
1. Sensitivity: vary future prefix length d in {1..D}, full past prefix, no offset
2. Translation future: fix d=D, full past prefix, vary offset on future prefix
3. Translation past: fix d=D, full future prefix, vary offset on past prefix

Usage:
    python scripts/analyze/analyze_delay_sensitivity.py \
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
from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import to_hex
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    ggtitle,
    labs,
    scale_color_gradient2,
    scale_color_manual,
)

from utils_plotting import save_plot, theme_publication

# Import ACTSmooth for type registration and loading
from lerobot_policy_act_smooth import ACTSmoothConfig, ACTSmoothPolicy  # noqa: F401
from lerobot_policy_act_smooth.modeling_act_smooth import INFERENCE_ACTION, INFERENCE_ACTION_IS_PAD


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
# Core
# ============================================================================


def load_dataset_for_episode(
    id_repo: str,
    idx_episode: int,
    length_prefix_past: int,
    length_prefix_future: int,
    chunk_size: int,
    fps: int,
    backend_video: str,
    meta_dataset: LeRobotDatasetMetadata,
) -> LeRobotDataset:
    """Load dataset with action delta_timestamps covering [t-K, t+D+C).

    The loaded action sequence layout is:
        [past(K) | future(D) | target(C)]
    where index K corresponds to t=0 (observation time).
    """
    delta_timestamps = {
        "action": [i / fps for i in range(-length_prefix_past, length_prefix_future + chunk_size)],
        "observation.state": [0],
    }
    for key in meta_dataset.features:
        if key.startswith("observation.images."):
            delta_timestamps[key] = [0]

    return LeRobotDataset(
        id_repo,
        episodes=[idx_episode],
        delta_timestamps=delta_timestamps,
        video_backend=backend_video,
    )


def predict_for_condition(
    policy: ACTSmoothPolicy,
    preprocessor,
    postprocessor,
    sample: dict,
    device: torch.device,
    length_prefix_past: int,
    delay: int,
    length_past: int,
    offset_future: float = 0.0,
    offset_past: float = 0.0,
) -> np.ndarray:
    """Run one inference with specified prefix condition.

    Args:
        sample: Dataset sample. sample["action"] has layout [past(K) | future(D) | target(C)].
        length_prefix_past: K from policy config (offset to t=0 in action array).
        delay: Future prefix length d (1..D). Determines which d actions starting at t=0 are given.
        length_past: How many past actions to use (0..K).
        offset_future: Constant additive offset applied to the future prefix.
        offset_past: Constant additive offset applied to the past prefix.

    Returns:
        Predicted action chunk [chunk_size, action_dim].
    """
    K = length_prefix_past
    actions_all = sample["action"]

    # Future prefix: actions at [t=0, t=d-1], i.e. indices [K, K+delay)
    prefix_future = actions_all[K : K + delay].clone()
    if offset_future != 0.0:
        prefix_future = prefix_future + offset_future
    prefix_future = prefix_future.unsqueeze(0).to(device)

    # Past prefix: most recent length_past actions before t=0, i.e. indices [K-length_past, K)
    if length_past > 0:
        prefix_past = actions_all[K - length_past : K].clone()
        if offset_past != 0.0:
            prefix_past = prefix_past + offset_past
        prefix_past = prefix_past.unsqueeze(0).to(device)
    else:
        prefix_past = torch.zeros(1, 0, actions_all.shape[-1], device=device)

    # Build observation batch
    state_t = sample["observation.state"].squeeze(0)
    batch = {"observation.state": state_t.unsqueeze(0).to(device)}
    for key in sample:
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            img = sample[key]
            if img.ndim == 4:
                img = img[-1]
            batch[key] = img.unsqueeze(0).to(device)

    batch = preprocessor(batch)

    # Remove non-observation keys added by the preprocessor (e.g. "action",
    # "next.reward") — they confuse the model's training-vs-inference detection.
    for key in list(batch):
        if not key.startswith("observation"):
            del batch[key]

    # Normalize prefixes to match training format
    for step in preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            prefix_future = step._normalize_action(prefix_future, inverse=False)
            if prefix_past.shape[1] > 0:
                prefix_past = step._normalize_action(prefix_past, inverse=False)
            break

    inference_action, inference_action_is_pad = policy.build_inference_action_prefix(
        action_prefix_future=prefix_future,
        action_prefix_past=prefix_past,
    )
    batch[INFERENCE_ACTION] = inference_action
    batch[INFERENCE_ACTION_IS_PAD] = inference_action_is_pad

    with torch.no_grad():
        actions = postprocessor(policy.predict_action_chunk(batch))

    return actions[0].cpu().numpy()


# ============================================================================
# Main
# ============================================================================


@draccus.wrap()
def main(cfg: ConfigDelaySensitivity):
    init_logging()

    logging.info(f"Setting seed: {cfg.seed}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading policy from {cfg.path_policy}")
    policy = ACTSmoothPolicy.from_pretrained(cfg.path_policy)
    policy.to(device)
    policy.eval()

    K = policy.config.length_prefix_past  # past prefix length
    D = policy.config.length_prefix_future  # max future prefix / delay
    C = policy.config.chunk_size
    logging.info(f"Policy config: length_prefix_past={K}, length_prefix_future={D}, chunk_size={C}")

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

    dataset = load_dataset_for_episode(
        cfg.id_repo_dataset, cfg.idx_episode, K, D, C, fps, cfg.backend_video, meta_dataset
    )

    # Sample or parse observation indices
    if cfg.indices_observation:
        indices_obs = [int(x.strip()) for x in cfg.indices_observation.split(",")]
    else:
        # Valid range: enough room for past actions and future actions + target
        idx_valid_from = idx_from + K
        idx_valid_to = idx_to - D - C
        if idx_valid_to <= idx_valid_from:
            raise ValueError(f"Episode too short: valid range [{idx_valid_from}, {idx_valid_to}) is empty")
        rng = np.random.RandomState(cfg.seed)
        indices_obs = sorted(
            rng.choice(
                range(idx_valid_from, idx_valid_to),
                size=min(cfg.n_observations, idx_valid_to - idx_valid_from),
                replace=False,
            ).tolist()
        )
    logging.info(f"Observation indices: {indices_obs}")

    # ------------------------------------------------------------------
    # Run all experiments, building DataFrame rows directly
    # ------------------------------------------------------------------
    rows_pred = []
    rows_truth = []

    offsets_translation = np.linspace(-cfg.max_translation, cfg.max_translation, cfg.n_translations).tolist()

    for idx_obs in indices_obs:
        logging.info(f"  Processing observation {idx_obs}")
        sample = dataset[idx_obs]
        actions_all = sample["action"]  # [K + D + C, action_dim]

        state_t = sample["observation.state"]
        if state_t.dim() > 1:
            state_t = state_t.squeeze(0)
        value_state = state_t[cfg.idx_joint].item()

        def add_prediction(
            experiment: str,
            condition,
            pred: np.ndarray,
            delay: int,
            prefix_future=None,
            prefix_past=None,
        ):
            """Append prediction and prefix rows for one condition."""
            # Prediction: C points starting at delay/fps
            time_pred = np.arange(C) / fps + delay / fps
            for t, val in zip(time_pred, pred[:, cfg.idx_joint]):
                rows_pred.append(
                    {
                        "idx_obs": idx_obs,
                        "experiment": experiment,
                        "condition": condition,
                        "time_s": t,
                        "value_position": float(val),
                        "value_state": value_state,
                        "line_type": "prediction",
                    }
                )

            # Future prefix at [0, 1/fps, ..., (delay-1)/fps] + bridge to first prediction
            if prefix_future is not None:
                time_pf = np.arange(len(prefix_future)) / fps
                for t, val in zip(time_pf, prefix_future[:, cfg.idx_joint]):
                    rows_pred.append(
                        {
                            "idx_obs": idx_obs,
                            "experiment": experiment,
                            "condition": condition,
                            "time_s": t,
                            "value_position": float(val),
                            "value_state": value_state,
                            "line_type": "prefix_future",
                        }
                    )
                # Bridge: connect last prefix point to first prediction point
                rows_pred.append(
                    {
                        "idx_obs": idx_obs,
                        "experiment": experiment,
                        "condition": condition,
                        "time_s": delay / fps,
                        "value_position": float(pred[0, cfg.idx_joint]),
                        "value_state": value_state,
                        "line_type": "prefix_future",
                    }
                )

            # Past prefix at [-n/fps, ..., -1/fps] + bridge to first future prefix
            if prefix_past is not None and len(prefix_past) > 0:
                n_pp = len(prefix_past)
                time_pp = np.arange(-n_pp, 0) / fps
                for t, val in zip(time_pp, prefix_past[:, cfg.idx_joint]):
                    rows_pred.append(
                        {
                            "idx_obs": idx_obs,
                            "experiment": experiment,
                            "condition": condition,
                            "time_s": t,
                            "value_position": float(val),
                            "value_state": value_state,
                            "line_type": "prefix_past",
                        }
                    )
                # Bridge: connect last past prefix to first future prefix (or prediction start)
                if prefix_future is not None and len(prefix_future) > 0:
                    bridge_val = float(prefix_future[0, cfg.idx_joint])
                else:
                    bridge_val = float(pred[0, cfg.idx_joint])
                rows_pred.append(
                    {
                        "idx_obs": idx_obs,
                        "experiment": experiment,
                        "condition": condition,
                        "time_s": 0.0,
                        "value_position": bridge_val,
                        "value_state": value_state,
                        "line_type": "prefix_past",
                    }
                )

        # Reference ground truth from t=-K (shared across all plots)
        truth_ref = actions_all[0 : K + D + C].cpu().numpy()
        time_ref = np.arange(-K, D + C) / fps
        for t, val in zip(time_ref, truth_ref[:, cfg.idx_joint]):
            rows_truth.append({"idx_obs": idx_obs, "time_s": t, "value_position": float(val)})

        prefix_past_gt = actions_all[0:K].cpu().numpy() if K > 0 else None

        # --- Experiment 1: Sensitivity (vary future prefix length) ---
        for delay in range(1, D + 1):
            pred = predict_for_condition(policy, preprocessor, postprocessor, sample, device, K, delay, K)
            prefix_future_vis = actions_all[K : K + delay].cpu().numpy()
            add_prediction(
                "sensitivity", delay, pred, delay, prefix_future=prefix_future_vis, prefix_past=prefix_past_gt
            )

        # --- Experiment 2: Future prefix translation ---
        for offset in offsets_translation:
            pred = predict_for_condition(
                policy, preprocessor, postprocessor, sample, device, K, D, K, offset_future=offset
            )
            prefix_future_vis = actions_all[K : K + D].cpu().numpy() + offset
            add_prediction(
                "translation_future", offset, pred, D, prefix_future=prefix_future_vis, prefix_past=prefix_past_gt
            )

        # --- Experiment 3: Past prefix translation (offset both past and future together) ---
        if K > 0:
            for offset in offsets_translation:
                pred = predict_for_condition(
                    policy,
                    preprocessor,
                    postprocessor,
                    sample,
                    device,
                    K,
                    D,
                    K,
                    offset_past=offset,
                    offset_future=offset,
                )
                prefix_past_vis = actions_all[0:K].cpu().numpy() + offset
                prefix_future_vis = actions_all[K : K + D].cpu().numpy() + offset
                add_prediction(
                    "translation_past",
                    offset,
                    pred,
                    D,
                    prefix_future=prefix_future_vis,
                    prefix_past=prefix_past_vis,
                )

    df_pred = pd.DataFrame(rows_pred)
    df_truth = pd.DataFrame(rows_truth)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    logging.info("Creating plots...")
    path_output = Path(cfg.path_output)

    # State points at t=0
    df_state = df_pred[["idx_obs", "value_state"]].drop_duplicates()
    df_state["time_s"] = 0.0

    n_obs = df_pred["idx_obs"].nunique()
    height = n_obs * 3 + 1

    # Qualitative color helper for discrete experiments
    tab10_colors = mpl_colormaps["tab10"].colors

    def colors_qualitative(n: int) -> list[str]:
        """Return n distinct qualitative colors from tab10."""
        return [to_hex(tab10_colors[i]) for i in range(n)]

    # --- Sensitivity plot (vary future prefix length) ---
    dp = df_pred[df_pred["experiment"] == "sensitivity"].copy()
    dp["condition"] = dp["condition"].astype(int)
    dp_pred = dp[dp["line_type"] == "prediction"]
    dp_pf = dp[dp["line_type"] == "prefix_future"]
    dp_pp = dp[dp["line_type"] == "prefix_past"].drop_duplicates(subset=["idx_obs", "time_s", "value_position"])
    colors_delay = colors_qualitative(dp_pred["condition"].nunique())

    plot_sensitivity = (
        ggplot()
        + geom_line(df_truth, aes("time_s", "value_position"), linetype="dashed", color="black", alpha=0.6, size=0.8)
        + geom_point(dp_pp, aes("time_s", "value_position"), color="gray", size=2, alpha=0.7, show_legend=False)
        + geom_line(
            dp_pf,
            aes("time_s", "value_position", color="factor(condition)", group="condition"),
            linetype="dotted",
            size=0.8,
            alpha=0.7,
            show_legend=False,
        )
        + geom_point(
            dp_pf,
            aes("time_s", "value_position", color="factor(condition)"),
            size=2,
            alpha=0.7,
            show_legend=False,
        )
        + geom_line(
            dp_pred,
            aes("time_s", "value_position", color="factor(condition)", group="condition"),
            size=0.6,
            alpha=0.8,
        )
        + geom_point(df_state, aes("time_s", "value_state"), size=3, color="black")
        + facet_wrap("~idx_obs", ncol=1, scales="free_y")
        + scale_color_manual(values=colors_delay)
        + labs(x="Time from observation (s)", y=f"Joint {cfg.idx_joint}", color="Delay")
        + ggtitle("Sensitivity to Future Prefix Length (dotted=prefix)")
        + theme_publication()
    )
    path_sensitivity = path_output.parent / f"{path_output.stem}_sensitivity{path_output.suffix}"
    save_plot(plot_sensitivity, path_sensitivity, width=8, height=height)
    logging.info(f"Saved {path_sensitivity}")

    # --- Future translation plot ---
    tp = df_pred[df_pred["experiment"] == "translation_future"]
    tp_pred = tp[tp["line_type"] == "prediction"]
    # Combine past + future prefix into one connected line per condition
    tp_prefix = tp[tp["line_type"].isin(["prefix_past", "prefix_future"])]

    plot_trans_future = (
        ggplot()
        + geom_line(df_truth, aes("time_s", "value_position"), linetype="dashed", color="black", alpha=0.6, size=0.8)
        + geom_line(
            tp_prefix,
            aes("time_s", "value_position", color="condition", group="condition"),
            linetype="dotted",
            size=0.6,
            alpha=0.6,
            show_legend=False,
        )
        + geom_point(
            tp_prefix,
            aes("time_s", "value_position", color="condition"),
            size=2,
            alpha=0.7,
            show_legend=False,
        )
        + geom_line(
            tp_pred,
            aes("time_s", "value_position", color="condition", group="condition"),
            size=0.6,
            alpha=0.8,
        )
        + geom_point(df_state, aes("time_s", "value_state"), size=3, color="black")
        + facet_wrap("~idx_obs", ncol=1, scales="free_y")
        + scale_color_gradient2(low="blue", mid="gray", high="red", midpoint=0)
        + labs(x="Time from observation (s)", y=f"Joint {cfg.idx_joint}", color="Offset")
        + ggtitle(f"Future Prefix Translation (delay={D})")
        + theme_publication()
    )
    path_trans_future = path_output.parent / f"{path_output.stem}_translation_future{path_output.suffix}"
    save_plot(plot_trans_future, path_trans_future, width=8, height=height)
    logging.info(f"Saved {path_trans_future}")

    # --- Past translation plot ---
    if K > 0:
        tp2 = df_pred[df_pred["experiment"] == "translation_past"]
        tp2_pred = tp2[tp2["line_type"] == "prediction"]
        # Combine past + future prefix into one connected line per condition
        tp2_prefix = tp2[tp2["line_type"].isin(["prefix_past", "prefix_future"])]

        plot_trans_past = (
            ggplot()
            + geom_line(
                df_truth,
                aes("time_s", "value_position"),
                linetype="dashed",
                color="black",
                alpha=0.6,
                size=0.8,
            )
            + geom_line(
                tp2_prefix,
                aes("time_s", "value_position", color="condition", group="condition"),
                linetype="dotted",
                size=0.6,
                alpha=0.6,
                show_legend=False,
            )
            + geom_point(
                tp2_prefix,
                aes("time_s", "value_position", color="condition"),
                size=2,
                alpha=0.7,
                show_legend=False,
            )
            + geom_line(
                tp2_pred,
                aes("time_s", "value_position", color="condition", group="condition"),
                size=0.6,
                alpha=0.8,
            )
            + geom_point(df_state, aes("time_s", "value_state"), size=3, color="black")
            + facet_wrap("~idx_obs", ncol=1, scales="free_y")
            + scale_color_gradient2(low="blue", mid="gray", high="red", midpoint=0)
            + labs(x="Time from observation (s)", y=f"Joint {cfg.idx_joint}", color="Offset")
            + ggtitle(f"Past Prefix Translation (delay={D})")
            + theme_publication()
        )
        path_trans_past = path_output.parent / f"{path_output.stem}_translation_past{path_output.suffix}"
        save_plot(plot_trans_past, path_trans_past, width=8, height=height)
        logging.info(f"Saved {path_trans_past}")

    logging.info("Done!")


if __name__ == "__main__":
    main()
