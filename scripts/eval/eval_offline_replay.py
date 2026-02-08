#!/usr/bin/env python
"""Offline evaluation: simulate async_smooth chunk loop on dataset episodes.

Replays dataset episodes through the ACTSmooth policy using ground truth actions
as prefix at each chunk boundary. Measures how smoothly predicted chunks connect
to the ground truth trajectory.

Design:
1. Load policy from checkpoint
2. Load dataset episodes (same dataset used for training)
3. For each episode, walk through frames at policy rate (advancing by n_action_steps):
   - At each chunk boundary timestep `t`:
     - Get observation (state + images) from dataset at timestep `t`
     - Extract ground truth prefix from dataset:
       - Past prefix: ground truth actions at [t - k, ..., t - 1]
       - Future prefix: ground truth actions at [t, ..., t + d - 1]
     - Run inference to get predicted action chunk (chunk_size actions starting at t + d)
   - Record: ground truth prefix + predicted chunk, stitched into continuous action sequence
4. Compute kinematic metrics on the stitched sequence
5. Output: text summary + trajectory/acceleration plots

Why ground truth prefix?
- No compounding prediction errors -- directly measures how well the model connects
  to the true trajectory
- Matches training distribution (model was trained with ground truth prefixes)
- Clean measurement of chunk boundary discontinuity

Usage:
    python scripts/eval/eval_offline_replay.py \
        --path_policy=outputs/lerobot_policy_act_smooth_30fps_smooth_p4f2/checkpoints/last/pretrained_model \
        --id_repo_dataset=giacomoran/lerobot_policy_act_smooth_30fps \
        --indices_episode=0,1,2 \
        --path_output=outputs/results-offline/baseline
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
    geom_vline,
    ggplot,
    ggtitle,
    labs,
)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "analyze"))
from utils_plotting import save_plot, theme_publication

# Import ACTSmooth for type registration and loading
from lerobot_policy_act_smooth import ACTSmoothConfig, ACTSmoothPolicy  # noqa: F401
from lerobot_policy_act_smooth.modeling_act_smooth import INFERENCE_ACTION, INFERENCE_ACTION_IS_PAD


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ConfigOfflineReplay:
    """Configuration for offline replay evaluation."""

    path_policy: str = ""
    id_repo_dataset: str = "giacomoran/lerobot_policy_act_smooth_30fps"
    indices_episode: str = "0"

    # Output stem prefix (no extension).
    # Produces: {path_output}_results.txt, {path_output}_trajectories.png, {path_output}_accelerations.png
    path_output: str = "outputs/results-offline/baseline"

    # Device and data loading
    device: str | None = None
    backend_video: str = "pyav"

    # Overlap blending: advance fewer steps than n_action_steps and blend overlapping predictions.
    # advance=0 means use n_action_steps (no overlap). advance < n_action_steps enables blending.
    advance: int = 0
    blend_overlap: bool = False

    # Seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        if not self.path_policy:
            raise ValueError("--path_policy must be provided")


# ============================================================================
# Dataset Loading
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


# ============================================================================
# Inference
# ============================================================================


def predict_chunk_at_timestep(
    policy: ACTSmoothPolicy,
    preprocessor,
    postprocessor,
    sample: dict,
    device: torch.device,
    length_prefix_past: int,
    length_prefix_future: int,
) -> np.ndarray:
    """Run inference with full ground truth prefix at a given timestep.

    Args:
        sample: Dataset sample. sample["action"] has layout [past(K) | future(D) | target(C)].
        length_prefix_past: K from policy config.
        length_prefix_future: D from policy config.

    Returns:
        Predicted action chunk [chunk_size, action_dim] in original scale.
    """
    K = length_prefix_past
    D = length_prefix_future
    actions_all = sample["action"]

    # Future prefix: actions at [t=0, t=D-1], i.e. indices [K, K+D)
    prefix_future = actions_all[K : K + D].clone().unsqueeze(0).to(device)

    # Past prefix: actions at [t=-K, t=-1], i.e. indices [0, K)
    if K > 0:
        prefix_past = actions_all[:K].clone().unsqueeze(0).to(device)
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

    # Remove non-observation keys added by the preprocessor
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
# Kinematics (reused from analyze_rerun.py)
# ============================================================================


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity and acceleration from position data.

    Args:
        df: Tidy DataFrame sorted by [name_motor, idx_frame].

    Returns:
        Augmented DataFrame with value_velocity and value_acceleration columns.
    """
    df = df.copy().sort_values(["name_motor", "idx_frame"])
    df["value_velocity"] = df.groupby("name_motor")["value_position"].diff()
    df["value_acceleration"] = df.groupby("name_motor")["value_velocity"].diff()
    return df


def compute_scalars_kinematic(df_kinematics: pd.DataFrame) -> dict[str, float]:
    """Compute kinematic scalar measures.

    Returns:
        Dict with uniformity_velocity, uniformity_acceleration, mean_velocity.
    """
    df = df_kinematics.dropna(subset=["value_velocity"])

    variance_velocity = df.groupby("name_motor")["value_velocity"].var()
    uniformity_velocity = float(variance_velocity.max())

    df_accel = df.dropna(subset=["value_acceleration"])
    variance_acceleration = df_accel.groupby("name_motor")["value_acceleration"].var()
    uniformity_acceleration = float(variance_acceleration.max())

    mean_velocity = float(df["value_velocity"].abs().mean())

    return {
        "uniformity_velocity": uniformity_velocity,
        "uniformity_acceleration": uniformity_acceleration,
        "mean_velocity": mean_velocity,
    }


def compute_scalars_kinematic_boundary(
    df_kinematics: pd.DataFrame,
    indices_boundary: set[int],
    window: int = 2,
) -> dict[str, float]:
    """Compute acceleration at boundary vs non-boundary timesteps.

    Args:
        df_kinematics: DataFrame with value_acceleration column.
        indices_boundary: Set of idx_frame values where chunk boundaries occur.
        window: Number of frames around boundary to include.

    Returns:
        Dict with mean_accel_boundary, mean_accel_non_boundary, ratio_boundary_spike.
    """
    df = df_kinematics.dropna(subset=["value_acceleration"]).copy()

    # Expand boundary indices by window
    indices_boundary_expanded = set()
    for idx in indices_boundary:
        for offset in range(-window, window + 1):
            indices_boundary_expanded.add(idx + offset)

    df["is_boundary"] = df["idx_frame"].isin(indices_boundary_expanded)

    df_boundary = df[df["is_boundary"]]
    df_non_boundary = df[~df["is_boundary"]]

    mean_accel_boundary = float(df_boundary["value_acceleration"].abs().mean()) if len(df_boundary) > 0 else 0.0
    mean_accel_non_boundary = (
        float(df_non_boundary["value_acceleration"].abs().mean()) if len(df_non_boundary) > 0 else 0.0
    )

    ratio_boundary_spike = (
        mean_accel_boundary / mean_accel_non_boundary if mean_accel_non_boundary > 0 else float("inf")
    )

    return {
        "mean_accel_boundary": mean_accel_boundary,
        "mean_accel_non_boundary": mean_accel_non_boundary,
        "ratio_boundary_spike": ratio_boundary_spike,
    }


# ============================================================================
# Plotting
# ============================================================================


def create_plot_trajectories(
    df_truth: pd.DataFrame,
    df_predicted: pd.DataFrame,
    indices_boundary: list[int],
    path_output: Path,
) -> None:
    """Create trajectory plot: ground truth line + predicted action dots + chunk boundaries.

    Args:
        df_truth: Ground truth DataFrame with idx_frame, name_motor, value_position.
        df_predicted: Predicted DataFrame with idx_frame, name_motor, value_position, source.
        indices_boundary: Frame indices where chunk boundaries occur.
        path_output: Path to save the plot.
    """
    df_chunk = pd.DataFrame({"idx_frame": indices_boundary})

    n_motors = df_truth["name_motor"].nunique()
    n_rows = (n_motors + 1) // 2
    height = n_rows * 3 + 1
    width = 14

    # Separate prefix and prediction
    df_prefix = df_predicted[df_predicted["source"] == "prefix"]
    df_pred = df_predicted[df_predicted["source"] == "prediction"]

    plot = (
        ggplot()
        + geom_line(
            data=df_truth,
            mapping=aes(x="idx_frame", y="value_position"),
            color="black",
            size=0.5,
            alpha=0.6,
        )
        + geom_point(
            data=df_prefix,
            mapping=aes(x="idx_frame", y="value_position"),
            color="#4DAF4A",
            size=1.5,
            alpha=0.5,
            stroke=0,
        )
        + geom_point(
            data=df_pred,
            mapping=aes(x="idx_frame", y="value_position"),
            color="#E41A1C",
            size=1.5,
            alpha=0.5,
            stroke=0,
        )
        + geom_vline(
            data=df_chunk,
            mapping=aes(xintercept="idx_frame"),
            linetype="dashed",
            color="gray",
            alpha=0.5,
            size=0.5,
        )
        + facet_wrap("~name_motor", ncol=2, scales="free_y")
        + labs(x="Frame index", y="Position (deg)")
        + ggtitle("GT (black line), Prefix (green), Predicted (red), Boundaries (dashed)")
        + theme_publication()
    )

    save_plot(plot, path_output, width=width, height=height)
    logging.info(f"Saved trajectories to {path_output}")


def create_plot_accelerations(
    df_accel_truth: pd.DataFrame,
    df_accel_predicted: pd.DataFrame,
    indices_boundary: list[int],
    path_output: Path,
) -> None:
    """Create acceleration plot: truth vs predicted, faceted by motor, with boundaries highlighted.

    Args:
        df_accel_truth: Ground truth acceleration DataFrame.
        df_accel_predicted: Predicted action acceleration DataFrame.
        indices_boundary: Frame indices where chunk boundaries occur.
        path_output: Path to save the plot.
    """
    df_truth = df_accel_truth[["idx_frame", "name_motor", "value_acceleration"]].copy()
    df_pred = df_accel_predicted[["idx_frame", "name_motor", "value_acceleration"]].copy()

    df_chunk = pd.DataFrame({"idx_frame": indices_boundary})

    n_motors = df_truth["name_motor"].nunique()
    n_rows = (n_motors + 1) // 2
    height = n_rows * 3 + 1
    width = 14

    plot = (
        ggplot()
        + geom_line(
            data=df_truth,
            mapping=aes(x="idx_frame", y="value_acceleration"),
            color="#377EB8",
            size=0.4,
            alpha=0.7,
        )
        + geom_line(
            data=df_pred,
            mapping=aes(x="idx_frame", y="value_acceleration"),
            color="#E41A1C",
            size=0.4,
            alpha=0.5,
            linetype="dashed",
        )
        + geom_vline(
            data=df_chunk,
            mapping=aes(xintercept="idx_frame"),
            linetype="dashed",
            color="gray",
            alpha=0.5,
            size=0.5,
        )
        + facet_wrap("~name_motor", ncol=2)
        + labs(x="Frame index", y="Acceleration (deg/frame^2)")
        + ggtitle("Acceleration: GT (blue solid) vs Predicted (red dashed)")
        + theme_publication()
    )

    save_plot(plot, path_output, width=width, height=height)
    logging.info(f"Saved accelerations to {path_output}")


# ============================================================================
# Main
# ============================================================================


@draccus.wrap()
def main(cfg: ConfigOfflineReplay):
    init_logging()

    logging.info(f"Setting seed: {cfg.seed}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Parse episode indices
    indices_episode = [int(x.strip()) for x in cfg.indices_episode.split(",")]
    logging.info(f"Episodes: {indices_episode}")

    # Load policy
    logging.info(f"Loading policy from {cfg.path_policy}")
    policy = ACTSmoothPolicy.from_pretrained(cfg.path_policy)
    policy.to(device)
    policy.eval()

    K = policy.config.length_prefix_past
    D = policy.config.length_prefix_future
    C = policy.config.chunk_size
    n_action_steps = policy.config.n_action_steps
    advance = cfg.advance if cfg.advance > 0 else n_action_steps
    logging.info(
        f"Policy config: length_prefix_past={K}, length_prefix_future={D}, "
        f"chunk_size={C}, n_action_steps={n_action_steps}, advance={advance}"
    )

    # Load preprocessor/postprocessor
    overrides_device = {"device_processor": {"device": str(device)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.path_policy,
        preprocessor_overrides=overrides_device,
        postprocessor_overrides=overrides_device,
    )

    # Load dataset metadata
    logging.info(f"Loading dataset metadata from {cfg.id_repo_dataset}")
    meta_dataset = LeRobotDatasetMetadata(cfg.id_repo_dataset)
    fps = meta_dataset.fps
    logging.info(f"Dataset FPS: {fps}")

    # Motor names from action features
    names_motor = list(meta_dataset.features["action"]["names"])
    logging.info(f"Motor names: {names_motor}")

    # Collect results across all episodes
    rows_truth_all = []
    rows_predicted_all = []
    indices_boundary_all = []
    results_per_episode = []

    for idx_episode in indices_episode:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Processing episode {idx_episode}")
        logging.info(f"{'=' * 60}")

        # Get episode frame range
        idx_from = meta_dataset.episodes["dataset_from_index"][idx_episode]
        idx_to = meta_dataset.episodes["dataset_to_index"][idx_episode]
        idx_from = int(idx_from.item() if hasattr(idx_from, "item") else idx_from)
        idx_to = int(idx_to.item() if hasattr(idx_to, "item") else idx_to)
        length_episode = idx_to - idx_from
        logging.info(f"Episode {idx_episode}: frames [{idx_from}, {idx_to}), length={length_episode}")

        # Load dataset for this episode
        dataset = load_dataset_for_episode(
            cfg.id_repo_dataset, idx_episode, K, D, C, fps, cfg.backend_video, meta_dataset
        )

        # Valid range: need K past actions and D+C future actions
        idx_valid_from = idx_from + K
        idx_valid_to = idx_to - D - C
        if idx_valid_to <= idx_valid_from:
            logging.warning(f"Episode {idx_episode} too short, skipping")
            continue

        # Simulate chunk loop: advance by n_action_steps per chunk
        # First chunk starts at idx_valid_from (first frame with enough past context)
        rows_truth_episode = []
        rows_predicted_episode = []
        indices_boundary_episode = []

        # Walk through episode at chunk boundaries
        # timestep_obs is the absolute dataset index where we take observation
        timestep_obs = idx_valid_from
        idx_chunk = 0

        while timestep_obs < idx_valid_to:
            logging.info(f"  Chunk {idx_chunk}: timestep_obs={timestep_obs}")

            # Dataset loaded with episodes=[idx] is 0-indexed within the episode
            sample = dataset[timestep_obs - idx_from]
            actions_all = sample["action"]  # [K + D + C, action_dim]

            # Run inference
            pred = predict_chunk_at_timestep(policy, preprocessor, postprocessor, sample, device, K, D)
            # pred shape: [chunk_size, action_dim]

            # Record chunk boundary (frame where prediction starts)
            idx_frame_boundary = timestep_obs + D
            if idx_chunk > 0:
                indices_boundary_episode.append(idx_frame_boundary)

            # Record ground truth for the full range covered
            # GT: from timestep_obs - K to timestep_obs + D + C (covers prefix + target)
            # But we only need GT for the target region for new frames
            gt_actions = actions_all.cpu().numpy()
            # gt_actions[K + D:] are the target actions starting at timestep_obs + D

            # Record prefix (past + future) as predicted with source="prefix"
            # Past prefix: frames [timestep_obs - K, timestep_obs)
            for i in range(K):
                idx_frame = timestep_obs - K + i
                for j, name_motor in enumerate(names_motor):
                    rows_predicted_episode.append(
                        {
                            "idx_episode": idx_episode,
                            "idx_frame": idx_frame,
                            "name_motor": name_motor,
                            "value_position": float(gt_actions[i, j]),
                            "source": "prefix",
                            "idx_chunk": idx_chunk,
                        }
                    )

            # Future prefix: frames [timestep_obs, timestep_obs + D)
            for i in range(D):
                idx_frame = timestep_obs + i
                for j, name_motor in enumerate(names_motor):
                    rows_predicted_episode.append(
                        {
                            "idx_episode": idx_episode,
                            "idx_frame": idx_frame,
                            "name_motor": name_motor,
                            "value_position": float(gt_actions[K + i, j]),
                            "source": "prefix",
                            "idx_chunk": idx_chunk,
                        }
                    )

            # Predicted chunk: frames [timestep_obs + D, timestep_obs + D + C)
            # Record full chunk when blending (need overlap), otherwise only n_action_steps
            cnt_pred_to_record = C if cfg.blend_overlap else min(n_action_steps, C)
            for i in range(cnt_pred_to_record):
                idx_frame = timestep_obs + D + i
                for j, name_motor in enumerate(names_motor):
                    rows_predicted_episode.append(
                        {
                            "idx_episode": idx_episode,
                            "idx_frame": idx_frame,
                            "name_motor": name_motor,
                            "value_position": float(pred[i, j]),
                            "source": "prediction",
                            "idx_chunk": idx_chunk,
                        }
                    )

            # Advance to next chunk boundary
            timestep_obs += advance
            idx_chunk += 1

        logging.info(f"Episode {idx_episode}: {idx_chunk} chunks processed")

        # Build ground truth DataFrame for this episode (full episode range)
        # Load ground truth actions directly from dataset (no delta_timestamps needed, just the raw actions)
        # We use the already-loaded dataset, iterating frame by frame
        dataset_gt = LeRobotDataset(
            cfg.id_repo_dataset,
            episodes=[idx_episode],
            delta_timestamps={"action": [0]},
            video_backend=cfg.backend_video,
        )
        for idx_frame in range(idx_from, idx_to):
            # Dataset loaded with episodes=[idx] is 0-indexed within the episode
            sample_gt = dataset_gt[idx_frame - idx_from]
            action_gt = sample_gt["action"].cpu().numpy().squeeze(0)
            for j, name_motor in enumerate(names_motor):
                rows_truth_episode.append(
                    {
                        "idx_episode": idx_episode,
                        "idx_frame": idx_frame,
                        "name_motor": name_motor,
                        "value_position": float(action_gt[j]),
                    }
                )

        # Compute episode-level metrics on the stitched predicted sequence
        df_predicted_episode = pd.DataFrame(rows_predicted_episode)
        df_truth_episode = pd.DataFrame(rows_truth_episode)

        # For kinematics, use only the "prediction" rows (stitched predicted actions)
        df_pred_only = df_predicted_episode[df_predicted_episode["source"] == "prediction"].copy()

        if cfg.blend_overlap and advance < n_action_steps:
            # Overlap blending: for frames with predictions from multiple chunks,
            # linearly blend from old chunk to new chunk across the overlap region.
            size_overlap = n_action_steps - advance
            rows_blended = []
            for (idx_frame, name_motor), group in df_pred_only.groupby(["idx_frame", "name_motor"]):
                if len(group) == 1:
                    rows_blended.append(group.iloc[0].to_dict())
                else:
                    # Sort by chunk index: earlier chunk first
                    group = group.sort_values("idx_chunk")
                    # Use newest chunk's prediction, weighted by position in overlap
                    # Position within overlap: how far into the overlap region this frame is
                    # relative to the latest chunk's start
                    idx_chunk_latest = group["idx_chunk"].max()
                    idx_chunk_earliest = group["idx_chunk"].min()
                    # The latest chunk starts predicting at its timestep_obs + D
                    # Frame's position in overlap: (idx_frame - (latest_chunk_boundary)) / size_overlap
                    # Simplified: latest chunk's boundary = idx_valid_from + idx_chunk_latest * advance + D
                    # But we can compute alpha from the fact that the overlap spans `size_overlap` frames
                    # at the start of the new chunk. Higher alpha = favor new chunk.
                    # The new chunk's first prediction frame = old chunk's boundary + advance
                    # overlap starts there, overlap ends at old chunk's boundary + n_action_steps
                    boundary_new = idx_valid_from + idx_chunk_latest * advance + D
                    alpha = (idx_frame - boundary_new) / size_overlap if size_overlap > 0 else 1.0
                    alpha = max(0.0, min(1.0, alpha))
                    value_old = group.iloc[0]["value_position"]  # earliest chunk
                    value_new = group.iloc[-1]["value_position"]  # latest chunk
                    value_blended = (1.0 - alpha) * value_old + alpha * value_new
                    row = group.iloc[-1].to_dict()
                    row["value_position"] = value_blended
                    rows_blended.append(row)
            df_pred_only = pd.DataFrame(rows_blended)
        else:
            # Deduplicate: keep the last prediction for each (idx_frame, name_motor)
            # (later chunks overwrite earlier ones at same frame)
            df_pred_only = df_pred_only.sort_values(["name_motor", "idx_chunk", "idx_frame"])
            df_pred_only = df_pred_only.drop_duplicates(subset=["idx_frame", "name_motor"], keep="last")

        df_pred_only = df_pred_only.sort_values(["name_motor", "idx_frame"]).reset_index(drop=True)

        if len(df_pred_only) > 0:
            df_kinematics_pred = compute_kinematics(df_pred_only)
            scalars_pred = compute_scalars_kinematic(df_kinematics_pred)

            df_kinematics_truth = compute_kinematics(df_truth_episode)
            scalars_truth = compute_scalars_kinematic(df_kinematics_truth)

            # Boundary-specific metrics
            indices_boundary_set = set(indices_boundary_episode)
            scalars_boundary = compute_scalars_kinematic_boundary(df_kinematics_pred, indices_boundary_set)

            results_per_episode.append(
                {
                    "idx_episode": idx_episode,
                    "cnt_chunks": idx_chunk,
                    **{f"pred_{k}": v for k, v in scalars_pred.items()},
                    **{f"truth_{k}": v for k, v in scalars_truth.items()},
                    **{f"boundary_{k}": v for k, v in scalars_boundary.items()},
                }
            )

            logging.info(f"Episode {idx_episode} metrics:")
            logging.info(f"  Predicted: U_a={scalars_pred['uniformity_acceleration']:.6f}")
            logging.info(f"  Truth:     U_a={scalars_truth['uniformity_acceleration']:.6f}")
            logging.info(
                f"  Boundary spike: {scalars_boundary['ratio_boundary_spike']:.2f}x "
                f"(boundary={scalars_boundary['mean_accel_boundary']:.4f}, "
                f"non-boundary={scalars_boundary['mean_accel_non_boundary']:.4f})"
            )

        # Accumulate for combined plots
        rows_truth_all.extend(rows_truth_episode)
        rows_predicted_all.extend(rows_predicted_episode)
        indices_boundary_all.extend(indices_boundary_episode)

    # =====================================================================
    # Output: Results text
    # =====================================================================
    path_output = Path(cfg.path_output)
    path_output.parent.mkdir(parents=True, exist_ok=True)

    lines_results = [
        f"policy: {cfg.path_policy}",
        f"dataset: {cfg.id_repo_dataset}",
        f"episodes: {indices_episode}",
        f"config: K={K}, D={D}, C={C}, n_action_steps={n_action_steps}, advance={advance}, blend={cfg.blend_overlap}",
        "",
    ]

    for result in results_per_episode:
        lines_results.append(f"=== Episode {result['idx_episode']} ({result['cnt_chunks']} chunks) ===")
        lines_results.append(f"  Predicted U_v:  {result['pred_uniformity_velocity']:.6f}  deg^2/frame^2")
        lines_results.append(f"  Predicted U_a:  {result['pred_uniformity_acceleration']:.6f}  deg^2/frame^4")
        lines_results.append(f"  Predicted mu_v: {result['pred_mean_velocity']:.6f}  deg/frame")
        lines_results.append(f"  Truth U_v:      {result['truth_uniformity_velocity']:.6f}  deg^2/frame^2")
        lines_results.append(f"  Truth U_a:      {result['truth_uniformity_acceleration']:.6f}  deg^2/frame^4")
        lines_results.append(f"  Truth mu_v:     {result['truth_mean_velocity']:.6f}  deg/frame")
        lines_results.append(
            f"  Boundary spike: {result['boundary_ratio_boundary_spike']:.2f}x "
            f"(boundary={result['boundary_mean_accel_boundary']:.4f}, "
            f"non-boundary={result['boundary_mean_accel_non_boundary']:.4f})"
        )
        lines_results.append("")

    # Aggregate across episodes
    if len(results_per_episode) > 1:
        lines_results.append("=== Aggregate ===")
        for metric in [
            "pred_uniformity_acceleration",
            "pred_uniformity_velocity",
            "pred_mean_velocity",
            "boundary_ratio_boundary_spike",
        ]:
            values = [r[metric] for r in results_per_episode]
            lines_results.append(f"  {metric}: mean={np.mean(values):.6f}, std={np.std(values):.6f}")
        lines_results.append("")

    path_results = Path(f"{cfg.path_output}_results.txt")
    path_results.write_text("\n".join(lines_results) + "\n")
    logging.info(f"Saved results to {path_results}")

    for line in lines_results:
        logging.info(line)

    # =====================================================================
    # Output: Plots (per episode)
    # =====================================================================
    df_truth_all = pd.DataFrame(rows_truth_all)
    df_predicted_all = pd.DataFrame(rows_predicted_all)

    for idx_episode in indices_episode:
        df_truth_ep = df_truth_all[df_truth_all["idx_episode"] == idx_episode]
        df_pred_ep = df_predicted_all[df_predicted_all["idx_episode"] == idx_episode]
        # Filter boundaries by episode frame range
        idx_from = meta_dataset.episodes["dataset_from_index"][idx_episode]
        idx_to = meta_dataset.episodes["dataset_to_index"][idx_episode]
        idx_from = int(idx_from.item() if hasattr(idx_from, "item") else idx_from)
        idx_to = int(idx_to.item() if hasattr(idx_to, "item") else idx_to)
        indices_boundary_ep = [idx for idx in indices_boundary_all if idx_from <= idx < idx_to]

        if len(df_truth_ep) == 0 or len(df_pred_ep) == 0:
            continue

        suffix_episode = f"_ep{idx_episode}"

        # Trajectories
        path_trajectories = Path(f"{cfg.path_output}{suffix_episode}_trajectories.png")
        create_plot_trajectories(df_truth_ep, df_pred_ep, indices_boundary_ep, path_trajectories)

        # Accelerations
        df_pred_only = df_pred_ep[df_pred_ep["source"] == "prediction"].copy()
        df_pred_only = df_pred_only.sort_values(["name_motor", "idx_chunk", "idx_frame"])
        df_pred_only = df_pred_only.drop_duplicates(subset=["idx_frame", "name_motor"], keep="last")

        df_kinematics_truth = compute_kinematics(df_truth_ep)
        df_kinematics_pred = compute_kinematics(df_pred_only)

        df_accel_truth = df_kinematics_truth.dropna(subset=["value_acceleration"])
        df_accel_pred = df_kinematics_pred.dropna(subset=["value_acceleration"])

        if len(df_accel_truth) > 0 and len(df_accel_pred) > 0:
            path_accelerations = Path(f"{cfg.path_output}{suffix_episode}_accelerations.png")
            create_plot_accelerations(df_accel_truth, df_accel_pred, indices_boundary_ep, path_accelerations)

    # Also save CSV with per-episode metrics for easy downstream comparison
    if results_per_episode:
        df_results = pd.DataFrame(results_per_episode)
        path_csv = Path(f"{cfg.path_output}_metrics.csv")
        df_results.to_csv(path_csv, index=False)
        logging.info(f"Saved metrics CSV to {path_csv}")

    logging.info("Done!")


if __name__ == "__main__":
    main()
