"""Measure ground-truth kinematic metrics on dataset episodes.

Computes U_a, U_v, and boundary spike ratio on the raw demonstration data
at different chunk boundary spacings (advance values). This establishes the
floor for model performance: the best achievable metrics if prediction were
perfect.

Usage:
    python scripts/measure_gt_kinematics.py \
        --id_repo_dataset=giacomoran/lerobot_policy_act_smooth_30fps \
        --indices_episode=0,1,2 \
        --advances=30,20,15 \
        --length_prefix_past=4 \
        --length_prefix_future=2 \
        --chunk_size=30
"""

import argparse
import logging

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================================
# Kinematics (same as eval_offline_replay.py)
# ============================================================================


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity and acceleration from position data."""
    df = df.copy().sort_values(["name_motor", "idx_frame"])
    df["value_velocity"] = df.groupby("name_motor")["value_position"].diff()
    df["value_acceleration"] = df.groupby("name_motor")["value_velocity"].diff()
    return df


def compute_scalars_kinematic(df_kinematics: pd.DataFrame) -> dict[str, float]:
    """Compute kinematic scalar measures (U_v, U_a, mu_v)."""
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
    """Compute acceleration at boundary vs non-boundary timesteps."""
    df = df_kinematics.dropna(subset=["value_acceleration"]).copy()

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
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Measure ground-truth kinematic metrics on dataset episodes.")
    parser.add_argument("--id_repo_dataset", type=str, required=True)
    parser.add_argument("--indices_episode", type=str, default="0,1,2")
    parser.add_argument("--advances", type=str, default="30,20,15")
    parser.add_argument("--length_prefix_past", type=int, default=4, help="K (past prefix length)")
    parser.add_argument("--length_prefix_future", type=int, default=2, help="D (future prefix length)")
    parser.add_argument("--chunk_size", type=int, default=30, help="C (chunk size)")
    args = parser.parse_args()

    indices_episode = [int(x) for x in args.indices_episode.split(",")]
    advances = [int(x) for x in args.advances.split(",")]
    K = args.length_prefix_past
    D = args.length_prefix_future
    C = args.chunk_size

    logging.info(f"Dataset: {args.id_repo_dataset}")
    logging.info(f"Episodes: {indices_episode}")
    logging.info(f"Advances: {advances}")
    logging.info(f"Config: K={K}, D={D}, C={C}")

    # Load dataset metadata
    meta_dataset = LeRobotDatasetMetadata(args.id_repo_dataset)
    names_motor = list(meta_dataset.features["action"]["names"])
    logging.info(f"Motors: {names_motor}")

    results_all = []

    for idx_episode in indices_episode:
        # Load GT actions for this episode
        dataset = LeRobotDataset(
            args.id_repo_dataset,
            episodes=[idx_episode],
            delta_timestamps={"action": [0]},
        )

        idx_from = meta_dataset.episodes["dataset_from_index"][idx_episode]
        idx_to = meta_dataset.episodes["dataset_to_index"][idx_episode]
        idx_from = int(idx_from.item() if hasattr(idx_from, "item") else idx_from)
        idx_to = int(idx_to.item() if hasattr(idx_to, "item") else idx_to)
        length_episode = idx_to - idx_from

        # Extract all GT actions
        actions = np.stack([dataset[i]["action"].squeeze(0).numpy() for i in range(length_episode)])

        # Build tidy DataFrame
        rows = []
        for t in range(length_episode):
            for j, name_motor in enumerate(names_motor):
                rows.append(
                    {
                        "idx_episode": idx_episode,
                        "idx_frame": idx_from + t,
                        "name_motor": name_motor,
                        "value_position": float(actions[t, j]),
                    }
                )
        df = pd.DataFrame(rows)

        # Compute kinematics
        df_kin = compute_kinematics(df)
        scalars = compute_scalars_kinematic(df_kin)

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Episode {idx_episode} (length={length_episode})")
        logging.info(f"{'=' * 60}")
        logging.info(f"  GT U_v:  {scalars['uniformity_velocity']:.6f}  deg^2/frame^2")
        logging.info(f"  GT U_a:  {scalars['uniformity_acceleration']:.6f}  deg^2/frame^4")
        logging.info(f"  GT mu_v: {scalars['mean_velocity']:.6f}  deg/frame")

        # Compute boundary metrics for each advance value
        for advance in advances:
            # Same boundary logic as eval_offline_replay.py
            idx_valid_from = idx_from + K
            idx_valid_to = idx_to - D - C

            indices_boundary = []
            timestep_obs = idx_valid_from
            idx_chunk = 0
            while timestep_obs < idx_valid_to:
                idx_frame_boundary = timestep_obs + D
                if idx_chunk > 0:
                    indices_boundary.append(idx_frame_boundary)
                timestep_obs += advance
                idx_chunk += 1

            if indices_boundary:
                scalars_boundary = compute_scalars_kinematic_boundary(df_kin, set(indices_boundary))
                logging.info(
                    f"  advance={advance:2d}: spike={scalars_boundary['ratio_boundary_spike']:.3f}x "
                    f"(boundary={scalars_boundary['mean_accel_boundary']:.4f}, "
                    f"non-boundary={scalars_boundary['mean_accel_non_boundary']:.4f}), "
                    f"n_boundaries={len(indices_boundary)}, n_chunks={idx_chunk}"
                )
                results_all.append(
                    {
                        "idx_episode": idx_episode,
                        "advance": advance,
                        "gt_uniformity_acceleration": scalars["uniformity_acceleration"],
                        "gt_uniformity_velocity": scalars["uniformity_velocity"],
                        "gt_mean_velocity": scalars["mean_velocity"],
                        "boundary_spike": scalars_boundary["ratio_boundary_spike"],
                        "mean_accel_boundary": scalars_boundary["mean_accel_boundary"],
                        "mean_accel_non_boundary": scalars_boundary["mean_accel_non_boundary"],
                        "n_boundaries": len(indices_boundary),
                        "n_chunks": idx_chunk,
                    }
                )

    # Aggregate across episodes
    if results_all:
        df_results = pd.DataFrame(results_all)
        logging.info(f"\n{'=' * 60}")
        logging.info("Aggregate (mean across episodes)")
        logging.info(f"{'=' * 60}")
        logging.info(f"  GT U_a:  {df_results['gt_uniformity_acceleration'].mean():.6f}")
        for advance in advances:
            df_adv = df_results[df_results["advance"] == advance]
            if len(df_adv) > 0:
                logging.info(
                    f"  advance={advance:2d}: spike={df_adv['boundary_spike'].mean():.3f}x "
                    f"(std={df_adv['boundary_spike'].std():.3f})"
                )


if __name__ == "__main__":
    main()
