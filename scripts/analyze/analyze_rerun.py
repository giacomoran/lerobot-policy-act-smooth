#!/usr/bin/env python
"""Analyze observations and actions from a .rrd rerun recording file.

Produces:
- Trajectory plot: obs (lines), commanded actions (small dots), policy actions (big dots)
- Acceleration plot: obs vs action accelerations overlaid, faceted by motor
- Kinematic scalar summaries (velocity uniformity, acceleration uniformity, mean velocity)

Usage:
    python scripts/analyze/analyze_rerun.py \
        --path_rrd=outputs/recordings/run.rrd \
        --path_output=outputs/recordings/analyze_run
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import pandas as pd
import rerun as rr
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
    scale_color_manual,
)

from utils_plotting import extract_scalar_from_rerun, save_plot, theme_publication


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ConfigRerunAnalysis:
    """Configuration for analyzing rerun recordings."""

    path_rrd: str = ""

    # Output stem prefix (no extension). Defaults to {rrd_dir}/analyze_{rrd_stem}.
    # Produces files like {path_output}_trajectories.png, {path_output}_accelerations.png.
    path_output: str = ""

    # Motor names (comma-separated). Auto-detect if None.
    names_motor: str = ""

    # Whether to facet by motor (default: False, all joints in same plot)
    faceted: bool = False

    def __post_init__(self):
        if not self.path_rrd:
            raise ValueError("--path_rrd must be provided")

        if not self.path_output:
            path_rrd = Path(self.path_rrd)
            self.path_output = str(Path("outputs/results-analyze") / f"analyze_{path_rrd.stem}")


# ============================================================================
# Data Loading
# ============================================================================


def load_recording_to_dataframe(path_rrd: Path) -> tuple[pd.DataFrame, str]:
    """Load .rrd recording and convert to pandas DataFrame.

    Args:
        path_rrd: Path to the .rrd recording file.

    Returns:
        Tuple of (DataFrame with scalar data, index column name).
    """
    logging.info(f"Loading rerun recording: {path_rrd}")
    recording = rr.dataframe.load_recording(str(path_rrd))

    schema = recording.schema()
    cols_index = [col.name for col in schema.index_columns()]

    name_index = "frame_nr" if "frame_nr" in cols_index else cols_index[0] if cols_index else "log_tick"
    logging.info(f"Using index: {name_index}")

    view = recording.view(index=name_index, contents="/**")
    table = view.select().read_all()
    logging.info(f"Query completed, got {table.num_rows} rows")

    cols_scalar = [col for col in table.schema.names if ":Scalars:scalars" in col and col != name_index]
    table_scalar = table.select(cols_scalar)
    df = table_scalar.to_pandas()

    table_index = table.select([name_index])
    df_index = table_index.to_pandas()
    df[name_index] = df_index[name_index]

    logging.info(f"Filtered to {len(cols_scalar)} scalar columns")
    return df, name_index


def detect_names_motor(df: pd.DataFrame) -> list[str]:
    """Auto-detect motor names from DataFrame columns.

    Looks for columns matching /observation/{motor}.pos:Scalars:scalars pattern.

    Args:
        df: DataFrame with rerun data.

    Returns:
        List of motor names (e.g., ["shoulder_pan", "shoulder_lift", ...]).
    """
    names = []
    for col in df.columns:
        if col.startswith("/observation/") and col.endswith(".pos:Scalars:scalars"):
            name_motor = col.replace("/observation/", "").replace(".pos:Scalars:scalars", "")
            names.append(name_motor)

    logging.info(f"Auto-detected motor names: {names}")
    return names


# ============================================================================
# DataFrame Building
# ============================================================================


def build_dataframe_observations(
    df: pd.DataFrame,
    names_motor: list[str],
    name_index: str,
) -> pd.DataFrame:
    """Build tidy DataFrame for observations.

    Args:
        df: Raw DataFrame from rerun.
        names_motor: List of motor names.
        name_index: Index column name.

    Returns:
        Tidy DataFrame with columns: idx_frame, name_motor, value_position.
    """
    rows = []

    for idx_frame, row in df.iterrows():
        for name_motor in names_motor:
            col_obs = f"/observation/{name_motor}.pos:Scalars:scalars"
            if col_obs in df.columns:
                value = extract_scalar_from_rerun(row[col_obs])
                if value is not None:
                    rows.append(
                        {
                            "idx_frame": int(row[name_index]) if name_index in row else idx_frame,
                            "name_motor": name_motor,
                            "value_position": value,
                        }
                    )

    df_result = pd.DataFrame(rows)

    if not df_result.empty:
        df_result = df_result.sort_values(["name_motor", "idx_frame"]).reset_index(drop=True)

    return df_result


def drop_consecutive_duplicates(
    df: pd.DataFrame,
    col_value: str = "value_position",
    col_group: str = "name_motor",
) -> pd.DataFrame:
    """Drop rows where col_value equals the previous row within each col_group.

    Useful for reducing data points before trajectory plotting, but should NOT
    be applied before computing kinematics (velocity/acceleration) since it
    creates frame gaps that produce wrong derivatives.
    """
    shifted = df.groupby(col_group)[col_value].shift()
    return df[df[col_value] != shifted].reset_index(drop=True)


def build_dataframe_actions(
    df: pd.DataFrame,
    names_motor: list[str],
    name_index: str,
    only_control_frames: bool = False,
) -> pd.DataFrame:
    """Build tidy DataFrame for actions.

    Args:
        df: Raw DataFrame from rerun.
        names_motor: List of motor names.
        name_index: Index column name.
        only_control_frames: If True, only include actions at control frames (policy timesteps).
            If False, include all frames (showing interpolated commands).

    Returns:
        Tidy DataFrame with columns: idx_frame, name_motor, value_position.
    """
    rows = []

    # Optionally filter to control frames only
    indices_control_frame = None
    if only_control_frames:
        col_timestep = "/timestep:Scalars:scalars"
        has_timestep = col_timestep in df.columns

        if has_timestep:
            vals_timestep = df[col_timestep].apply(extract_scalar_from_rerun).values
            # Find first frame for each unique timestep
            vals_unique, indices_first = np.unique(vals_timestep[~pd.isna(vals_timestep)], return_index=True)
            indices_control_frame = set(indices_first)
        else:
            indices_control_frame = set(range(len(df)))

    for idx_frame, row in df.iterrows():
        if indices_control_frame is not None and idx_frame not in indices_control_frame:
            continue

        for name_motor in names_motor:
            col_action = f"/action/{name_motor}/pos:Scalars:scalars"
            if col_action in df.columns:
                value = extract_scalar_from_rerun(row[col_action])
                if value is not None:
                    rows.append(
                        {
                            "idx_frame": int(row[name_index]) if name_index in row else idx_frame,
                            "name_motor": name_motor,
                            "value_position": value,
                        }
                    )

    df_result = pd.DataFrame(rows)

    return df_result


def extract_indices_chunk_switch(df: pd.DataFrame, name_index: str) -> list[int]:
    """Extract frame indices where chunk switches occur.

    Args:
        df: Raw DataFrame from rerun.
        name_index: Index column name (e.g., 'frame_nr').

    Returns:
        List of frame numbers where idx_chunk changes.
    """
    col_idx_chunk = "/idx_chunk:Scalars:scalars"
    indices = []

    if col_idx_chunk not in df.columns:
        return indices

    data_chunk = df[col_idx_chunk].apply(extract_scalar_from_rerun)
    data_chunk = data_chunk.ffill().fillna(-1)

    # Get frame numbers from the index column
    if name_index in df.columns:
        vals_frame = df[name_index].values
    else:
        vals_frame = np.arange(len(df))

    for i in range(1, len(data_chunk)):
        if data_chunk.iloc[i] != data_chunk.iloc[i - 1]:
            # Return actual frame number, not DataFrame row index
            indices.append(int(vals_frame[i]))

    logging.info(f"Found {len(indices)} chunk switches")
    return indices


# ============================================================================
# Kinematics
# ============================================================================


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity and acceleration from position data.

    Works for both observations and actions — any tidy DataFrame with
    idx_frame, name_motor, value_position.

    Args:
        df: Tidy DataFrame sorted by [name_motor, idx_frame].

    Returns:
        Augmented DataFrame with value_velocity and value_acceleration columns.
        First 1-2 rows per motor will have NaN for derivatives.
    """
    df = df.copy().sort_values(["name_motor", "idx_frame"])
    df["value_velocity"] = df.groupby("name_motor")["value_position"].diff()
    df["value_acceleration"] = df.groupby("name_motor")["value_velocity"].diff()
    return df


def compute_scalars_kinematic(df_kinematics: pd.DataFrame) -> dict[str, float]:
    """Compute Liu 2026 kinematic scalar measures.

    Args:
        df_kinematics: DataFrame with value_velocity and value_acceleration columns
            (output of compute_kinematics).

    Returns:
        Dict with keys:
        - uniformity_velocity (U_v): max across joints of per-joint velocity variance
        - uniformity_acceleration (U_a): max across joints of per-joint acceleration variance
        - mean_velocity (mu_v): mean absolute velocity across all joints and frames
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


# ============================================================================
# Plotting
# ============================================================================


def create_plot(
    df_obs: pd.DataFrame,
    df_action_commanded: pd.DataFrame,
    df_action_policy: pd.DataFrame,
    indices_chunk: list[int],
    faceted: bool,
    path_output: Path,
) -> None:
    """Create and save the rerun visualization plot.

    Shows three layers per motor:
    - Solid line: observations (actual robot state)
    - Small dots: commanded actions (including interpolated frames)
    - Big dots: policy actions (at control frames only)

    Args:
        df_obs: DataFrame with observations (all frames).
        df_action_commanded: DataFrame with commanded actions (all frames, including interpolated).
        df_action_policy: DataFrame with policy actions (control frames only).
        indices_chunk: List of frame indices where chunks switch.
        faceted: Whether to facet by motor.
        path_output: Path to save the plot.
    """
    df_chunk = pd.DataFrame({"idx_frame": indices_chunk})

    n_motors = df_obs["name_motor"].nunique()

    # Colors that are distinguishable (avoiding yellow)
    colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#A65628",  # brown
        "#F781BF",  # pink
        "#666666",  # gray
    ]
    names_motor = sorted(df_obs["name_motor"].unique())
    color_mapping = {name: colors[i % len(colors)] for i, name in enumerate(names_motor)}

    layers_common = [
        geom_line(
            data=df_obs,
            mapping=aes(x="idx_frame", y="value_position", color="name_motor"),
            size=0.5,
            alpha=0.7,
        ),
        geom_point(
            data=df_action_commanded,
            mapping=aes(x="idx_frame", y="value_position", color="name_motor"),
            size=1.2,
            alpha=0.4,
            stroke=0,
        ),
        geom_point(
            data=df_action_policy,
            mapping=aes(x="idx_frame", y="value_position", color="name_motor"),
            size=2.0,
            alpha=0.5,
            stroke=0,
        ),
        geom_vline(
            data=df_chunk,
            mapping=aes(xintercept="idx_frame"),
            linetype="dashed",
            color="gray",
            alpha=0.5,
            size=0.5,
        ),
        scale_color_manual(values=color_mapping),
        labs(x="Frame index", y="Position (deg)", color="Motor"),
        ggtitle("Obs (line), Commands (small dots), Policy actions (big dots)"),
        theme_publication(),
    ]

    if faceted:
        plot = ggplot() + facet_wrap("~name_motor", ncol=2, scales="free_y")
        for layer in layers_common:
            plot = plot + layer

        n_rows = (n_motors + 1) // 2
        height = n_rows * 3 + 1
        width = 14
    else:
        plot = ggplot()
        for layer in layers_common:
            plot = plot + layer

        height = 8
        width = 14

    save_plot(plot, path_output, width=width, height=height)
    logging.info(f"Saved plot to {path_output}")


def create_plot_accelerations(
    df_accel_obs: pd.DataFrame,
    df_accel_action: pd.DataFrame,
    indices_chunk: list[int],
    path_output: Path,
) -> None:
    """Create and save an acceleration plot with obs and action overlaid.

    Faceted by motor, with chunk boundaries as vertical dashed lines.
    Observation accelerations are solid lines, action accelerations are dashed.

    Args:
        df_accel_obs: Observation kinematics with value_acceleration (NaN-dropped).
        df_accel_action: Action kinematics with value_acceleration (NaN-dropped).
        indices_chunk: Frame indices where chunks switch.
        path_output: Path to save the plot.
    """
    # Tag source for legend
    df_obs = df_accel_obs[["idx_frame", "name_motor", "value_acceleration"]].copy()
    df_obs["source"] = "observation"
    df_act = df_accel_action[["idx_frame", "name_motor", "value_acceleration"]].copy()
    df_act["source"] = "action"

    df_chunk = pd.DataFrame({"idx_frame": indices_chunk})

    n_motors = df_obs["name_motor"].nunique()
    n_rows = (n_motors + 1) // 2
    height = n_rows * 3 + 1
    width = 14

    plot = (
        ggplot()
        + geom_line(
            data=df_obs,
            mapping=aes(x="idx_frame", y="value_acceleration"),
            color="#377EB8",
            size=0.4,
            alpha=0.7,
        )
        + geom_line(
            data=df_act,
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
        + labs(x="Frame index", y="Acceleration (deg/frame²)")
        + ggtitle("Acceleration: observation (blue solid) vs action (red dashed)")
        + theme_publication()
    )

    save_plot(plot, path_output, width=width, height=height)
    logging.info(f"Saved plot to {path_output}")


# ============================================================================
# Main
# ============================================================================


@draccus.wrap()
def main(cfg: ConfigRerunAnalysis):
    """Main entry point for rerun analysis."""
    init_logging()

    path_rrd = Path(cfg.path_rrd)
    if not path_rrd.exists():
        raise FileNotFoundError(f"Recording file not found: {path_rrd}")

    df, name_index = load_recording_to_dataframe(path_rrd)

    if cfg.names_motor:
        names_motor = [n.strip() for n in cfg.names_motor.split(",")]
        logging.info(f"Using provided motor names: {names_motor}")
    else:
        names_motor = detect_names_motor(df)

    if not names_motor:
        raise ValueError("No motor names found. Provide --names_motor or check the recording.")

    # Build DataFrames (unfiltered)
    logging.info("Building observation DataFrame...")
    df_obs = build_dataframe_observations(df, names_motor, name_index)
    logging.info(f"  {len(df_obs)} observation rows")

    logging.info("Building action DataFrames...")
    df_action_commanded = build_dataframe_actions(df, names_motor, name_index, only_control_frames=False)
    df_action_policy = build_dataframe_actions(df, names_motor, name_index, only_control_frames=True)
    logging.info(f"  {len(df_action_commanded)} commanded action rows, {len(df_action_policy)} policy action rows")

    indices_chunk = extract_indices_chunk_switch(df, name_index)

    # Kinematics for observations
    df_kinematics_obs = compute_kinematics(df_obs)
    scalars_obs = compute_scalars_kinematic(df_kinematics_obs)

    # Kinematics for commanded actions
    df_kinematics_action = compute_kinematics(df_action_commanded)
    scalars_action = compute_scalars_kinematic(df_kinematics_action)

    # Log and write kinematic summaries
    lines_results = [
        f"source: {path_rrd}",
        "",
        "=== Kinematic Summary (Observations) ===",
        f"  U_v  (velocity uniformity):     {scalars_obs['uniformity_velocity']:.6f}  deg²/frame²",
        f"  U_a  (acceleration uniformity): {scalars_obs['uniformity_acceleration']:.6f}  deg²/frame⁴",
        f"  mu_v (mean velocity):           {scalars_obs['mean_velocity']:.6f}  deg/frame",
        "",
        "=== Kinematic Summary (Actions) ===",
        f"  U_v  (velocity uniformity):     {scalars_action['uniformity_velocity']:.6f}  deg²/frame²",
        f"  U_a  (acceleration uniformity): {scalars_action['uniformity_acceleration']:.6f}  deg²/frame⁴",
        f"  mu_v (mean velocity):           {scalars_action['mean_velocity']:.6f}  deg/frame",
    ]
    for line in lines_results:
        logging.info(line)

    path_results = Path(f"{cfg.path_output}_results.txt")
    path_results.parent.mkdir(parents=True, exist_ok=True)
    path_results.write_text("\n".join(lines_results) + "\n")
    logging.info(f"Saved results to {path_results}")

    # Plot 1: Trajectories (dedup for clean lines)
    path_trajectories = Path(f"{cfg.path_output}_trajectories.png")
    df_obs_deduped = drop_consecutive_duplicates(df_obs)
    logging.info("Creating trajectory plot...")
    create_plot(df_obs_deduped, df_action_commanded, df_action_policy, indices_chunk, cfg.faceted, path_trajectories)

    # Plot 2: Accelerations (obs + action overlaid, faceted by motor)
    path_accelerations = Path(f"{cfg.path_output}_accelerations.png")
    df_accel_obs = df_kinematics_obs.dropna(subset=["value_acceleration"])
    df_accel_action = df_kinematics_action.dropna(subset=["value_acceleration"])
    logging.info("Creating acceleration plot...")
    create_plot_accelerations(df_accel_obs, df_accel_action, indices_chunk, path_accelerations)

    logging.info("Done!")


if __name__ == "__main__":
    main()
