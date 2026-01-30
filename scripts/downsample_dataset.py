#!/usr/bin/env python
"""Downsample a LeRobot v3.0 dataset by selecting every Nth frame.

Example: Convert 30fps dataset to 10fps.

Usage:
    python scripts/downsample_dataset.py \
        --repo-id-source user/my_dataset_30fps \
        --repo-id-output user/my_dataset_10fps \
        --fps-output 10 \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import av
import av.video
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    load_info,
    load_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


def _decode_video_frames(path_video: Path) -> list[np.ndarray]:
    """Decode all frames from a video file.

    Args:
        path_video: Path to the video file

    Returns:
        List of frames as numpy arrays with shape (H, W, C) and dtype uint8
    """
    container = av.open(str(path_video))
    stream = container.streams.video[0]

    frames: list[np.ndarray] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            if not isinstance(frame, av.video.frame.VideoFrame):
                continue
            # Convert to numpy array (H, W, C)
            arr: np.ndarray = frame.to_ndarray(format="rgb24")
            frames.append(arr)

    container.close()
    return frames


def downsample_dataset(
    repo_id_source: str,
    repo_id_output: str,
    fps_output: int,
    path_root_source: Path | None = None,
    path_root_output: Path | None = None,
    vcodec: str = "libsvtav1",
    push_to_hub: bool = False,
) -> None:
    """Downsample a LeRobot dataset by selecting every Nth frame.

    Args:
        repo_id_source: Source dataset repo_id (e.g., "user/dataset_30fps")
        repo_id_output: Output dataset repo_id (e.g., "user/dataset_10fps")
        fps_output: Target frame rate (must evenly divide source fps)
        path_root_source: Optional root path for source dataset
        path_root_output: Optional root path for output dataset
        vcodec: Video codec for encoding (default: libsvtav1)
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    # Resolve paths
    if path_root_source is None:
        path_root_source = HF_LEROBOT_HOME / repo_id_source
    if path_root_output is None:
        path_root_output = HF_LEROBOT_HOME / repo_id_output

    path_root_source = Path(path_root_source)
    path_root_output = Path(path_root_output)

    print(f"Source: {path_root_source}")
    print(f"Output: {path_root_output}")

    # Clean output directory if it exists
    if path_root_output.exists():
        shutil.rmtree(path_root_output)

    # Load source dataset info
    info_source = load_info(path_root_source)
    fps_source = info_source["fps"]

    # Validate that source fps is evenly divisible by output fps
    if fps_source % fps_output != 0:
        raise ValueError(
            f"Source fps ({fps_source}) must be evenly divisible by output fps ({fps_output}). "
            f"Ratio is {fps_source / fps_output:.2f}."
        )

    factor_downsample = fps_source // fps_output
    print(f"Downsampling: {fps_source} fps -> {fps_output} fps (factor={factor_downsample})")

    # Get video features
    features_video = {k: v for k, v in info_source["features"].items() if v.get("dtype") == "video"}

    # Load all episodes metadata
    path_episodes_dir = path_root_source / "meta" / "episodes"
    list_df_episodes = []
    for pq_file in sorted(path_episodes_dir.glob("*/*.parquet")):
        list_df_episodes.append(pd.read_parquet(pq_file))
    df_episodes_source = pd.concat(list_df_episodes, ignore_index=True)
    list_idx_episode = df_episodes_source["episode_index"].tolist()
    count_episodes = len(list_idx_episode)

    # Load tasks
    df_tasks = load_tasks(path_root_source)
    # Get task string for task_index 0 (assuming single task dataset)
    task_str = df_tasks.index[0] if len(df_tasks) > 0 else "default_task"

    # Create output dataset
    ds_output = LeRobotDataset.create(
        repo_id=repo_id_output,
        fps=fps_output,
        features=info_source["features"],
        root=path_root_output,
        robot_type=info_source.get("robot_type"),
        use_videos=len(features_video) > 0,
        vcodec=vcodec,
    )

    # Process each episode
    for i, idx_episode_source in enumerate(list_idx_episode):
        print(f"Processing episode {i + 1}/{count_episodes} (index={idx_episode_source})")

        # Get episode metadata
        ep_meta = df_episodes_source[df_episodes_source["episode_index"] == idx_episode_source].iloc[0]

        # Load episode data from parquet
        chunk_idx = ep_meta["data/chunk_index"]
        file_idx = ep_meta["data/file_index"]
        path_parquet = path_root_source / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df_data = pd.read_parquet(path_parquet)
        df_episode = df_data[df_data["episode_index"] == idx_episode_source].reset_index(drop=True)

        # Load video frames for this episode
        dict_video_frames: dict[str, list[np.ndarray]] = {}
        for key_video in features_video:
            chunk_video = ep_meta[f"videos/{key_video}/chunk_index"]
            file_video = ep_meta[f"videos/{key_video}/file_index"]
            path_video = path_root_source / DEFAULT_VIDEO_PATH.format(
                video_key=key_video,
                chunk_index=chunk_video,
                file_index=file_video,
            )
            print(f"  Decoding video: {key_video}")
            dict_video_frames[key_video] = _decode_video_frames(path_video)

        # Select frames to keep (every Nth frame)
        count_frames_source = len(df_episode)
        indices_keep = [j for j in range(count_frames_source) if j % factor_downsample == 0]

        print(f"  Keeping {len(indices_keep)}/{count_frames_source} frames")

        # Add each selected frame to the output dataset
        for idx_frame in indices_keep:
            row = df_episode.iloc[idx_frame]

            frame: dict[str, str | np.ndarray] = {"task": task_str}

            # Add non-video features
            for key_feature, spec_feature in info_source["features"].items():
                if spec_feature.get("dtype") == "video":
                    continue
                if key_feature in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
                    continue  # These are auto-generated
                if key_feature in df_episode.columns:
                    val = row[key_feature]
                    if isinstance(val, np.ndarray):
                        frame[key_feature] = val
                    else:
                        frame[key_feature] = np.array(val)

            # Add video frames
            for key_video in features_video:
                frame[key_video] = dict_video_frames[key_video][idx_frame]

            ds_output.add_frame(frame)

        # Save the episode (this encodes videos and computes stats)
        ds_output.save_episode()

    # Finalize the dataset
    ds_output.finalize()

    print(f"\nDownsampling complete!")
    print(f"  Episodes: {count_episodes}")
    print(f"  Total frames: {ds_output.meta.total_frames}")
    print(f"  FPS: {fps_source} -> {fps_output}")
    print(f"  Output: {path_root_output}")

    # Push to HuggingFace Hub if requested
    if push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        ds_output.push_to_hub()
        print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample a LeRobot dataset by selecting every Nth frame.")
    parser.add_argument(
        "--repo-id-source",
        type=str,
        required=True,
        help="Source dataset repo_id (e.g., user/dataset_30fps)",
    )
    parser.add_argument(
        "--repo-id-output",
        type=str,
        required=True,
        help="Output dataset repo_id (e.g., user/dataset_10fps)",
    )
    parser.add_argument(
        "--fps-output",
        type=int,
        required=True,
        help="Target frame rate (must evenly divide source fps)",
    )
    parser.add_argument(
        "--path-root-source",
        type=Path,
        default=None,
        help="Optional root path for source dataset (defaults to HF_LEROBOT_HOME/repo_id)",
    )
    parser.add_argument(
        "--path-root-output",
        type=Path,
        default=None,
        help="Optional root path for output dataset (defaults to HF_LEROBOT_HOME/repo_id)",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        choices=["libsvtav1", "h264", "hevc"],
        help="Video codec for encoding (default: libsvtav1)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to HuggingFace Hub after creation",
    )

    args = parser.parse_args()

    downsample_dataset(
        repo_id_source=args.repo_id_source,
        repo_id_output=args.repo_id_output,
        fps_output=args.fps_output,
        path_root_source=args.path_root_source,
        path_root_output=args.path_root_output,
        vcodec=args.vcodec,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
