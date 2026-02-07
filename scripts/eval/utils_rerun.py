"""ReRun visualization utilities with slash-separated paths for blueprint matching.

This module provides custom logging functions that use "/" instead of "." in entity paths
to match blueprint path patterns.
"""

import numbers
from threading import RLock

import numpy as np
import rerun as rr


class MetricsTracker:
    """Generic metrics tracker with rerun logging (thread-safe).

    Base class for tracking numeric values over time with optional rerun visualization.
    """

    def __init__(self, rerun_path: str, series_name: str, color: list[int]):
        """Initialize tracker.

        Args:
            rerun_path: Path for rerun logging (e.g., "/metrics/inference_latency")
            series_name: Display name for the series (e.g., "Inference (ms)")
            color: RGB color for the series (e.g., [255, 100, 100])
        """
        self.rerun_path = rerun_path
        self.series_name = series_name
        self.color = color
        self.values: list[float] = []
        self.lock = RLock()

    def setup_rerun(self) -> None:
        """Set up the rerun series. Call once before recording."""
        rr.log(
            self.rerun_path,
            rr.SeriesLines(names=self.series_name, colors=self.color),
            static=True,
        )

    def record(self, value: float, log_to_rerun: bool = True) -> None:
        """Record a single measurement."""
        with self.lock:
            self.values.append(value)
            if log_to_rerun:
                rr.log(self.rerun_path, rr.Scalars(value))

    def get_stats(self) -> dict:
        """Compute summary statistics. Override in subclasses for custom stats."""
        with self.lock:
            if not self.values:
                return {}
            arr = np.array(self.values)
            return {
                "count": len(arr),
                "total": float(np.sum(arr)),
                "mean": float(np.mean(arr)),
            }

    def reset(self) -> None:
        """Reset tracker for a new episode."""
        with self.lock:
            self.values = []


class LatencyTracker(MetricsTracker):
    """Tracks inference latency with detailed statistics."""

    def __init__(self):
        super().__init__("/metrics/inference_latency", "Inference (ms)", [255, 100, 100])

    def get_stats(self) -> dict:
        """Compute detailed latency statistics."""
        with self.lock:
            if not self.values:
                return {}
            arr = np.array(self.values)
            return {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }

    def log_summary_to_rerun(self) -> None:
        """Log summary statistics and histogram to rerun."""
        with self.lock:
            if not self.values:
                return

            stats = self.get_stats()

            summary_text = f"""**Inference Latency Stats**
- Count: {stats["count"]}
- Mean: {stats["mean"]:.1f}ms
- Std: {stats["std"]:.1f}ms
- Min: {stats["min"]:.1f}ms
- Max: {stats["max"]:.1f}ms
- P50: {stats["p50"]:.1f}ms
- P95: {stats["p95"]:.1f}ms
"""
            rr.log(
                "/metrics/latency_summary",
                rr.TextDocument(summary_text, media_type=rr.MediaType.MARKDOWN),
            )

            hist, _ = np.histogram(self.values, bins=20)
            rr.log("/metrics/latency_histogram", rr.BarChart(hist))


class DiscardTracker(MetricsTracker):
    """Tracks discarded actions."""

    def __init__(self):
        super().__init__("/metrics/discarded_actions", "Discarded", [255, 200, 0])

    def get_stats(self) -> dict:
        """Compute discard statistics."""
        with self.lock:
            if not self.values:
                return {}
            arr = np.array(self.values)
            return {
                "count_switches": len(arr),
                "count_total": int(np.sum(arr)),
                "mean": float(np.mean(arr)),
            }


def _is_scalar(x):
    """Check if value is a scalar."""
    return isinstance(x, (float, numbers.Real, np.integer, np.floating)) or (isinstance(x, np.ndarray) and x.ndim == 0)


def log_rerun_data(
    idx_frame: int | None = None,
    observation: dict[str, np.ndarray] | None = None,
    action: dict[str, np.ndarray] | None = None,
    timestep: int | None = None,
    idx_chunk: int | None = None,
) -> None:
    """Log observation and action data to ReRun using slash-separated paths.

    Paths logged:
    - Observations: /observation/state_0, /observation/state_1, ..., /observation/images/wrist, etc.
    - Actions: /action/shoulder_pan/pos, /action/shoulder_lift/pos, etc.
    - Timestep: /timestep (the control timestep, not display frame index)
    - Chunk index: /idx_chunk (the current chunk being executed)

    Args:
        idx_frame: Optional observation-rate frame index. Sets the rerun "frame_nr" timeline
            so all data within a single frame shares the same index.
        observation: Optional dictionary containing observation data to log.
        action: Optional dictionary containing action data to log.
        timestep: Optional control timestep to log.
        idx_chunk: Optional chunk index to log.
    """
    if idx_frame is not None:
        rr.set_time_sequence("frame_nr", idx_frame)

    if timestep is not None:
        rr.log("/timestep", rr.Scalars(float(timestep)))

    if idx_chunk is not None:
        rr.log("/idx_chunk", rr.Scalars(float(idx_chunk)))

    if observation:
        for k, v in observation.items():
            if v is None:
                continue

            # Convert key to use slashes instead of dots
            if k.startswith("observation."):
                key = "/" + k.replace(".", "/")
            elif k.startswith("observation"):
                key = "/observation/" + k.replace("observation", "").lstrip(".")
            else:
                key = "/observation/" + k

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue

            # Convert key to use slashes instead of dots
            if k.startswith("action."):
                key = "/" + k.replace(".", "/")
            else:
                key = "/action/" + k.replace(".", "/")

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
