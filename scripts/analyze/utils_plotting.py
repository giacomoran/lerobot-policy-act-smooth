"""Shared plotting utilities for evaluation visualizations.

Provides consistent theming, saving, and data extraction utilities for plotnine-based plots.
"""

from pathlib import Path

import numpy as np
from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    theme,
    theme_bw,
)


def theme_publication() -> theme:
    """Create a publication-quality theme for plotnine plots.

    Returns:
        plotnine theme with clean, minimal styling suitable for publications.
    """
    return theme_bw() + theme(
        # Figure background
        plot_background=element_rect(fill="white", color="white"),
        panel_background=element_rect(fill="white"),
        # Grid
        panel_grid_major=element_line(color="#E0E0E0", size=0.3),
        panel_grid_minor=element_blank(),
        # Axis
        axis_line=element_line(color="#333333", size=0.5),
        axis_ticks=element_line(color="#333333", size=0.3),
        axis_text=element_text(color="#333333", size=9),
        axis_title=element_text(color="#333333", size=10),
        # Legend
        legend_background=element_rect(fill="white", alpha=0.9),
        legend_key=element_rect(fill="white"),
        legend_text=element_text(size=8),
        legend_title=element_text(size=9),
        # Strip (for facets)
        strip_background=element_rect(fill="#F5F5F5"),
        strip_text=element_text(size=9, color="#333333"),
        # Title
        plot_title=element_text(size=11, face="bold"),
        # Margins
        plot_margin=0.02,
    )


def save_plot(
    plot,
    path_output: Path | str,
    width: float = 12,
    height: float = 8,
    dpi: int = 300,
) -> None:
    """Save a plotnine plot with publication-quality settings.

    Args:
        plot: plotnine ggplot object to save.
        path_output: Path to save the plot (supports PNG, PDF, SVG).
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Resolution for raster formats (PNG).
    """
    path_output = Path(path_output)
    path_output.parent.mkdir(parents=True, exist_ok=True)

    plot.save(
        path_output,
        width=width,
        height=height,
        dpi=dpi,
        verbose=False,
    )


def extract_scalar_from_rerun(value) -> float | None:
    """Extract a scalar value from rerun data.

    Handles various formats returned by rerun dataframe queries:
    - list: extract first element
    - numpy array: convert to scalar
    - scalar types: return directly

    Args:
        value: Value from rerun dataframe.

    Returns:
        Extracted float value, or None if extraction fails.
    """
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) > 0:
            return float(value[0])
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        if value.size > 0:
            return float(value.flat[0])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
