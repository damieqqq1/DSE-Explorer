"""DSE evaluation metrics and Pareto-front visualisation tools for the developer mode."""

from __future__ import annotations

import math
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Try to use numpy for faster computation; fall back to pure Python.
try:
    import numpy as np  # type: ignore[import-untyped]
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# dominance / non-dominated helpers
# ---------------------------------------------------------------------------


def _dominates(a: list[float], b: list[float]) -> bool:
    """Return True if *a* Pareto-dominates *b* (minimisation)."""
    better = False
    for va, vb in zip(a, b, strict=True):
        if va > vb:
            return False
        if va < vb:
            better = True
    return better


def _non_dominated(points: list[list[float]]) -> list[list[float]]:
    """Keep only non-dominated points."""
    result: list[list[float]] = []
    for p in points:
        if not any(_dominates(q, p) for q in points):
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# distance helpers
# ---------------------------------------------------------------------------


def _euclidean(a: list[float], b: list[float]) -> float:
    total = 0.0
    for va, vb in zip(a, b, strict=True):
        total += (va - vb) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# 2-D Hypervolume (exact)
# ---------------------------------------------------------------------------


def _hypervolume_2d(points: list[list[float]], ref: list[float]) -> float:
    """Exact hypervolume for two objectives."""
    nd = _non_dominated(points)
    if not nd:
        return 0.0
    nd.sort(key=lambda p: p[0])
    hv = 0.0
    prev_f1 = nd[0][0]
    min_f2 = nd[0][1]
    hv += (ref[0] - prev_f1) * (ref[1] - min_f2)
    for i in range(1, len(nd)):
        hv += (ref[0] - nd[i][0]) * (nd[i - 1][1] - nd[i][1])
    return max(hv, 0.0)


def _hypervolume_numpy_nd(points_arr: Any, ref_arr: Any) -> float:  # pragma: no cover
    """Monte-Carlo hypervolume for n-dim points using numpy."""
    nd = points_arr[~np.any(np.all(points_arr[:, None] <= points_arr, axis=2)
                             & np.any(points_arr[:, None] < points_arr, axis=2),
                             axis=1)]
    if nd.shape[0] == 0:
        return 0.0
    samples = 200_000
    dim = nd.shape[1]
    mins = nd.min(axis=0)
    rng = np.random.default_rng(42)
    pts = rng.uniform(size=(samples, dim)) * (ref_arr - mins) + mins
    hits = np.any(np.all(pts[:, None] >= nd[None, :], axis=2), axis=1)
    volume = np.prod(ref_arr - mins)
    return float(hits.mean() * volume)


def _compute_hv(points: list[list[float]], ref: list[float]) -> float:
    if not points or not ref:
        return 0.0
    dim = len(ref)
    if dim == 2:
        return _hypervolume_2d(points, ref)
    if _HAS_NUMPY:
        return _hypervolume_numpy_nd(np.array(points, dtype=float), np.array(ref, dtype=float))
    return -1.0  # sentinel: not computable without numpy for dim > 2


# ---------------------------------------------------------------------------
# IGD
# ---------------------------------------------------------------------------


def _compute_igd(found: list[list[float]], true_pf: list[list[float]]) -> float:
    if not true_pf or not found:
        return 0.0
    total = 0.0
    for tp in true_pf:
        total += min(_euclidean(tp, fp) for fp in found)
    return total / len(true_pf)


# ---------------------------------------------------------------------------
# GD
# ---------------------------------------------------------------------------


def _compute_gd(found: list[list[float]], true_pf: list[list[float]]) -> float:
    if not true_pf or not found:
        return 0.0
    total = 0.0
    for fp in found:
        total += min(_euclidean(fp, tp) for tp in true_pf)
    return total / len(found)


# ---------------------------------------------------------------------------
# Spread  (Deb et al., 2002)
# ---------------------------------------------------------------------------


def _compute_spread(found: list[list[float]], true_pf: list[list[float]]) -> float:
    if len(found) < 2 or not true_pf:
        return 0.0
    nd = _non_dominated(found)
    if len(nd) < 2:
        return 0.0
    nd.sort(key=lambda p: p[0])

    # extreme distances
    d_extreme = 0.0
    for dim_idx in range(len(found[0])):
        nd_dim = sorted(nd, key=lambda p: p[dim_idx])
        d_extreme += _euclidean(nd_dim[0], true_pf[0]) if len(true_pf) > 0 else 0.0
        d_extreme += _euclidean(nd_dim[-1], true_pf[-1]) if len(true_pf) > 1 else 0.0

    distances = [_euclidean(nd[i], nd[i + 1]) for i in range(len(nd) - 1)]
    d_avg = sum(distances) / len(distances)
    d_sum = sum(abs(d - d_avg) for d in distances)

    denominator = d_extreme + (len(nd) - 1) * d_avg
    if denominator == 0:
        return 0.0
    return (d_extreme + d_sum) / denominator


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


def compute_dse_metrics(
    points: list[list[float]],
    reference_point: list[float] | None = None,
    true_pareto_front: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Compute standard DSE/MOO evaluation metrics for a solution set.

    All objectives are assumed to be **minimised**.

    Args:
        points: Solution set as a list of objective vectors, e.g. ``[[0.3, 0.8], [0.5, 0.4]]``.
        reference_point: Reference point for Hypervolume calculation (required for HV).
        true_pareto_front: True Pareto front for IGD / GD / Spread calculation.

    Returns:
        A dictionary with computed metrics and basic statistics.
    """
    if not points:
        raise ValueError("points must not be empty.")
    dim = len(points[0])
    if not all(len(p) == dim for p in points):
        raise ValueError("All points must have the same number of objectives.")
    if reference_point is not None and len(reference_point) != dim:
        raise ValueError("reference_point must have the same dimension as points.")
    if true_pareto_front is not None:
        if not all(len(p) == dim for p in true_pareto_front):
            raise ValueError("true_pareto_front points must have the same dimension.")
        true_pareto_front = _non_dominated(true_pareto_front)

    nd = _non_dominated(points)
    metrics: dict[str, Any] = {
        "num_points": len(points),
        "num_non_dominated": len(nd),
        "non_dominated_points": nd,
    }

    if reference_point is not None:
        hv = _compute_hv(points, reference_point)
        if hv == -1.0:
            metrics["hypervolume"] = "requires numpy for dimension > 2"
        else:
            metrics["hypervolume"] = round(hv, 6)

    if true_pareto_front:
        metrics["igd"] = round(_compute_igd(points, true_pareto_front), 6)
        metrics["gd"] = round(_compute_gd(points, true_pareto_front), 6)
        metrics["spread"] = round(_compute_spread(points, true_pareto_front), 6)

    return {"success": True, **metrics}


# ---------------------------------------------------------------------------
# Pareto-front plot (delegates to code sandbox for rendering)
# ---------------------------------------------------------------------------

def _format_list(py_list: list[list[float]]) -> str:
    return repr(py_list)


def _format_labels(labels: list[str] | None) -> str:
    return repr(labels) if labels else "None"


PARETO_PLOT_SCRIPT = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

points = {points}
labels = {labels}
nd = {nd}
tf = {tf}

fig, ax = plt.subplots(figsize=(8, 6))

# all points
xs = [p[0] for p in points]
ys = [p[1] for p in points]
ax.scatter(xs, ys, c='steelblue', alpha=0.6, s=40, zorder=5)

# non-dominated front
if nd:
    nd_x = [p[0] for p in nd]
    nd_y = [p[1] for p in nd]
    nd_pairs = sorted(zip(nd_x, nd_y), key=lambda v: v[0])
    nd_x, nd_y = zip(*nd_pairs) if nd_pairs else ([], [])
    ax.plot(nd_x, nd_y, 'o-', c='crimson', lw=2, markersize=6, label='Non-dominated', zorder=10)

# true Pareto front (for comparison)
if tf:
    tf_x = [p[0] for p in tf]
    tf_y = [p[1] for p in tf]
    tf_pairs = sorted(zip(tf_x, tf_y), key=lambda v: v[0])
    tf_x, tf_y = zip(*tf_pairs) if tf_pairs else ([], [])
    ax.plot(tf_x, tf_y, 's--', c='darkgreen', lw=2, markersize=6, label='True PF', zorder=8)

# labels
if labels:
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(labels[i], (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8, alpha=0.8)

ax.set_xlabel({x_label!r}, fontsize=12)
ax.set_ylabel({y_label!r}, fontsize=12)
ax.set_title({title!r}, fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
fig.tight_layout()
fig.savefig('pareto_front.png', dpi=120)
print('Pareto front plot saved.')
"""


def generate_pareto_front(
    points: list[list[float]],
    labels: list[str] | None = None,
    title: str = "Pareto Front",
    x_label: str = "Objective 1 (minimise)",
    y_label: str = "Objective 2 (minimise)",
    true_pareto_front: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Generate a 2-D Pareto-front scatter plot via the code sandbox.

    Args:
        points: Objective vectors, e.g. ``[[0.3, 0.8], [0.5, 0.4], ...]``.
        labels: Optional point labels (same length as *points*).
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        true_pareto_front: Optional true Pareto front points for comparison.

    Returns:
        A dictionary with the plot image (base64) and non-dominated point indices.
    """
    if not points:
        raise ValueError("points must not be empty.")
    if len(points[0]) != 2:
        raise ValueError("generate_pareto_front supports 2-objective plots only.")

    if labels is not None and len(labels) != len(points):
        raise ValueError("labels must have the same length as points.")
    if true_pareto_front is not None and len(true_pareto_front[0]) != 2:
        raise ValueError("true_pareto_front must have 2 objectives.")

    nd = _non_dominated(points)
    nd_indices = [i for i, p in enumerate(points) if p in nd]

    tf = _non_dominated(true_pareto_front) if true_pareto_front else []

    script = PARETO_PLOT_SCRIPT.format(
        points=_format_list(points),
        labels=_format_labels(labels),
        nd=_format_list(nd),
        tf=_format_list(tf),
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    from mcp_server.tools.code_sandbox import execute_python_code

    result = execute_python_code(script, timeout=30)

    return {
        "success": result["success"],
        "num_points": len(points),
        "num_non_dominated": len(nd),
        "non_dominated_indices": nd_indices,
        "images": result.get("images", []),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def register_dse_metrics_tools(mcp: Any) -> None:
    """Register DSE metrics and plotting tools on a FastMCP server instance."""

    mcp.tool(
        name="compute_dse_metrics",
        description=(
            "Compute DSE/MOO evaluation metrics for a set of objective vectors: "
            "Hypervolume (HV), Inverted Generational Distance (IGD), Generational "
            "Distance (GD), and Spread. All objectives are assumed to be minimised. "
            "Provide a reference point for HV and a true Pareto front for IGD/GD/Spread. "
            "Use this when you need to evaluate solution quality, compare algorithm "
            "performance, or validate reproduced results."
        ),
    )(compute_dse_metrics)

    mcp.tool(
        name="generate_pareto_front",
        description=(
            "Generate a 2-D Pareto-front scatter plot. Provide objective vectors "
            "and optionally the true Pareto front for comparison. Non-dominated "
            "points are automatically highlighted in red. Returns the plot as a "
            "base64 PNG image. Use this to visualise trade-offs between two "
            "objectives."
        ),
    )(generate_pareto_front)
