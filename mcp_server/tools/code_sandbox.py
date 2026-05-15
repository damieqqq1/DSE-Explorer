"""Code execution sandbox tool for algorithm validation and visualization."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120

PLOT_PREAMBLE = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import os as _os\n"
    "__sandbox_dir = _os.path.dirname(_os.path.abspath(__file__))\n"
)


def execute_python_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute Python code in an isolated subprocess sandbox.

    Use this to run DSE/MOO algorithm snippets, evaluate formulas, or generate
    visualizations (matplotlib). Save plots with ``plt.savefig('plot.png')``
    to return them as base64-encoded images.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds, capped at 120.

    Returns:
        A dictionary with ``stdout``, ``stderr``, ``images`` (base64 data-uris),
        and ``success`` (bool).
    """
    safe_timeout = min(max(1, timeout), MAX_TIMEOUT)

    full_code = code
    if any(kw in code for kw in ("matplotlib", "plt.", "seaborn", "plotly")):
        full_code = PLOT_PREAMBLE + "\n" + code

    script = full_code.replace("\r\n", "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=safe_timeout,
                cwd=tmpdir,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {safe_timeout} seconds.",
                "images": [],
            }

        images: list[str] = []
        for fname in sorted(os.listdir(tmpdir)):
            if fname == "script.py":
                continue
            lower = fname.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
                fpath = os.path.join(tmpdir, fname)
                try:
                    with open(fpath, "rb") as img_file:
                        b64 = base64.b64encode(img_file.read()).decode("ascii")
                    ext = fname.rsplit(".", 1)[-1]
                    mime = {"svg": "svg+xml", "jpg": "jpeg"}.get(ext, ext)
                    images.append(f"data:image/{mime};base64,{b64}")
                except OSError:
                    pass

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "images": images,
        }


def register_code_sandbox_tools(mcp: Any) -> None:
    """Register code sandbox tools on a FastMCP server instance."""

    mcp.tool(
        name="execute_python_code",
        description=(
            "Execute Python code in a sandboxed subprocess. Use this to run DSE/MOO "
            "algorithm snippets, validate formulas, compute metrics (Hypervolume, IGD, "
            "etc.), or generate visualizations with matplotlib. Save plots to disk "
            "(e.g. plt.savefig('plot.png')) to have them returned as images. "
            "The sandbox has numpy, scipy, and matplotlib available."
        ),
    )(execute_python_code)
