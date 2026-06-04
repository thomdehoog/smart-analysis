"""Create conda environments for the target_discovery workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _env_setup import setup_workflow_env  # noqa: E402


WORKFLOW = "target_discovery"
PYTHON_VERSION = "3.12"

STEP_PROFILES = {
    "main": {
        "description": "Lightweight target selection",
        "pip_packages": [
            "pyyaml",
            "numpy",
        ],
        "diagnostics": [
            ("numpy", "import numpy as np; print(np.__version__)"),
        ],
    },
    "cluster": {
        "description": "Embedding clustering, Leiden, and UMAP",
        "pip_packages": [
            "pyyaml",
            "numpy",
            "scipy",
            "scikit-learn",
            "pynndescent",
            "umap-learn",
            "igraph",
            "leidenalg",
        ],
        "diagnostics": [
            ("numpy", "import numpy as np; print(np.__version__)"),
            (
                "nearest neighbors",
                "from sklearn.neighbors import NearestNeighbors; "
                "NearestNeighbors(metric='cosine'); print('OK')",
            ),
            (
                "UMAP",
                "import umap; print(getattr(umap, '__version__', 'OK'))",
            ),
            (
                "Leiden",
                "import igraph, leidenalg; "
                "g = igraph.Graph.Ring(5); "
                "part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition); "
                "print(len(part))",
            ),
        ],
    },
}


def _selected_profile() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--step", default="main")
    args, _ = parser.parse_known_args()
    if args.step not in STEP_PROFILES:
        expected = ", ".join(sorted(STEP_PROFILES))
        raise SystemExit(f"Unknown --step {args.step!r}. Expected one of: {expected}")
    return STEP_PROFILES[args.step]


if __name__ == "__main__":
    profile = _selected_profile()
    setup_workflow_env(
        workflow=WORKFLOW,
        pip_packages=profile["pip_packages"],
        diagnostics=profile["diagnostics"],
        python_version=PYTHON_VERSION,
        install_torch=False,
    )
