"""Create conda environments for the object_analysis workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _env_setup import setup_workflow_env  # noqa: E402


WORKFLOW = "object_analysis"
PYTHON_VERSION = "3.12"

STEP_PROFILES = {
    "vision": {
        "description": "Cellpose detection and DINOv2 deep features",
        "install_torch": True,
        "pip_packages": [
            "pyyaml",
            "numpy",
            "tifffile",
            "imagecodecs",
            "pooch",
            "cellpose",
        ],
        "diagnostics": [
            (
                "TIFF IO",
                "import numpy as np, tifffile; "
                "a = np.zeros((8, 8), dtype=np.uint8); "
                "print('OK' if a.shape == (8, 8) else 'FAIL')",
            ),
            ("cellpose", "from cellpose import models; print('OK')"),
            (
                "DINOv2 hub API",
                "import os; os.environ.setdefault('XFORMERS_DISABLED', '1'); "
                "import torch; "
                "print('OK' if hasattr(torch, 'hub') else 'FAIL')",
            ),
        ],
    },
    "classical": {
        "description": "scikit-image classical feature extraction",
        "install_torch": False,
        "pip_packages": [
            "pyyaml",
            "numpy",
            "scikit-image>=0.23",
        ],
        "diagnostics": [
            (
                "scikit-image",
                "from skimage.measure import regionprops_table; "
                "import numpy as np; "
                "m = np.zeros((8, 8), dtype=np.int32); "
                "m[2:4, 2:4] = 1; "
                "regionprops_table(m, properties=('label', 'area')); "
                "print('OK')",
            ),
        ],
    },
}


def _selected_profile() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--step", default="vision")
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
        install_torch=profile["install_torch"],
        default_step="vision",
    )
