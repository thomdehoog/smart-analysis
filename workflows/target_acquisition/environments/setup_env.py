"""Create conda environments for the target_acquisition workflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _env_setup import setup_workflow_env  # noqa: E402


WORKFLOW = "target_acquisition"
PYTHON_VERSION = "3.12"

PIP_PACKAGES = [
    "pyyaml",
    "numpy",
    "scikit-image>=0.23",
    "tifffile",
    "imagecodecs",
    "pooch",
    "cellpose",
]

DIAGNOSTICS = [
    ("PyTorch loads", "import torch; print(torch.__version__)"),
    (
        "GPU backend available",
        "import torch; "
        "cuda = torch.cuda.is_available(); "
        "mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(); "
        "print('CUDA' if cuda else ('MPS' if mps else 'CPU'))",
    ),
    (
        "TIFF IO",
        "import numpy as np, tifffile; "
        "a = np.zeros((8, 8), dtype=np.uint8); "
        "print('OK' if a.shape == (8, 8) else 'FAIL')",
    ),
    (
        "scikit-image",
        "from skimage.measure import regionprops; "
        "import numpy as np; "
        "regionprops(np.zeros((8, 8), dtype=np.int32)); "
        "print('OK')",
    ),
    ("cellpose", "from cellpose import models; print('OK')"),
]


if __name__ == "__main__":
    setup_workflow_env(
        workflow=WORKFLOW,
        pip_packages=PIP_PACKAGES,
        diagnostics=DIAGNOSTICS,
        python_version=PYTHON_VERSION,
    )
