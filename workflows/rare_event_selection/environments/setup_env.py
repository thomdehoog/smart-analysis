"""Create conda environments for the rare_event_selection workflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _env_setup import setup_workflow_env  # noqa: E402


WORKFLOW = "rare_event_selection"
PYTHON_VERSION = "3.12"

PIP_PACKAGES = [
    "pyyaml",
    "numpy",
    "scikit-image>=0.23",
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
        "scikit-image",
        "from skimage.filters import gaussian; "
        "import numpy as np; "
        "gaussian(np.zeros((10,10)), sigma=1); "
        "print('OK')",
    ),
    (
        "scipy + torch coexist",
        "from skimage.filters import gaussian; "
        "import numpy as np; "
        "gaussian(np.zeros((10,10)), sigma=1); "
        "import torch; print('OK')",
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
