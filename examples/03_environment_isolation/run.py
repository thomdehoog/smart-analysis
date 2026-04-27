"""03 -- Environment isolation: the engine's headline feature.

Different steps run in different conda envs. This means a single pipeline
can mix, say, a TensorFlow step, a PyTorch step, and a legacy OpenCV step
without their deps fighting on a single Python install.

What this teaches:
    - METADATA["environment"] selects the conda env for a step
    - Steps with no METADATA inherit the orchestrator's env
    - sys.executable inside each step proves they ran in different procs

Prerequisites:
    - The conda env 'SMART--basic_test--env_a' must exist on this machine.
      If missing, this example exits cleanly with a skip message.

Run:
    python examples/03_environment_isolation/run.py

Expected output:
    here:       <orchestrator python.exe>
    elsewhere:  <SMART--basic_test--env_a python.exe>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import Engine

HERE = Path(__file__).parent
TARGET_ENV = "SMART--basic_test--env_a"


def env_exists(name):
    """Use 'conda info --json' (via CONDA_EXE) to see if env is installed."""
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    try:
        out = subprocess.run(
            [conda_exe, "info", "--json"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return False
    if out.returncode != 0:
        return False
    info = json.loads(out.stdout)
    return any(Path(p).name == name for p in info.get("envs", []))


def main():
    if not env_exists(TARGET_ENV):
        print(f"SKIP: conda env '{TARGET_ENV}' not found on this machine.")
        return

    with Engine() as e:
        e.register("isol", HERE / "pipeline.yaml")
        e.submit("isol", {})

        while not (results := e.results("isol")):
            time.sleep(0.05)

    r = results[0]
    print(f"here:       {r['here']['executable']}")
    print(f"elsewhere:  {r['elsewhere']['executable']}")


if __name__ == "__main__":
    main()
