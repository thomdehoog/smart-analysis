"""01 -- Hello world: minimal Engine usage.

What this teaches:
    - Engine context-manager lifecycle (with Engine() as e: ...)
    - register() with a YAML path
    - submit() one job
    - results() polling

Prerequisites:
    - pyyaml (engine's only runtime dep)

Run:
    python examples/01_hello_world/run.py

Expected output:
    Doubled: 42
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import Engine

HERE = Path(__file__).parent


def main():
    with Engine() as e:
        e.register("hello", HERE / "pipeline.yaml")
        e.submit("hello", {"n": 21})

        # Poll until the single result lands. Engine work happens in
        # background workers, so we drain the queue until non-empty.
        while not (results := e.results("hello")):
            time.sleep(0.05)

    print(f"Doubled: {results[0]['doubled']}")


if __name__ == "__main__":
    main()
