"""02 -- Scoped aggregation: the canonical tile-and-stitch pattern.

What this teaches:
    - Scope-tagged submissions (scope={"group": "demo"})
    - The complete="group" signal that fires the scoped phase
    - How a scoped step receives pipeline_data["results"] (a list of
      every prior-phase output for that scope group)
    - Two distinct results streams: per-tile (Phase 0) and one
      stitched result (Phase 1)

This is the bread-and-butter pattern for smart microscopy: acquire many
tiles, aggregate them once the region is complete.

Run:
    python examples/02_scoped_aggregation/run.py

Expected output:
    Stitched 5 tiles, sum=15
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import Engine

HERE = Path(__file__).parent


def main():
    values = [1, 2, 3, 4, 5]  # sum -> 15

    with Engine() as e:
        e.register("agg", HERE / "pipeline.yaml")

        # Submit each tile tagged with the same scope group. The last
        # submission carries complete="group" -- this signals the engine
        # to run the scoped 'stitch' step on the accumulated batch.
        for i, v in enumerate(values):
            is_last = (i == len(values) - 1)
            e.submit(
                "agg",
                {"value": v},
                scope={"group": "demo"},
                complete="group" if is_last else None,
            )

        # Wait for tiles + the stitched result. We expect 5 + 1 = 6.
        collected = []
        while len(collected) < len(values) + 1:
            collected.extend(e.results("agg"))
            time.sleep(0.05)

    # Phase 1 (the stitched output) is the one with the 'stitched' key.
    stitched = next(r for r in collected if "stitched" in r)
    s = stitched["stitched"]
    print(f"Stitched {s['n_tiles']} tiles, sum={s['sum']}")


if __name__ == "__main__":
    main()
