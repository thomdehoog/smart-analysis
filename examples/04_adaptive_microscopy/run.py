"""04 -- Adaptive microscopy: cheap survey -> targeted re-acquisition.

The live-feedback pattern. Acquire a cheap overview of a region, let a
scoped step decide which spots are worth a closer look, then submit
expensive high-res jobs only for those spots.

What this teaches:
    - Two registered pipelines on one Engine
    - A scoped step that selects regions of interest
    - Driving a second pipeline with results from the first
      (this is the loop in 'live adaptive microscopy')

Run:
    python examples/04_adaptive_microscopy/run.py

Expected output:
    Overview brightnesses: [10, 80, 30, 90]
    Picked 2 spots: [(0, 1), (1, 1)]
    Acquired 2 high-res tiles at: [(0, 1), (1, 1)]
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import Engine

HERE = Path(__file__).parent


def drain_until(engine, name, predicate, interval=0.05):
    """Poll engine.results(name) until predicate(collected_so_far) is true."""
    collected = []
    while not predicate(collected):
        collected.extend(engine.results(name))
        time.sleep(interval)
    return collected


def main():
    overview_tiles = [
        {"row": 0, "col": 0, "brightness": 10},
        {"row": 0, "col": 1, "brightness": 80},
        {"row": 1, "col": 0, "brightness": 30},
        {"row": 1, "col": 1, "brightness": 90},
    ]

    with Engine() as e:
        e.register("overview", HERE / "pipeline_overview.yaml")
        e.register("target", HERE / "pipeline_target.yaml")

        # 1) Cheap overview pass. Every tile shares scope={"region": "A"}.
        #    The last one carries complete="region" -- this triggers the
        #    scoped 'feedback' step on the accumulated batch.
        for i, tile in enumerate(overview_tiles):
            is_last = (i == len(overview_tiles) - 1)
            e.submit(
                "overview", tile,
                scope={"region": "A"},
                complete="region" if is_last else None,
            )

        # 2) Wait for all per-tile results + the one scoped feedback result.
        overview_results = drain_until(
            e, "overview",
            lambda c: len(c) >= len(overview_tiles) + 1,
        )

        feedback = next(r for r in overview_results if "picks" in r)
        picks = feedback["picks"]

        brightnesses = [t["brightness"] for t in overview_tiles]
        print(f"Overview brightnesses: {brightnesses}")
        print(f"Picked {len(picks)} spots: {picks}")

        # 3) Targeted pass: feed picks into the second pipeline.
        for row, col in picks:
            e.submit("target", {"row": row, "col": col})

        target_results = drain_until(
            e, "target", lambda c: len(c) >= len(picks),
        )

    coords = [(r["target"]["row"], r["target"]["col"])
              for r in target_results]
    coords.sort()
    print(f"Acquired {len(coords)} high-res tiles at: {coords}")


if __name__ == "__main__":
    main()
