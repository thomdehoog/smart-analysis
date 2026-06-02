"""CLI runner for the target_discovery workflow."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR = ROOT / "workflows"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKFLOWS_DIR))

from engine import Engine  # noqa: E402
from _contracts import load_overview  # noqa: E402


WORKFLOW_DIR = Path(__file__).resolve().parent
YAML_PATH = WORKFLOW_DIR / "pipelines" / "target_discovery.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Select revisit targets from an overview JSON file."
    )
    parser.add_argument("overview_json", help="Path to a saved overview JSON file.")
    parser.add_argument("--feature", default="area")
    parser.add_argument("--direction", default="high", choices=["high", "low"])
    parser.add_argument("--n-per-tile", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Return all valid targets.")
    parser.add_argument("--border-margin-px", type=float, default=0.0)
    parser.add_argument("--area-threshold", type=float, default=None)
    parser.add_argument("--intensity-threshold", type=float, default=None)
    args = parser.parse_args()

    overview = load_overview(args.overview_json)
    n_per_tile = None if args.all else args.n_per_tile
    payload = {
        "tiles": overview["tiles"],
        "feature": args.feature,
        "direction": args.direction,
        "n_per_tile": n_per_tile,
        "border_margin_px": args.border_margin_px,
        "area_threshold": args.area_threshold,
        "intensity_threshold": args.intensity_threshold,
    }

    print(f"Pipeline:      {YAML_PATH}")
    print(f"Overview:      {args.overview_json}")
    print(f"Selection:     {args.feature} {args.direction}, n_per_tile={n_per_tile}")
    print()

    with Engine() as engine:
        engine.register("target_discovery", str(YAML_PATH))
        engine.submit("target_discovery", payload)

        while True:
            results = engine.results("target_discovery")
            if results:
                break
            status = engine.status("target_discovery")
            if status["failed"]:
                failure = status["failures"][0]
                raise RuntimeError(f"{failure['step']}: {failure['error']}")
            time.sleep(0.2)

    targets = results[0]["target_discovery"]["targets"]
    print("=" * 60)
    print("  Result")
    print("=" * 60)
    print(f"  Targets: {len(targets)}")
    for target in targets[:20]:
        print(
            f"    {target['target_id']}  "
            f"stage=({target['stage_x_um']:.2f}, {target['stage_y_um']:.2f}) um  "
            f"{target['source_feature']}={target['score']:.3g}"
        )
    if len(targets) > 20:
        print(f"    ... {len(targets) - 20} more")
    print()


if __name__ == "__main__":
    main()
