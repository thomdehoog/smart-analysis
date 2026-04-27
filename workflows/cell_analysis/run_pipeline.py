"""CLI runner for the cell_analysis pipeline.

Run from a cellpose-capable conda environment::

    python workflows/cell_analysis/run_pipeline.py
    python workflows/cell_analysis/run_pipeline.py --source path/to/img.tif
    python workflows/cell_analysis/run_pipeline.py --feature mean_intensity --top-n 10

The CLI writes a temporary YAML with your selection parameters and
registers that. To run the workflow with the shipped defaults, point at
``pipelines/cell_analysis_pipeline.yaml`` directly via the engine API.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Add repo root to sys.path so this script runs from a fresh clone
# without installing the engine package.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import Engine


WORKFLOW_DIR = Path(__file__).resolve().parent
STEPS_DIR = WORKFLOW_DIR / "steps"


def _build_yaml(args) -> str:
    """Render a one-off YAML using the CLI's selection parameters."""
    select_lines = [
        f'      feature: "{args.feature}"',
        f'      mode: "{args.mode}"',
        f'      direction: "{args.direction}"',
    ]
    if args.mode == "percentile":
        p = args.percentile if args.percentile is not None else 95
        select_lines.append(f"      percentile: {p}")
    elif args.mode == "threshold":
        if args.threshold is None:
            raise SystemExit("--mode threshold requires --threshold")
        select_lines.append(f"      threshold: {args.threshold}")
    elif args.mode == "top_n":
        if args.top_n is None:
            raise SystemExit("--mode top_n requires --top-n")
        select_lines.append(f"      top_n: {args.top_n}")

    select_block = "\n".join(select_lines)
    gpu = "true" if args.gpu else "false"

    return (
        f'metadata:\n'
        f'  purpose: "Cell analysis (CLI run)"\n'
        f'  version: "1.0"\n'
        f'  verbose: 2\n'
        f'  functions_dir: "{STEPS_DIR.as_posix()}"\n'
        f'\n'
        f'cell_analysis:\n'
        f'  - preprocess:\n'
        f'      sigma: 1.0\n'
        f'      clip_limit: 0.03\n'
        f'\n'
        f'  - segment:\n'
        f'      diameter: null\n'
        f'      gpu: {gpu}\n'
        f'\n'
        f'  - extract_features:\n'
        f'\n'
        f'  - select_features:\n'
        f'{select_block}\n'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the cell_analysis pipeline on one image.",
    )
    parser.add_argument(
        "--source", default="skimage.human_mitosis",
        help="Image source: a file path, or 'skimage.<dataset>'.",
    )
    parser.add_argument(
        "--feature", default="area",
        help="regionprops feature to rank cells by (default: area).",
    )
    parser.add_argument(
        "--mode", default=None, choices=["percentile", "threshold", "top_n"],
        help="Selection mode (default: percentile, or inferred from "
             "--threshold / --top-n).",
    )
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-n", dest="top_n", type=int, default=None)
    parser.add_argument(
        "--direction", default="high", choices=["high", "low"],
    )
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()

    # Infer mode from which selection arg the user provided.
    if args.mode is None:
        if args.threshold is not None:
            args.mode = "threshold"
        elif args.top_n is not None:
            args.mode = "top_n"
        else:
            args.mode = "percentile"

    yaml_text = _build_yaml(args)
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    tmp_yaml = Path(tmp_path)
    tmp_yaml.write_text(yaml_text)

    print(f"Source:        {args.source}")
    print(f"Selection:     {args.feature} {args.direction} {args.mode}")
    print()

    try:
        with Engine() as e:
            e.register("ca", str(tmp_yaml))
            e.submit("ca", {"data_source": args.source})

            while True:
                results = e.results("ca")
                if results:
                    break
                status = e.status("ca")
                if status["failed"] > 0:
                    f = status["failures"][0]
                    print(f"FAILED at step '{f['step']}':")
                    print(f"  {f['error'][:400]}")
                    sys.exit(1)
                time.sleep(0.2)
    finally:
        try:
            tmp_yaml.unlink()
        except OSError:
            pass

    r = results[0]
    sel = r["select_features"]

    print("=" * 60)
    print("  Result")
    print("=" * 60)
    print(f"  Image shape:      {r['preprocess']['shape']}")
    print(f"  Cells segmented:  {r['segment']['n_cells']}")
    print(f"  Cells selected:   {sel['n_selected']} / {sel['n_total']}")
    print(f"  Selection rule:   {sel['feature']} {sel['direction']} "
          f"{sel['mode']} (cutoff={sel['cutoff']})")
    shown = sel["selected_labels"][:20]
    print(f"  Selected labels:  {shown}"
          f"{'  ...' if len(sel['selected_labels']) > 20 else ''}")
    print()


if __name__ == "__main__":
    main()
