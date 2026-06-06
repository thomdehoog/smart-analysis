"""CLI runner for the target_acquisition workflow.

Run from an environment with the engine installed. On the microscope
workstation the Cellpose segmentation worker uses the LAS X driver
environment, ``lasxapi_extended``.

Examples:
    python workflows/target_acquisition/run_pipeline.py
    python workflows/target_acquisition/run_pipeline.py --image-path tile.ome.tiff
    python workflows/target_acquisition/run_pipeline.py --all
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import Engine


WORKFLOW_DIR = Path(__file__).resolve().parent
YAML_PATH = WORKFLOW_DIR / "pipelines" / "overview.yaml"


def _write_sample_image() -> Path:
    from skimage.data import human_mitosis
    import tifffile

    path = Path(tempfile.gettempdir()) / "smart_analysis_target_sample.ome.tiff"
    tifffile.imwrite(path, human_mitosis(), photometric="minisblack")
    return path


def _image_size_px(path: Path) -> tuple[int, int]:
    import tifffile

    shape = tifffile.imread(path).shape
    if len(shape) == 2:
        ny, nx = shape
    elif len(shape) == 3 and shape[0] <= 4:
        _, ny, nx = shape
    elif len(shape) == 3 and shape[2] <= 4:
        ny, nx, _ = shape
    else:
        raise SystemExit(f"Cannot infer 2D image size from shape {shape}.")
    return int(nx), int(ny)


def _parse_matrix(text: str) -> list[list[float]]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "expected four comma-separated values, e.g. 1,0,0,1"
        )
    return [[values[0], values[1]], [values[2], values[3]]]


def _build_payload(args, image_path: Path) -> dict:
    n_picks = None if args.all else args.n_picks
    return {
        "image_path": str(image_path),
        "tile_id": (args.region, args.row, args.col),
        "tile_stage_xy_um": (args.stage_x_um, args.stage_y_um),
        "tile_zwide_um": args.zwide_um,
        "source_pixel_size_um": (args.pixel_size_um, args.pixel_size_um),
        "source_image_size_px": _image_size_px(image_path),
        "image_to_stage": args.image_to_stage,
        "n_picks": n_picks,
        "feature": args.feature,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run target_acquisition on one overview tile."
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to a TIFF tile. Defaults to skimage.data.human_mitosis.",
    )
    parser.add_argument("--region", default="R0")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--col", type=int, default=0)
    parser.add_argument("--stage-x-um", type=float, default=0.0)
    parser.add_argument("--stage-y-um", type=float, default=0.0)
    parser.add_argument("--zwide-um", type=float, default=0.0)
    parser.add_argument("--pixel-size-um", type=float, default=1.0)
    parser.add_argument(
        "--image-to-stage",
        type=_parse_matrix,
        default=[[1.0, 0.0], [0.0, 1.0]],
        help="2x2 image-to-stage matrix as a,b,c,d (default: identity).",
    )
    parser.add_argument(
        "--feature",
        default="area",
        choices=["area", "mean_intensity", "eccentricity"],
    )
    parser.add_argument("--n-picks", type=int, default=5)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Return all segmented cells instead of the top N.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path) if args.image_path else _write_sample_image()
    payload = _build_payload(args, image_path)

    print(f"Pipeline:      {YAML_PATH}")
    print(f"Image:         {image_path}")
    print(f"Tile:          {payload['tile_id']}")
    print(f"Selection:     {args.feature}, n_picks={payload['n_picks']}")
    print()

    with Engine() as engine:
        engine.register("overview", str(YAML_PATH))
        engine.submit("overview", payload)

        while True:
            results = engine.results("overview")
            if results:
                break
            status = engine.status("overview")
            if status["failed"] > 0:
                print("Pipeline failed:")
                for failure in status["failures"]:
                    print(f"  Step:  {failure['step']}")
                    print(f"  Error: {failure['error'][:500]}")
                sys.exit(1)
            time.sleep(0.2)

    picks = results[0]["pick_targets"]["picks"]

    print("=" * 60)
    print("  Result")
    print("=" * 60)
    print(f"  Cells picked: {len(picks)}")
    for pick in picks[:20]:
        x, y = pick["cell_source_stage_xy_um"]
        print(
            f"    {pick['pick_id']}  "
            f"stage=({x:.2f}, {y:.2f}) um  "
            f"area={pick['area_px']} px"
        )
    if len(picks) > 20:
        print(f"    ... {len(picks) - 20} more")
    print()


if __name__ == "__main__":
    main()
