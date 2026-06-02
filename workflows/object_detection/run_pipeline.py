"""CLI runner for the object_detection workflow."""

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
from _contracts import save_overview  # noqa: E402


WORKFLOW_DIR = Path(__file__).resolve().parent
YAML_PATH = WORKFLOW_DIR / "pipelines" / "object_detection.yaml"


def _image_size_px(path: Path) -> list[int]:
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
    return [int(nx), int(ny)]


def _parse_pair(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated values")
    return values


def _parse_matrix(text: str) -> list[list[float]]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "expected four comma-separated values, e.g. 1,0,0,1"
        )
    return [[values[0], values[1]], [values[2], values[3]]]


def _parse_tile_id(text: str) -> list:
    parts = [part.strip() for part in text.split(",")]
    out = []
    for part in parts:
        try:
            out.append(int(part))
        except ValueError:
            out.append(part)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Detect and measure all objects in one image tile."
    )
    parser.add_argument("image_path", help="Path to a TIFF tile.")
    parser.add_argument("--tile-id", default="R0,0,0")
    parser.add_argument("--stage-xy-um", type=_parse_pair, default=[0.0, 0.0])
    parser.add_argument("--zwide-um", type=float, default=0.0)
    parser.add_argument("--pixel-size-um", type=_parse_pair, default=[1.0, 1.0])
    parser.add_argument(
        "--image-to-stage",
        type=_parse_matrix,
        default=[[1.0, 0.0], [0.0, 1.0]],
        help="2x2 image-to-stage matrix as a,b,c,d (default: identity).",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument(
        "--save-overview",
        default=None,
        help="Optional path for an overview JSON containing this one tile.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    payload = {
        "image_path": str(image_path),
        "tile_id": _parse_tile_id(args.tile_id),
        "tile_stage_xy_um": args.stage_xy_um,
        "tile_zwide_um": args.zwide_um,
        "source_pixel_size_um": args.pixel_size_um,
        "source_image_size_px": _image_size_px(image_path),
        "image_to_stage": args.image_to_stage,
        "channel": args.channel,
        "diameter": args.diameter,
        "gpu": args.gpu,
    }

    print(f"Pipeline:      {YAML_PATH}")
    print(f"Image:         {image_path}")
    print(f"Tile:          {payload['tile_id']}")
    print()

    with Engine() as engine:
        engine.register("object_detection", str(YAML_PATH))
        engine.submit("object_detection", payload)

        while True:
            results = engine.results("object_detection")
            if results:
                break
            status = engine.status("object_detection")
            if status["failed"]:
                failure = status["failures"][0]
                raise RuntimeError(f"{failure['step']}: {failure['error']}")
            time.sleep(0.2)

    tile = results[0]["object_detection"]
    n_objects = tile["objects"]["n_objects"]
    print("=" * 60)
    print("  Result")
    print("=" * 60)
    print(f"  Objects detected: {n_objects}")

    if args.save_overview:
        path = save_overview(args.save_overview, {"tiles": [tile]})
        print(f"  Overview saved:   {path}")
    print()


if __name__ == "__main__":
    main()
