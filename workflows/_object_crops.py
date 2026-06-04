"""Shared object crop helpers for object-centered workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _contracts import to_builtin
from _object_ids import object_name, tile_name


def extract_object_crops(
    *,
    image,
    masks,
    tile_id,
    context_multiplier: float = 1.5,
    min_crop_size_px: int = 64,
    mode: str = "neighborhood",
    output_dir: str | Path | None = None,
) -> dict:
    """Extract square per-object crops from an image and label mask.

    The crop is centered on the object centroid, sized from the larger bbox
    dimension times ``context_multiplier``, and zero-padded at image borders.
    ``mode='single_cell'`` masks non-object pixels to zero; ``neighborhood``
    preserves local context.
    """
    if mode not in {"neighborhood", "single_cell"}:
        raise ValueError("mode must be 'neighborhood' or 'single_cell'.")
    if context_multiplier <= 0:
        raise ValueError("context_multiplier must be > 0.")
    if min_crop_size_px <= 0:
        raise ValueError("min_crop_size_px must be > 0.")

    image = np.asarray(image)
    masks = np.asarray(masks)
    if image.shape[:2] != masks.shape:
        raise ValueError(
            f"image shape {image.shape[:2]} and masks shape {masks.shape} differ."
        )

    out_root = Path(output_dir) if output_dir else None
    t_name = tile_name(tile_id)
    tile_artifacts = {}
    if out_root:
        tile_dir = out_root / "tiles" / t_name
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_artifacts = _write_tile_artifacts(tile_dir, masks, t_name)

    rows = []
    labels = [int(label) for label in np.unique(masks) if int(label) != 0]
    for label in labels:
        rows_px, cols_px = np.nonzero(masks == label)
        if rows_px.size == 0:
            continue
        bbox = (
            int(rows_px.min()),
            int(cols_px.min()),
            int(rows_px.max()) + 1,
            int(cols_px.max()) + 1,
        )
        centroid = (float(rows_px.mean()), float(cols_px.mean()))
        crop = _crop_one(
            image=image,
            masks=masks,
            label=label,
            bbox=bbox,
            centroid=centroid,
            context_multiplier=float(context_multiplier),
            min_crop_size_px=int(min_crop_size_px),
            mode=mode,
        )
        row = {
            "label": label,
            "object_id": object_name(tile_id, label),
            "tile_name": t_name,
            "bbox_min_row_px": int(bbox[0]),
            "bbox_min_col_px": int(bbox[1]),
            "bbox_max_row_px": int(bbox[2]),
            "bbox_max_col_px": int(bbox[3]),
            "crop_origin_row_px": int(crop["origin_row_px"]),
            "crop_origin_col_px": int(crop["origin_col_px"]),
            "crop_height_px": int(crop["image"].shape[0]),
            "crop_width_px": int(crop["image"].shape[1]),
            "crop_image": crop["image"],
            "crop_mask": crop["mask"],
            "crop_path": None,
            "mask_path": None,
        }
        if out_root:
            _write_object_artifacts(out_root, row)
        rows.append(row)

    return {
        "tile_name": t_name,
        "n_objects": len(rows),
        "objects": rows,
        "crop_policy": {
            "context_multiplier": float(context_multiplier),
            "min_crop_size_px": int(min_crop_size_px),
            "mode": mode,
        },
        "tile_artifacts": tile_artifacts,
    }


def _crop_one(
    *,
    image,
    masks,
    label: int,
    bbox,
    centroid,
    context_multiplier: float,
    min_crop_size_px: int,
    mode: str,
) -> dict:
    min_row, min_col, max_row, max_col = [int(value) for value in bbox]
    bbox_h = max(1, max_row - min_row)
    bbox_w = max(1, max_col - min_col)
    size = max(min_crop_size_px, int(np.ceil(max(bbox_h, bbox_w) * context_multiplier)))
    row_c, col_c = [float(value) for value in centroid]
    row0 = int(np.floor(row_c - size / 2.0))
    col0 = int(np.floor(col_c - size / 2.0))
    row1 = row0 + size
    col1 = col0 + size

    ny, nx = masks.shape
    src_r0 = max(0, row0)
    src_c0 = max(0, col0)
    src_r1 = min(ny, row1)
    src_c1 = min(nx, col1)
    dst_r0 = src_r0 - row0
    dst_c0 = src_c0 - col0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    crop_shape = (size, size) + tuple(image.shape[2:])
    crop_image = np.zeros(crop_shape, dtype=image.dtype)
    crop_mask = np.zeros((size, size), dtype=np.uint8)

    if src_r1 > src_r0 and src_c1 > src_c0:
        crop_image[dst_r0:dst_r1, dst_c0:dst_c1] = image[src_r0:src_r1, src_c0:src_c1]
        crop_mask[dst_r0:dst_r1, dst_c0:dst_c1] = (
            masks[src_r0:src_r1, src_c0:src_c1] == label
        )

    if mode == "single_cell":
        if crop_image.ndim == 2:
            crop_image = np.where(crop_mask, crop_image, 0)
        else:
            crop_image = np.where(crop_mask[..., None], crop_image, 0)

    return {
        "image": crop_image,
        "mask": crop_mask,
        "origin_row_px": row0,
        "origin_col_px": col0,
    }


def _write_tile_artifacts(tile_dir: Path, masks, t_name: str) -> dict:
    import tifffile

    masks_path = tile_dir / "masks.tif"
    metadata_path = tile_dir / "tile.json"
    tifffile.imwrite(masks_path, np.asarray(masks, dtype=np.int32))
    metadata = {"tile_name": t_name, "masks_path": str(masks_path)}
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "masks_path": str(masks_path),
        "tile_json_path": str(metadata_path),
    }


def _write_object_artifacts(out_root: Path, row: dict) -> None:
    import tifffile

    obj_dir = out_root / "objects" / row["object_id"]
    obj_dir.mkdir(parents=True, exist_ok=True)
    crop_path = obj_dir / "crop.tif"
    mask_path = obj_dir / "mask.tif"
    metadata_path = obj_dir / "object.json"
    tifffile.imwrite(crop_path, row["crop_image"])
    tifffile.imwrite(mask_path, row["crop_mask"].astype(np.uint8))
    row["crop_path"] = str(crop_path)
    row["mask_path"] = str(mask_path)

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"crop_image", "crop_mask"}
    }
    metadata["object_json_path"] = str(metadata_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(metadata), handle, indent=2, allow_nan=False)
        handle.write("\n")
