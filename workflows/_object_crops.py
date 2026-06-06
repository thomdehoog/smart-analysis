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
    crop_size_px: int = 128,
    mask: bool = False,
    drop_incomplete: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    """Extract square per-object crops from an image and label mask.

    The crop is centered on the object centroid, has the same fixed size for
    every object, and is zero-padded at image borders. Set ``mask=True`` to
    zero non-object pixels while still returning the object mask separately.
    When ``drop_incomplete=True``, objects whose own mask touches the image
    boundary or does not fit inside the fixed crop are skipped. Context crops
    (``mask=False``) also require the full crop window to stay inside the
    image; masked crops may use zero padding outside the tile because the
    background is dark by design.
    """
    if crop_size_px <= 0:
        raise ValueError("crop_size_px must be > 0.")

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
    skipped_incomplete = []
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
            centroid=centroid,
            bbox=bbox,
            crop_size_px=int(crop_size_px),
            mask=bool(mask),
        )
        if drop_incomplete and not crop["complete"]:
            skipped_incomplete.append(label)
            continue
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
            "crop_complete": bool(crop["complete"]),
            "crop_in_bounds": bool(crop["crop_in_bounds"]),
            "object_complete": bool(crop["object_complete"]),
            "object_fits_crop": bool(crop["object_fits_crop"]),
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
            "crop_size_px": int(crop_size_px),
            "mask": bool(mask),
            "drop_incomplete": bool(drop_incomplete),
        },
        "skipped_incomplete_labels": skipped_incomplete,
        "tile_artifacts": tile_artifacts,
    }


def _crop_one(
    *,
    image,
    masks,
    label: int,
    centroid,
    bbox,
    crop_size_px: int,
    mask: bool,
) -> dict:
    size = int(crop_size_px)
    row_c, col_c = [float(value) for value in centroid]
    row0 = int(np.floor(row_c - size / 2.0))
    col0 = int(np.floor(col_c - size / 2.0))
    row1 = row0 + size
    col1 = col0 + size

    ny, nx = masks.shape
    crop_in_bounds = row0 >= 0 and col0 >= 0 and row1 <= ny and col1 <= nx
    bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = [int(v) for v in bbox]
    object_complete = (
        bbox_min_row > 0
        and bbox_min_col > 0
        and bbox_max_row < ny
        and bbox_max_col < nx
    )
    object_fits_crop = (
        bbox_min_row >= row0
        and bbox_min_col >= col0
        and bbox_max_row <= row1
        and bbox_max_col <= col1
    )
    complete = object_complete and object_fits_crop and (mask or crop_in_bounds)
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

    if mask:
        if crop_image.ndim == 2:
            crop_image = np.where(crop_mask, crop_image, 0)
        else:
            crop_image = np.where(crop_mask[..., None], crop_image, 0)

    return {
        "image": crop_image,
        "mask": crop_mask,
        "complete": complete,
        "crop_in_bounds": crop_in_bounds,
        "object_complete": object_complete,
        "object_fits_crop": object_fits_crop,
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
