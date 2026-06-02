"""pick_targets -- top-N cell selection with pixel-to-stage conversion.

Reads masks and the analysis image from segment_tile, computes
regionprops, selects top-N cells by a feature, and converts each
centroid from pixel coordinates to absolute stage coordinates.

Inputs
    pipeline_data["segment_tile"]["image_2d"] : ndarray
    pipeline_data["segment_tile"]["masks"] : ndarray
    pipeline_data["segment_tile"]["image_size_px"] : (nx, ny)
    pipeline_data["input"]["tile_id"] : (region, row, col)
    pipeline_data["input"]["tile_stage_xy_um"] : (x, y)
    pipeline_data["input"]["tile_zwide_um"] : float
    pipeline_data["input"]["source_pixel_size_um"] : (pw, ph)
    pipeline_data["input"]["image_to_stage"] : [[a, b], [c, d]]
    pipeline_data["input"]["n_picks"] : int | None
        Number of picks. None returns all cells sorted by label.
    pipeline_data["input"]["feature"] : str

Outputs (under pipeline_data["pick_targets"])
    picks : list[dict]
        Each dict matches the schema expected by
        workflow.overview._collect_picks_from_results.
"""

METADATA = {"max_workers": 1}

VALID_FEATURES = {"area", "mean_intensity", "eccentricity"}
SUPPORTS_NONE_NPICKS = True


def run(pipeline_data: dict, state: dict, **params) -> dict:
    import numpy as np
    from skimage.measure import regionprops

    seg = pipeline_data["segment_tile"]
    inp = pipeline_data["input"]
    verbose = pipeline_data.get("metadata", {}).get("verbose", 0)

    masks = seg["masks"]
    n_cells = seg["n_cells"]

    if masks is None or n_cells == 0:
        pipeline_data["pick_targets"] = {"picks": []}
        return pipeline_data

    image_2d = seg["image_2d"]
    nx, ny = seg["image_size_px"]

    tile_id = inp["tile_id"]
    tile_stage_xy = inp["tile_stage_xy_um"]
    tile_zwide = inp["tile_zwide_um"]
    pw, ph = inp["source_pixel_size_um"]
    M = np.array(inp["image_to_stage"])
    n_picks = inp.get("n_picks", params.get("n_picks", 5))
    feature = inp.get("feature", params.get("feature", "area"))

    if feature not in VALID_FEATURES:
        raise ValueError(
            f"Unknown feature {feature!r}. "
            f"Expected one of {sorted(VALID_FEATURES)}.")

    regions = regionprops(masks, intensity_image=image_2d)

    cells = []
    for prop in regions:
        row, col = prop.centroid
        bbox = prop.bbox
        w_px = bbox[3] - bbox[1]
        h_px = bbox[2] - bbox[0]
        if hasattr(prop, "intensity_mean"):
            mean_intensity = prop.intensity_mean
        else:
            mean_intensity = prop.mean_intensity

        offset_px = np.array([col - nx / 2, row - ny / 2])
        offset_um = offset_px * np.array([float(pw), float(ph)])
        delta_xy = M @ offset_um
        cell_x = float(tile_stage_xy[0]) + float(delta_xy[0])
        cell_y = float(tile_stage_xy[1]) + float(delta_xy[1])

        cells.append({
            "label": int(prop.label),
            "centroid_col_row_px": (float(col), float(row)),
            "bbox_px": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            "bbox_um": (float(w_px * pw), float(h_px * ph)),
            "area_px": int(prop.area),
            "eccentricity": float(prop.eccentricity),
            "mean_intensity": float(mean_intensity),
            "cell_source_stage_xy_um": (cell_x, cell_y),
        })

    if n_picks is None:
        # Return all cells in a deterministic order for downstream sampling.
        selected = sorted(cells, key=lambda c: c["label"])
    else:
        feature_values = [c[feature if feature != "area" else "area_px"] for c in cells]
        order = np.argsort(feature_values)[::-1]
        n = min(int(n_picks), len(cells))
        selected = [cells[i] for i in order[:n]]

    picks = []
    for c in selected:
        picks.append({
            "pick_id": (
                str(tile_id[0]),
                int(tile_id[1]),
                int(tile_id[2]),
                c["label"],
            ),
            "tile_stage_xy_um": (float(tile_stage_xy[0]), float(tile_stage_xy[1])),
            "tile_zwide_um": float(tile_zwide),
            "source_pixel_size_um": (float(pw), float(ph)),
            "source_image_size_px": (int(nx), int(ny)),
            "centroid_col_row_px": c["centroid_col_row_px"],
            "bbox_px": c["bbox_px"],
            "bbox_um": c["bbox_um"],
            "area_px": c["area_px"],
            "eccentricity": c["eccentricity"],
            "mean_intensity": c["mean_intensity"],
            "cell_source_stage_xy_um": c["cell_source_stage_xy_um"],
        })

    if verbose >= 2:
        print(f"  [pick_targets] {len(picks)}/{len(cells)} picks by {feature}")

    pipeline_data["pick_targets"] = {"picks": picks}
    return pipeline_data
