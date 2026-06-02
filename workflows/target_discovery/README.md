# `target_discovery` workflow

Select revisit targets from object-detection tables.

```
discover_targets
```

This workflow is the selection phase. It consumes an aggregated overview:
a list of per-tile object detections, each with its own tile geometry. It
does not segment images, read masks, or recompute object measurements.

## Input Payload

Submit an overview under `tiles`:

```python
{
    "tiles": [tile_detection, tile_detection],
    "feature": "area",
    "direction": "high",
    "n_per_tile": 5,
    "border_margin_px": 10,
    "area_threshold": None,
    "intensity_threshold": None,
}
```

`feature` must name a column in each tile's object table, such as
`area`, `intensity_mean`, or `eccentricity`. `direction` is `high` or
`low`. `border_margin_px` excludes objects whose bounding boxes cross
the tile margin. `n_per_tile=None` returns all valid objects after
filtering.

## Output

The stable public result is stored under `pipeline_data["target_discovery"]`:

```python
{
    "targets": [{
        "target_id": ["R0", 3, 7, 12],
        "tile_id": ["R0", 3, 7],
        "object_label": 12,
        "score": 1800.0,
        "source_feature": "area",
        "centroid_row_px": 930.2,
        "centroid_col_px": 1024.5,
        "bbox_min_row_px": 900,
        "bbox_min_col_px": 990,
        "bbox_max_row_px": 960,
        "bbox_max_col_px": 1050,
        "stage_x_um": 10048.6,
        "stage_y_um": 14982.4,
    }]
}
```

All public fields are JSON-native. Image coordinates are row/col pixels;
stage coordinates are x/y microns.

## Run

Use an overview JSON written by the object-detection contract helpers:

```bash
python workflows/target_discovery/run_pipeline.py overview.json
python workflows/target_discovery/run_pipeline.py overview.json --feature intensity_mean --n-per-tile 3
python workflows/target_discovery/run_pipeline.py overview.json --all
```

## Test

```bash
pytest workflows/target_discovery/tests/
```
