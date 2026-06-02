# `object_detection` workflow

Detect and measure every object in one acquired image tile.

```
detect_objects -> extract_object_features -> package_objects
```

This workflow is the detection phase. It reads one TIFF tile, segments it
with Cellpose, extracts the shared object feature table, and packages the
result into the public tile-detection contract. It does not select,
rank, or discard objects.

## Input Payload

Submit one tile at a time:

```python
{
    "image_path": "path/to/tile.ome.tiff",
    "tile_id": ["R0", 3, 7],
    "tile_stage_xy_um": [10000.0, 15000.0],
    "tile_zwide_um": 2500.0,
    "source_pixel_size_um": [0.65, 0.65],
    "source_image_size_px": [2048, 2048],
    "image_to_stage": [[0.0, -1.0], [1.0, 0.0]],
    "channel": 0,
    "diameter": None,
    "gpu": False,
}
```

`source_pixel_size_um` is `[pixel_width_um, pixel_height_um]`.
`source_image_size_px` is `[nx, ny]`.

## Output

The stable public result is stored under
`pipeline_data["object_detection"]`:

```python
{
    "objects": {
        "properties": {
            "label": [1, 2],
            "centroid_row_px": [120.5, 340.0],
            "centroid_col_px": [220.0, 410.5],
            "bbox_min_row_px": [100, 320],
            "bbox_min_col_px": [200, 390],
            "bbox_max_row_px": [145, 365],
            "bbox_max_col_px": [245, 435],
            "area": [900.0, 1200.0],
            "intensity_mean": [132.5, 98.2],
            "eccentricity": [0.2, 0.6]
        },
        "n_objects": 2
    },
    "geometry": {
        "tile_id": ["R0", 3, 7],
        "tile_stage_xy_um": [10000.0, 15000.0],
        "tile_zwide_um": 2500.0,
        "source_pixel_size_um": [0.65, 0.65],
        "source_image_size_px": [2048, 2048],
        "image_to_stage": [[0.0, -1.0], [1.0, 0.0]]
    }
}
```

Additional feature columns are allowed as row-aligned pass-through data
inside `objects["properties"]`. Public coordinate columns are unscaled
pixel coordinates in row/col order.

## Environment

`detect_objects` declares the workflow-owned conda environment
`SMART--object_detection--main`. Build it with:

```bash
cd workflows/object_detection/environments
python setup_env.py
```

Remove it with:

```bash
python clean_env.py
```

## Run

```bash
python workflows/object_detection/run_pipeline.py path/to/tile.ome.tiff
python workflows/object_detection/run_pipeline.py path/to/tile.ome.tiff --save-overview overview.json
```

The saved overview JSON can be passed directly to
`workflows/target_discovery/run_pipeline.py`.

## Test

Most tests use synthetic tables and a stub Cellpose model:

```bash
pytest workflows/object_detection/tests/ -m "not cellpose"
```
