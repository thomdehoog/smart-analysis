# `target_acquisition` workflow

Per-tile target selection for adaptive microscopy. The workflow segments
each acquired overview tile with Cellpose, measures candidate cells, and
returns microscope-ready target coordinates.

```
segment_tile -> pick_targets
```

| Step | Does | Key inputs |
|---|---|---|
| `segment_tile` | Reads a TIFF tile from disk and runs Cellpose instance segmentation. | `image_path`, `channel`, `diameter`, `gpu` |
| `pick_targets` | Measures segmented cells, ranks them, and converts pixel centroids to stage coordinates. | `tile_stage_xy_um`, `source_pixel_size_um`, `image_to_stage`, `n_picks`, `feature` |

## Input payload

Submit one tile at a time. The input payload must include the acquired
image path and the microscope geometry needed to convert image
coordinates into stage coordinates:

```python
{
    "image_path": "path/to/tile.ome.tiff",
    "tile_id": ("R0", 3, 7),
    "tile_stage_xy_um": (10000.0, 15000.0),
    "tile_zwide_um": 2500.0,
    "source_pixel_size_um": (0.65, 0.65),
    "source_image_size_px": (2048, 2048),
    "image_to_stage": [[0.0, -1.0], [1.0, 0.0]],
    "n_picks": 5,
    "feature": "area",
}
```

`n_picks=None` returns all segmented cells in deterministic label order.
Supported ranking features are `area`, `mean_intensity`, and
`eccentricity`.

## Output

The final result is stored under `pipeline_data["pick_targets"]["picks"]`.
Each pick is a plain Python dict that includes:

```python
{
    "pick_id": ("R0", 3, 7, 12),
    "tile_stage_xy_um": (10000.0, 15000.0),
    "tile_zwide_um": 2500.0,
    "source_pixel_size_um": (0.65, 0.65),
    "source_image_size_px": (2048, 2048),
    "centroid_col_row_px": (1024.5, 930.2),
    "bbox_px": (900, 990, 960, 1050),
    "bbox_um": (39.0, 39.0),
    "area_px": 1800,
    "eccentricity": 0.2,
    "mean_intensity": 132.5,
    "cell_source_stage_xy_um": (10048.6, 14982.4),
}
```

## Environment

On the microscope workstation, `segment_tile` declares the LAS X driver
environment `lasxapi_extended` so notebook control and Cellpose overview
analysis run from the same production environment.

The legacy workflow-owned environment setup scripts are kept for
standalone development:

```bash
cd workflows/target_acquisition/environments
python setup_env.py
```

Use `python setup_env.py --gpu cpu` for a CPU-only install, or
`python setup_env.py --dry-run` to inspect the commands first. Remove the
environment with:

```bash
python clean_env.py
```

## Run

Run the shipped CLI from an environment with the engine installed:

```bash
python workflows/target_acquisition/run_pipeline.py
python workflows/target_acquisition/run_pipeline.py --image-path path/to/tile.ome.tiff
python workflows/target_acquisition/run_pipeline.py --image-path path/to/tile.ome.tiff --all
```

Or register the workflow from Python:

```python
import time
from engine import Engine

payload = {
    # See "Input payload" above for the required microscope geometry fields.
    "image_path": "path/to/tile.ome.tiff",
    "tile_id": ("R0", 3, 7),
    "tile_stage_xy_um": (10000.0, 15000.0),
    "tile_zwide_um": 2500.0,
    "source_pixel_size_um": (0.65, 0.65),
    "source_image_size_px": (2048, 2048),
    "image_to_stage": [[0.0, -1.0], [1.0, 0.0]],
    "n_picks": 5,
    "feature": "area",
}

with Engine() as engine:
    engine.register(
        "overview",
        "workflows/target_acquisition/pipelines/overview.yaml",
    )
    engine.submit("overview", payload)

    while not (results := engine.results("overview")):
        status = engine.status("overview")
        if status["failed"]:
            raise RuntimeError(status["failures"][0]["error"])
        time.sleep(0.2)

    picks = results[0]["pick_targets"]["picks"]
```

## Test

Most tests use synthetic masks and do not require Cellpose:

```bash
pytest workflows/target_acquisition/tests/ -m "not cellpose"
```

The optional end-to-end test is marked `cellpose` and `slow`; it runs
only from an environment where Cellpose and Torch import successfully.
