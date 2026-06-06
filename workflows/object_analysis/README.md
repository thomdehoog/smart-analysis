# `object_analysis` workflow

Object-centered analysis for one acquired image tile.

Classical path:

```text
detect_objects -> extract_classical_features -> build_object_table
```

Deep-feature path:

```text
detect_objects -> extract_classical_features -> extract_deep_features -> build_object_table
```

`extract_deep_features` is optional. Include the deep YAML when DINOv2
embeddings and object crops are needed; otherwise use the classical YAML.

Checkpointed path for iterative rare-event selection:

```text
object_detection.yaml:      detect_objects
object_features_deep.yaml:  load_detected_objects -> extract_classical_features -> extract_deep_features -> build_object_table
```

Use the checkpointed path when segmentation should be run once and feature
extraction/clustering should be re-run with different crop or selection
settings. The detection step writes masks plus a checkpoint JSON; the feature
pipeline reloads that checkpoint and rejects stale masks when the recorded
segmentation-parameter hash does not match the requested run.
The hash covers true mask-generation parameters (`channels`, Cellpose
thresholds, `niter`, `diameter`, and `max_segmentation_size_px`). It
deliberately excludes GPU/CPU placement and area filters. Area filters are
post-segmentation filters and can be retuned from the raw masks without
running Cellpose again.

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
    "channels": None,
    "gpu": False,
}
```

`channels=None` keeps up to three channels for 2D+channels input. Pass an
explicit list such as `[0, 2]` to choose channels from a larger stack. Plain
2D images accept `None` or `[0]`; any other channel is rejected.

Object-size filtering is configured in the workflow YAML:

```yaml
- detect_objects:
    channels: null
    cellprob_threshold: 0.0
    flow_threshold: 0.4
    niter: null
    diameter: null
    max_segmentation_size_px: 512
    min_area_px: 1000
    max_area_px: null
```

`min_area_px` and `max_area_px` are area thresholds on the detected mask
labels. Use `null` to disable either bound.
`max_segmentation_size_px` downscales large tiles before Cellpose and then
rescales masks back to the original image size; it never upsamples small
tiles. `cellprob_threshold` controls how much marginal cell-probability is
included (lower values usually produce larger/more masks), `flow_threshold`
controls Cellpose's flow-consistency QC, and `niter` can be increased for
very long objects.

Optional artifact persistence:

```python
{
    "output_dir": "analysis/object_analysis/run_001"
}
```

When `output_dir` is provided, `detect_objects` writes filtered `masks.tif`,
unfiltered `raw_masks.tif`, and `detection_checkpoint.json` under
`tiles/<tile_name>/`. The checkpoint records content hashes for the source
image and raw masks so feature extraction cannot silently pair stale masks
with changed image data. On the deep path, `extract_deep_features` also writes
per-object crop artifacts under `objects/`. Classical-only runs do not cut
crops by default.

To re-enter feature extraction from a saved detection checkpoint, submit:

```python
{
    "detection_checkpoint_path": "analysis/object_detection/tiles/R0_r003_c007/detection_checkpoint.json",
    "segmentation_params_hash": "<hash from detection>",
    "min_area_px": 1000,
    "max_area_px": null,
    "output_dir": "analysis/object_features/run_001",
}
```

Deep crops are configured in `pipelines/object_analysis_deep.yaml`:

```yaml
- extract_deep_features:
    crop_size_px: 128
    mask: true
    drop_incomplete_crops: true
```

`crop_size_px` is the fixed square extraction size used for every object
in the run. `mask: true` zeros non-object pixels so the deep crop contains
the segmented object only, while still writing the object mask separately.
Set `mask: false` only when local context should be embedded.
`drop_incomplete_crops: true` excludes objects whose mask touches the tile
boundary or whose bbox does not fit inside the fixed crop. For masked crops,
the crop window may extend outside the tile and is zero-padded; for unmasked
context crops, the full crop window must stay inside the tile. The DINO path
does not perform adaptive intensity normalization; integer crops are converted
by dtype range, and float crops must already be in `[0, 1]` from upstream
preprocessing.

## Output

The stable public result is stored under
`pipeline_data["object_analysis"]`:

```python
{
    "objects": {
        "properties": {
            "label": [1, 2],
            "object_id": ["R0_r003_c007_obj00001", "R0_r003_c007_obj00002"],
            "centroid_row_px": [120.5, 340.0],
            "centroid_col_px": [220.0, 410.5],
            "bbox_min_row_px": [100, 320],
            "bbox_min_col_px": [200, 390],
            "bbox_max_row_px": [145, 365],
            "bbox_max_col_px": [245, 435],
            "stage_x_um": [9990.0, 10012.0],
            "stage_y_um": [15020.0, 14995.0],
            "area": [900.0, 1200.0],
            "intensity_mean": [132.5, 98.2],
            "intensity_mean_c0": [132.5, 98.2],
            "intensity_mean_c1": [80.0, 72.0],
            "intensity_mean_c2": [44.0, 55.0],
            "eccentricity": [0.2, 0.6]
        },
        "embeddings": {
            "label": [1, 2],
            "vectors": [[...], [...]],
            "model": "dinov2_vitb14",
            "backend": "dinov2"
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

The `embeddings` block appears only in the deep-feature pipeline.
Crop-path columns also appear only in the deep-feature pipeline.
Additional classical feature columns are allowed as row-aligned
pass-through data inside `objects["properties"]`.

## Environment

Build the workflow environments with:

```bash
cd workflows/object_analysis/environments
python setup_env.py
python setup_env.py --step classical
```

The default `vision` environment is for Cellpose detection and optional
DINOv2 embedding extraction. The `classical` environment is for
scikit-image feature extraction. This keeps the Torch/Cellpose stack
separate from the scikit-image/SciPy stack while installing PyTorch only
once.

Remove them with:

```bash
python clean_env.py
```

## Run

Classical features only:

```bash
python workflows/object_analysis/run_pipeline.py path/to/tile.ome.tiff
```

With DINOv2 embeddings:

```bash
python workflows/object_analysis/run_pipeline.py path/to/tile.ome.tiff --deep
```

Save a one-tile overview JSON:

```bash
python workflows/object_analysis/run_pipeline.py path/to/tile.ome.tiff --save-overview overview.json
```

## Test

Fast tests:

```bash
pytest workflows/object_analysis/tests/ -m "not cellpose and not deep"
```

Real Cellpose end-to-end test:

```bash
pytest workflows/object_analysis/tests/ -m cellpose
```
