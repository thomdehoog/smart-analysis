# Build Spec: Object Detection and Target Discovery Workflows

This document defines the workflow split for Smart Analysis. It records
the implementation contract for the object-detection and
target-discovery path.

The goal is a simple, internally consistent foundation:

1. Detect and measure objects.
2. Aggregate detected-object tiles into an overview.
3. Discover revisit targets from that overview.

The design is additive. The existing combined `target_acquisition`
workflow stays in place while the new workflows are built and verified.

## Design Rules

- Keep one obvious path: detect objects, aggregate tiles, discover
  targets.
- Public workflow output must not depend on internal step names.
- Contracts are plain JSON-native dictionaries: dicts, lists, strings,
  ints, floats, booleans, and nulls.
- Public contracts must contain no tuples, numpy arrays, or numpy scalar
  values.
- Shared workflow code lives in underscore-prefixed modules under
  `workflows/`.
- Engine-facing step files remain real step files with literal top-level
  `METADATA`.
- Tests live inside each workflow's `tests/` directory.
- Do not remove or rename `target_acquisition` in this build.

## Workflow Shape

The repository should contain three target-related workflows:

```text
workflows/object_detection/
workflows/target_discovery/
workflows/target_acquisition/
```

`object_detection` is the canonical detection phase. It detects and
measures all objects in acquired image tiles.

`target_discovery` is the canonical selection phase. It consumes object
tables and tile geometry, then emits revisit targets.

`target_acquisition` remains the existing combined workflow. Public docs
may describe it as the combined target-acquisition workflow. Do not call
it legacy until the replacement path is implemented and consumers have
migrated.

## Shared Modules

Add these modules at the `workflows/` root:

```text
workflows/_contracts.py
workflows/_features.py
workflows/_geometry.py
workflows/_segmentation.py
```

### `_contracts.py`

This module owns the inter-workflow contract. Keep it small. It is not a
schema framework.

Required functions:

```python
validate_tile_detection(tile)
validate_overview(overview)
validate_targets(result)
to_builtin(obj)
save_overview(path, overview)
load_overview(path)
```

Responsibilities:

- Check required fields.
- Check row-aligned object property columns.
- Convert numpy arrays and numpy scalar values to native Python values.
- Enforce JSON round-trippability.
- Save and load overviews as JSON.

### `_features.py`

Move the comprehensive feature extraction implementation from
`workflows/cell_analysis/steps/extract_features.py` into this module as a
mechanical relocation.

Do not refactor behavior during the move.

After the move, `cell_analysis/steps/extract_features.py` should be a
thin wrapper over `_features.py`.

`object_detection/steps/extract_object_features.py` should also be a thin
wrapper over `_features.py`.

### `_segmentation.py`

Extract the reusable TIFF loading, 2D channel handling, and warm Cellpose
model logic needed by new object detection.

`object_detection/steps/detect_objects.py` should call this shared
module.

`target_acquisition/steps/segment_tile.py` is compatibility code and
stays unchanged in the first implementation pass, even if that leaves a
temporary copy of related logic. This is deliberate: the compatibility
workflow is frozen while the new path is built. Remove that duplication
only after the combined workflow is retired or migrated.

### `_geometry.py`

This module owns shared image-to-stage coordinate conversion.

Required function:

```python
image_point_to_stage_xy(
    *,
    centroid_row_px,
    centroid_col_px,
    image_size_px,
    pixel_size_um,
    tile_stage_xy_um,
    image_to_stage,
)
```

It must use the same math already tested in
`target_acquisition/steps/pick_targets.py`:

1. Image coordinates are row/col.
2. Stage-offset coordinates are x/y, so use col for x and row for y.
3. Offsets are measured from the image center.
4. Pixel offsets are scaled by pixel size.
5. The 2x2 `image_to_stage` matrix maps image x/y offsets to stage x/y
   offsets.
6. The stage offset is added to `tile_stage_xy_um`.

`target_discovery` must call this helper rather than reimplementing the
affine conversion locally.

## Step Metadata Rule

The engine reads step `METADATA` by AST literal parsing. It only
recognizes a top-level literal assignment such as:

```python
METADATA = {
    "environment": "SMART--object_detection--main",
    "max_workers": 1,
}
```

Therefore, wrappers must not import `METADATA` from shared modules.

Correct wrapper pattern:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _features import run  # noqa: E402

METADATA = {
    "description": "Extract object features",
    "version": "1.0",
}
```

Each engine-facing step owns its own literal `METADATA`, even when it
shares the same implementation function.

Use the same import pattern for `_contracts.py`, `_features.py`,
`_geometry.py`, and `_segmentation.py`. The workflow directories are
loaded by path and should not rely on `workflows` being an importable
Python package.

## Coordinate Convention

Object tables use image coordinates:

```text
row, col = y, x
```

Stage coordinates use microscope coordinates:

```text
x, y in microns
```

The row/col to x/y conversion happens only in target discovery.

Avoid tuple-valued public fields. Use explicit scalar columns and fields.

## Public Contracts

### Tile Detection

One `object_detection` run emits one tile detection:

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

Required object columns:

- `label`
- `centroid_row_px`
- `centroid_col_px`
- `bbox_min_row_px`
- `bbox_min_col_px`
- `bbox_max_row_px`
- `bbox_max_col_px`
- `area`
- `intensity_mean`
- `eccentricity`

All other feature columns are allowed as row-aligned pass-through data.
The contract should not enumerate every possible feature column.

`n_objects` must equal the length of every property column. Empty
detections are valid and must use empty lists with `n_objects == 0`.

### Overview

The orchestrator aggregates tile detections into an overview:

```python
{
    "tiles": [
        tile_detection,
        tile_detection
    ]
}
```

Aggregation is not a workflow step. `object_detection` emits one tile.
The caller assembles the list of tiles. `target_discovery` consumes the
assembled overview.

### Targets

`target_discovery` emits:

```python
{
    "targets": [
        {
            "target_id": ["R0", 3, 7, 1],
            "tile_id": ["R0", 3, 7],
            "object_label": 1,
            "score": 900.0,
            "source_feature": "area",
            "centroid_row_px": 120.5,
            "centroid_col_px": 220.0,
            "bbox_min_row_px": 100,
            "bbox_min_col_px": 200,
            "bbox_max_row_px": 145,
            "bbox_max_col_px": 245,
            "stage_x_um": 9994.8,
            "stage_y_um": 14976.7
        }
    ]
}
```

Target fields are scalar and JSON-native. Do not use tuple fields such as
`centroid_col_row_px`, `bbox_px`, or `cell_source_stage_xy_um` in the new
public contract.

## Build Order

### Phase 1: Build Spec

Add this document and keep it generic. It should describe Smart Analysis
workflow structure only.

### Phase 2: Contracts

Implement `workflows/_contracts.py`.

Add tests in:

```text
workflows/target_discovery/tests/test_contracts.py
```

Test:

- required detection fields
- required overview fields
- required target fields
- row-aligned object properties
- sparse labels
- empty detections
- native Python type conversion
- JSON save/load round-trip
- row/col coordinate convention

The synthetic fixtures in these tests must pass the same validators that
real workflow outputs will use.

### Phase 3: Feature Extraction Sharing

Move the feature implementation to `workflows/_features.py`.

Update:

```text
workflows/cell_analysis/steps/extract_features.py
```

to be a wrapper with literal `METADATA`.

Run:

```bash
pytest workflows/cell_analysis/tests/ -m "not cellpose"
```

Do not continue if this changes behavior.

### Phase 4: Target Discovery

Add:

```text
workflows/target_discovery/
  README.md
  pipelines/target_discovery.yaml
  steps/discover_targets.py
  tests/test_target_discovery.py
  tests/test_handoff.py
  run_pipeline.py
```

`tests/test_contracts.py` is introduced in Phase 2 and remains in this
workflow's `tests/` directory.

V1 behavior:

- Consume `pipeline_data["input"]["tiles"]`.
- Validate the overview before selection.
- Filter by explicit thresholds when provided.
- Rank by one feature.
- Support `direction="high"` and `direction="low"`.
- Support `n_per_tile`.
- Support `border_margin_px` by excluding objects whose bbox crosses the
  tile margin.
- Convert image centroids to stage coordinates per tile.
- Emit JSON-native `targets`.

V1 must not:

- read masks
- run `regionprops`
- segment images
- compute deep features
- apply undocumented auto-thresholds

Auto-thresholds are required later for migration parity, but not in V1.
When added, the rule must be explicit and test-pinned.

Target discovery tests:

- rank by feature from the table
- high and low direction
- explicit threshold filtering
- `n_per_tile` per tile
- `border_margin_px` using each tile's image size
- stage conversion per tile
- no numpy scalar values in output
- `n_per_tile=None` returns all valid targets
- missing required columns fail clearly

### Phase 5: Object Detection

Add:

```text
workflows/object_detection/
  README.md
  pipelines/object_detection.yaml
  steps/detect_objects.py
  steps/extract_object_features.py
  steps/package_objects.py
  tests/test_object_detection.py
  run_pipeline.py
```

Pipeline:

```text
detect_objects -> extract_object_features -> package_objects
```

`detect_objects`:

- read TIFF input
- handle 2D, channel-first, and channel-last data
- use warm Cellpose state
- output masks, image, and image size for downstream internal steps

`extract_object_features`:

- wrapper over `_features.py`
- literal `METADATA`

`package_objects`:

- convert internal detection and feature outputs to the public tile
  detection contract
- normalize feature names to the public scalar column names
- attach required geometry from input
- call `validate_tile_detection`

The required feature-name mapping is:

| `_features.py` / `regionprops_table` column | Public object column |
|---|---|
| `label` | `label` |
| `centroid-0` | `centroid_row_px` |
| `centroid-1` | `centroid_col_px` |
| `bbox-0` | `bbox_min_row_px` |
| `bbox-1` | `bbox_min_col_px` |
| `bbox-2` | `bbox_max_row_px` |
| `bbox-3` | `bbox_max_col_px` |
| `area` | `area` |
| `intensity_mean` | `intensity_mean` |
| `eccentricity` | `eccentricity` |

Use the geometric centroid, not `weighted_centroid`, for
`centroid_row_px` and `centroid_col_px`.

Public pixel coordinate columns are always unscaled pixel coordinates.
Do not pass `spacing=` when computing the required public coordinate and
bbox fields. Physical-unit feature columns may be added later under
separate explicit names.

Object detection tests:

- valid detection output
- empty detection output
- sparse label row alignment
- geometry validation
- scalar public coordinate fields
- no numpy scalar values
- optional Cellpose-marked real run

### Phase 6: Handoff

Add a handoff test in:

```text
workflows/target_discovery/tests/test_handoff.py
```

The test should:

1. Build two or more synthetic validated tile detections.
2. Assemble `{"tiles": [...]}`.
3. Run `discover_targets`.
4. Validate the target output.
5. Assert per-tile quota and per-tile stage conversion.

Later, once `object_detection` is implemented, add a non-Cellpose
handoff test that runs object packaging outputs through discovery.

## Verification

Before committing implementation work, run:

```bash
pytest workflows/target_discovery/tests/
pytest workflows/object_detection/tests/ -m "not cellpose"
pytest workflows/cell_analysis/tests/ -m "not cellpose"
pytest workflows/target_acquisition/tests/ -m "not cellpose"
pytest -m "not cellpose and not slow"
```

`target_acquisition` tests must stay green throughout.

## Documentation

Add public, generic READMEs for the new workflows.

Root README wording:

- `object_detection`: detects and measures objects in acquired image
  tiles.
- `target_discovery`: selects revisit targets from object tables.
- `target_acquisition`: combined target-acquisition workflow.

Keep workflow documentation generic to Smart Analysis. Do not include
private deployment details or repo-specific integration context.

## Later Work

After the two-phase path works:

- add deep feature extraction as optional object columns or an optional
  `embeddings` block
- add clustering as a target discovery strategy
- add explicit, tested auto-threshold rules
- migrate consumers to `object_detection` plus `target_discovery`
- then decide whether to deprecate or remove the combined workflow

Do not mix these later items into the first implementation pass.
