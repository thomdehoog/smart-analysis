# Build Spec: Object Analysis and Target Discovery

This document records the implemented workflow contract for the current
object-analysis and target-discovery path.

The intended path is:

```text
object_analysis -> overview -> target_discovery
```

`target_acquisition` remains the combined compatibility workflow until
its consumers are migrated.

## Workflows

```text
workflows/object_analysis/
workflows/target_discovery/
workflows/target_acquisition/
```

`object_analysis` detects objects in one tile, extracts classical
features, optionally extracts DINOv2 embeddings, and emits one validated
tile record.

`target_discovery` consumes one aggregated overview of tile records. It
can either select targets directly, or cluster embeddings first and then
select targets from the same enriched table.

## Object Analysis

Classical path:

```text
detect_objects -> extract_classical_features -> build_object_table
```

Deep-feature path:

```text
detect_objects -> extract_classical_features -> extract_deep_features -> build_object_table
```

Environment split:

```text
detect_objects / extract_deep_features -> SMART--object_analysis--vision
extract_classical_features             -> SMART--object_analysis--classical
build_object_table                     -> orchestrator env
```

The vision env owns Torch, Cellpose, TIFF IO, and DINOv2. The classical
env owns scikit-image/SciPy feature extraction. Do not mix these stacks
unless a tested package set proves it is safe.

The public tile record is stored under `pipeline_data["object_analysis"]`
and has:

```python
{
    "objects": {
        "properties": {
            "label": [...],
            "object_id": [...],
            "tile_name": [...],
            "centroid_row_px": [...],
            "centroid_col_px": [...],
            "bbox_min_row_px": [...],
            "bbox_min_col_px": [...],
            "bbox_max_row_px": [...],
            "bbox_max_col_px": [...],
            "stage_x_um": [...],
            "stage_y_um": [...],
            "area": [...],
            "intensity_mean": [...],
            "eccentricity": [...]
        },
        "embeddings": {...},  # optional
        "n_objects": n,
    },
    "geometry": {...},
}
```

All columns are row-aligned. Pixel coordinates are row/col. Stage
coordinates are absolute x/y microns derived from the tile geometry
provided in the object-analysis input.

## Target Discovery

Simple selection:

```text
select_targets
```

Clustering plus selection:

```text
cluster_objects -> select_targets
```

Environment split:

```text
select_targets   -> SMART--target_discovery--main
cluster_objects  -> SMART--target_discovery--cluster
```

`cluster_objects` reads row-aligned embedding vectors from the object
table, builds a cosine kNN graph, runs Leiden clustering, computes UMAP
coordinates, and writes clustering artifacts.

The clustering table is the source of truth. It must include:

```text
object_id
tile_id
object_label
stage_x_um
stage_y_um
centroid_row_px
centroid_col_px
area
intensity_mean
eccentricity
cluster_id
umap_x
umap_y
```

When `output_dir` is provided, clustering writes:

```text
features/clusters.csv
features/clusters.json
features/clusters_umap.svg
```

The scatterplot is a visual artifact of the table, not a replacement for
the table.

## Shared Modules

Shared workflow code lives under `workflows/`:

```text
_contracts.py      JSON-native handoff validation and overview IO
_features.py       classical feature extraction implementation
_geometry.py       image row/col to stage x/y conversion
_segmentation.py   TIFF loading and Cellpose segmentation
_object_ids.py     stable tile/object naming
_object_crops.py   crop extraction for deep features
```

Step files remain real engine steps with literal top-level `METADATA`.
Do not import `METADATA` from shared modules; the engine reads it with
AST literal parsing.

## Verification

Run the public fast path:

```bash
pytest -m "not cellpose and not deep and not cluster and not conda_env"
```

Run the object/target focused suites:

```bash
pytest workflows/object_analysis/tests/ -m "not cellpose and not deep"
pytest workflows/target_discovery/tests/ -m "not cluster"
pytest workflows/target_discovery/tests/ -m cluster
```

Run real Cellpose and DINO checks from capable environments before
publishing:

```bash
pytest workflows/object_analysis/tests/ -m cellpose
pytest workflows/object_analysis/tests/ -m deep
```
