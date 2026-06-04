# Object Workflow Plan

This document records the clean target design for object-centered
analysis workflows in Smart Analysis. It separates the current stable v1
work from later scale and deep-feature expansion.

## Goal

Keep the workflow intuitive:

```text
detect objects
extract classical features
optionally extract deep features
build the object table
discover targets
```

The user-facing result is one object-centered record. Internal step
names, scikit-image column names, numpy values, masks, crops, and model
details should not leak into the public contract.

## Implemented Workflow

The implemented object-centered workflow is `workflows/object_analysis/`.
The classical path is:

```text
detect_objects
extract_classical_features
build_object_table
```

The deep-feature path adds one optional step:

```text
detect_objects
extract_classical_features
extract_deep_features
build_object_table
```

Behavior:

- `detect_objects` runs Cellpose and produces masks, the analysis image,
  object count, and image size.
- `extract_classical_features` computes scikit-image features.
- `extract_deep_features` optionally creates crops and computes DINOv2
  embeddings.
- `build_object_table` converts internal outputs into the public
  `object_analysis` contract.

Verification:

```bash
pytest workflows/object_analysis/tests/ -m "not cellpose and not deep"
pytest workflows/target_discovery/tests/
pytest -m "not cellpose and not slow"
```

Run the Cellpose and deep markers from capable envs before publishing:

```bash
pytest -m cellpose
pytest workflows/object_analysis/tests/ -m deep
```

## Step Responsibilities

### `detect_objects`

Tile-level image analysis.

Answers:

```text
Where are the objects in this tile?
```

Responsibilities:

- read the tile image
- select the segmentation channel or input plane
- run Cellpose
- produce masks and object localization inputs
- keep the Cellpose model warm in step state

Environment:

```text
Cellpose env
```

### `extract_classical_features`

Object-level classical feature extraction.

Answers:

```text
What are the classical properties of each object?
```

Responsibilities:

- compute scikit-image shape, intensity, morphology, texture, and
  neighbourhood features
- remain independent of Torch and DINO
- output row-aligned feature columns keyed by object label

Environment:

```text
scikit-image env
```

### `extract_deep_features`

Optional object-level deep feature extraction.

Answers:

```text
What learned appearance representation describes each selected object?
```

Responsibilities:

- create object crops internally
- define crop padding, masking/background policy, resize, and
  channel-to-RGB mapping
- load and reuse the DINO model in step state
- output row-aligned embeddings
- write optional tile/object crop artifacts when `output_dir` is provided

Environment:

```text
Torch / DINO env
```

This step is optional. Optionality is expressed by step presence in the
YAML, not by a `deep=true/false` flag inside the classical feature step.

Initial implementation target:

- use DINOv2 first; keep the metadata model-agnostic so DINOv3 can replace
  it later
- start with ViT-B/14 as the practical default
- use 3-channel crops mapped to RGB by an explicit channel map
- percentile-normalize each crop before ImageNet normalization
- run batched inference
- store a pooled, L2-normalized global vector per object
- keep patch-token grids optional; they are useful for later masked
  pooling and explainability but are too large to make mandatory

Crop policy for the first deep-feature pass:

- square crop centered on object centroid
- fixed `crop_size_px`, declared once in the workflow YAML
- keep crop size out of per-tile input so it cannot drift during a run
- zero-pad at image boundaries
- drop objects whose full fixed crop would require padding
- support `mask: false` / `mask: true`:
  - `false`: keep context pixels around the object
  - `true`: zero non-object pixels while still writing the object mask
- do not perform adaptive intensity normalization in the DINO step;
  integer crops are converted by dtype range, and any real intensity
  normalization belongs in upstream preprocessing
- map fewer-than-three channels deterministically:
  - 1 channel: `[c0, c0, c0]`
  - 2 channels: `[c0, c1, (c0 + c1) / 2]`
  - 3 channels: `[c0, c1, c2]`
- do not add rotation/scale-invariance augmentation in the first pass;
  record those as later options if embeddings need them

### `build_object_table`

Contract boundary.

Answers:

```text
What is the stable object table that downstream workflows consume?
```

Responsibilities:

- rename internal feature columns to public names
- create stable position-based `tile_name` and `object_id` columns
- attach geometry and stage coordinates
- convert numpy values to JSON-native values
- merge optional embeddings
- merge optional crop-path columns from the deep-feature path
- validate row alignment
- emit the stable `object_analysis` record

Environment:

```text
orchestrator / light env
```

## Deep-Feature Pipeline

Classical-only pipeline:

```text
detect_objects
extract_classical_features
build_object_table
```

Classical plus deep features:

```text
detect_objects
extract_classical_features
extract_deep_features
build_object_table
```

The final public output remains one object record:

```python
{
    "objects": {
        "properties": {...},
        "embeddings": {...},  # optional
        "n_objects": n,
    },
    "geometry": {...},
}
```

If `objects["embeddings"]` is present, it must align to the same object
order as `objects["properties"]["label"]`.

## Two-Level Analysis

Keep the model explicit:

```text
tile analysis   = detection and localization
object analysis = classical/deep feature extraction
```

Tile analysis is position-based. It knows:

- tile name
- tile geometry
- masks
- object labels
- bboxes
- centroids
- stage coordinates

Object analysis is object-based. It knows:

- object identity
- object crop or mask artifact, when deep features or QC export requested
- classical features
- deep embeddings, when requested

## Scalable Persistence Direction

Do not make this part of the rename pass. This is the later scale layer.

At scale, use a run folder with clean artifacts:

```text
object_analysis/<run>/
  tiles/
  objects/
  features/
  overview.json
```

Meaning:

```text
tiles/    tile-level image analysis and localization
objects/  extracted object artifacts, usually a subset
features/ aggregated feature tables for the full population
```

Rules:

- small data can stay inline
- large binaries are files plus paths
- masks, images, crops, and embeddings should not be held in final
  results as arrays
- the population feature table is the source of truth for all objects
- per-object folders are for selected, embedded, or QC objects, not
  necessarily every detection
- the orchestrator should drain results and persist per tile so memory is
  bounded by concurrency, not total tile count

Suggested layout:

```text
object_analysis/<run>/
  tiles/
    R0_r003_c007/
      masks.tif
      tile.json
  objects/
    R0_r003_c007_obj00017/
      crop.tif
      mask.tif
      object.json
  features/
    classical_features.parquet
    deep_features.npy
    deep_features.json
    object_index.json
  overview.json
```

Use position-based names:

```python
tile_name = f"{region}_r{row:03d}_c{col:03d}"
object_name = f"{tile_name}_obj{label:05d}"
```

Folder names are for humans and paths. The table and JSON metadata are
the source of truth.

## Target Discovery

`target_discovery` should consume the clean object table or overview
record, not raw internal step outputs.

It should rank and filter from public columns and optional embeddings:

- classical feature columns
- deep embedding vectors
- clustering labels
- target scores
- stage coordinates

It should not read masks, rerun segmentation, or recompute classical
region properties.

The clustering path is:

```text
cluster_objects
select_targets
```

`cluster_objects` clusters object embeddings with a cosine kNN graph,
Leiden clustering, and UMAP coordinates. It writes `cluster_id`,
`umap_x`, and `umap_y` back as row-aligned object columns and exports a
table plus a scatterplot. The table keeps `object_id`, tile/object
identity, absolute `stage_x_um` / `stage_y_um`, clustering labels, and
UMAP coordinates; the scatterplot is only a visual view of that table.

## Later Work

After the implemented workflow is stable:

1. Design the scalable `tiles/`, `objects/`, `features/` artifact
   helpers.
2. Add explicit, tested auto-threshold rules.
3. Migrate consumers to `object_analysis` plus `target_discovery`.
4. Decide whether to deprecate the combined `target_acquisition`
    workflow.

Do not mix these phases. Keep each change small enough that its tests
explain what changed.
