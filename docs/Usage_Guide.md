# Smart Analysis Engine v4 -- Usage Guide

## Quick Start

```python
from engine import Engine

engine = Engine()
engine.register("analysis", "path/to/pipeline.yaml")
engine.submit("analysis", {"image": my_data})

# Poll for results
results = engine.results("analysis")
engine.shutdown()
```

---

## Writing a Step

A step is a Python file with a `run()` function and an optional `METADATA` dict.

### Minimal step

```python
def run(pipeline_data, state, **params):
    image = pipeline_data["input"]["image"]
    pipeline_data["result"] = process(image)
    return pipeline_data
```

### Step with METADATA

```python
METADATA = {
    "environment": "SMART--segment",   # conda env (omit for default)
    "max_workers": 1,                  # max parallel instances (default: 1)
}

def run(pipeline_data, state, **params):
    ...
    return pipeline_data
```

### The `run()` function

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline_data` | dict | Input data. Contains `"input"` (your data) and `"metadata"` (engine info). For scoped steps, contains `"results"` (list of accumulated outputs). |
| `state` | dict | Persistent per-step per-worker dict. Empty on first call. Use it to cache expensive objects (models, lookup tables). |
| `**params` | kwargs | Parameters from the YAML configuration. |

The function must return a dict.

### The `state` dict (warm models)

Use `state` to avoid re-loading expensive resources on every call:

```python
from cellpose.models import CellposeModel

METADATA = {"environment": "SMART--segment", "max_workers": 1}

def run(pipeline_data, state, **params):
    # First call: load model (cold start)
    # Subsequent calls: reuse cached model (warm)
    if "model" not in state:
        state["model"] = CellposeModel(gpu=True)

    masks, flows, styles = state["model"].eval(pipeline_data["image"])
    pipeline_data["masks"] = masks
    return pipeline_data
```

The state dict is:
- Per-step: different steps get separate state dicts
- Per-worker: each worker subprocess has its own state
- Garbage collected when the worker shuts down (frees GPU memory)

### Imports

Use normal top-level imports. No special conventions:

```python
import numpy as np
from cellpose.models import CellposeModel

def run(pipeline_data, state, **params):
    ...
```

Top-level imports are safe because the engine never executes step files.
It only reads METADATA via AST parsing. Workers run in the correct conda
environment, so imports resolve against the right packages.

### METADATA fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment` | str | orchestrator's env | Conda environment name. Steps with different environments run in separate worker subprocesses. |
| `max_workers` | int | 1 | Maximum parallel instances of this step. Set higher for CPU-bound steps that benefit from parallelism. Set to 1 for GPU steps (VRAM constraint). |

---

## Writing a Pipeline YAML

The YAML defines step order, parameters, and optional scopes.

### Simple pipeline (no scopes)

```yaml
metadata:
  functions_dir: "../steps"

my-analysis:
  - preprocess:
      sigma: 1.0
      clip_limit: 0.03

  - segment:
      diameter: null

  - extract_features:
      select_by: "area"
      percentile: 99

  - feedback:
      output_dir: "./output"
```

### Pipeline with scopes

Scopes control when a step runs. A scoped step waits until the scope
level is signaled complete, then receives accumulated results from all
jobs in that scope group.

```yaml
metadata:
  functions_dir: "../steps"

overview:
  - preprocess:
      sigma: 1.0

  - segment:

  - stitch:
      scope: group

  - normalize:
      scope: all
```

- Steps before a scope run immediately per job (Phase 0)
- `stitch` waits for `complete="group"` and receives all segment outputs
- `normalize` waits for `complete="all"` and receives all stitch outputs

### YAML metadata fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `functions_dir` | str | `"../steps"` | Directory containing step .py files, relative to the YAML file. |
| `verbose` | int | 2 | Logging level. 0=silent, 1=errors, 2=steps, 3=debug. |

---

## Using the Engine

### register(name, yaml_path)

Register a pipeline. Call once per pipeline.

```python
engine = Engine()
engine.register("overview", "pipelines/overview.yaml")
engine.register("target", "pipelines/target.yaml")
```

### submit(name, data, scope, priority, complete)

Submit a job. Non-blocking.

```python
# Simple (no scopes)
engine.submit("analysis", {"image": img})

# With scope labels
engine.submit("overview", tile_data, scope={"group": "R3"})

# Signal scope completion on the last job
engine.submit("overview", last_tile, scope={"group": "R3"}, complete="group")

# Signal multiple scope levels at once
engine.submit("overview", last_tile, scope={"group": "R5"},
              complete=["group", "all"])

# With priority (higher = more urgent)
engine.submit("overview", data, priority=10)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Registered pipeline name. |
| `data` | dict | Input data (available as `pipeline_data["input"]`). |
| `scope` | dict | Scope group labels. E.g., `{"group": "R3"}`. |
| `priority` | int | Higher = more urgent. Default is FIFO. |
| `complete` | str or list | Signals scope completion. |

### status(name)

Query pipeline state.

```python
# One pipeline
s = engine.status("overview")
# s = {"pending": 3, "running": 2, "completed": 14, "failed": 1, "failures": [...]}

# All pipelines
s = engine.status()
# s = {"overview": {...}, "target": {...}}
```

### results(name)

Retrieve completed results. Consumed on retrieval (calling again returns
only new results).

```python
results = engine.results("overview")
for r in results:
    phase = r["_phase"]        # 0 for immediate, 1+ for scoped
    scope = r["_scope"]        # scope labels from submit
    scope_level = r["_scope_level"]  # which scope triggered this result
```

### shutdown()

Clean up workers and threads.

```python
engine.shutdown()

# Or use as context manager
with Engine() as engine:
    engine.register("test", "pipeline.yaml")
    engine.submit("test", data)
    ...
# Automatic cleanup
```

---

## Scopes

Scopes control when aggregation steps run. They are defined in the YAML
and signaled via the `complete` parameter on `submit()`.

### How it works

1. Jobs are submitted with scope labels: `scope={"group": "R3"}`
2. Immediate steps (no scope in YAML) run per job
3. When `complete="group"` is signaled, the engine collects all results
   where `scope["group"]` matches and passes them to the scoped step
4. The scoped step receives `pipeline_data["results"]` (a list)

### Scope matching

When `complete="X"` is signaled:
- If `"X"` is a key in jobs' scope dicts: match by value
- If `"X"` is not a key (like `"all"`): collect everything

This means `"all"` is not a keyword. It works because no job has `"all"`
as a scope key, so the engine collects everything.

### Writing a scoped step

```python
def run(pipeline_data, state, **params):
    results = pipeline_data["results"]       # list of prior step outputs
    failures = pipeline_data.get("failures", [])  # any failed jobs

    # Process accumulated results
    all_masks = [r["masks"] for r in results]
    stitched = stitch(all_masks)

    pipeline_data["stitched"] = stitched
    return pipeline_data
```

### Chained scopes

Multiple scope levels can be chained. Signal both on the last job:

```python
engine.submit("analysis", data, scope={"group": "R5"},
              complete=["group", "all"])
```

The engine processes them in order: "group" completes first, then "all"
collects the group results.

---

## Common Patterns

### Post-acquisition batch analysis

```python
with Engine() as engine:
    engine.register("batch", "analysis.yaml")

    for image_path in image_paths:
        engine.submit("batch", {"path": str(image_path)})

    # Wait for all results
    import time
    while True:
        status = engine.status("batch")
        if status["pending"] == 0 and status["running"] == 0:
            break
        time.sleep(1)

    results = engine.results("batch")
```

### Adaptive feedback microscopy

```python
with Engine() as engine:
    engine.register("overview", "overview.yaml")
    engine.register("target", "target.yaml")

    # Submit overview tiles
    for i, tile in enumerate(tiles):
        is_last = (i == len(tiles) - 1)
        engine.submit("overview", tile,
                      scope={"group": region_id},
                      priority=10,
                      complete="group" if is_last else None)

    # Poll for feedback
    while True:
        results = engine.results("overview")
        feedback = [r for r in results if r["_phase"] == 1]
        if feedback:
            break
        time.sleep(0.5)

    # Act on feedback
    for position in feedback[0]["interesting_positions"]:
        engine.submit("target", {"position": position})
```

### Multiple scope levels (tile -> region -> experiment)

```yaml
experiment:
  - preprocess:
  - segment:
  - stitch:
      scope: region
  - normalize:
      scope: experiment
```

```python
# Submit tiles for region R1
for tile in region_r1_tiles:
    engine.submit("exp", tile, scope={"region": "R1"})
engine.submit("exp", last_tile, scope={"region": "R1"}, complete="region")

# Submit tiles for region R2
for tile in region_r2_tiles:
    engine.submit("exp", tile, scope={"region": "R2"})
engine.submit("exp", last_tile, scope={"region": "R2"},
              complete=["region", "experiment"])
```

---

## Error Handling

The engine handles errors gracefully. A failed job does not crash the
pipeline or block other jobs.

- Failed jobs are recorded in `status()` with error details
- At scope boundaries, the scoped step receives successful results
  plus a `failures` list
- Workers survive step exceptions and continue serving new jobs

```python
status = engine.status("analysis")
if status["failed"] > 0:
    for f in status["failures"]:
        print(f"Step {f['step']} failed: {f['error']}")
```

---

## Configuration Summary

All configuration lives in two places:

**YAML** (workflow definition):
```yaml
metadata:
  functions_dir: "../steps"
  verbose: 2

workflow-name:
  - step_name:
      param: value
      scope: level      # optional
```

**Step METADATA** (execution requirements):
```python
METADATA = {
    "environment": "conda-env-name",   # optional
    "max_workers": 4,                  # optional, default 1
}
```

**Engine constructor** (runtime tuning):
```python
Engine(
    idle_timeout=300.0,       # seconds before idle workers shut down
    max_concurrent=8,         # thread pool size
    execution_timeout=300.0,  # per-step timeout
)
```
