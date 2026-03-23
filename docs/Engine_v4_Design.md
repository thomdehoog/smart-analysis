# Smart Analysis Engine v4 -- Design

## Overview

v4 is a ground-up simplification of the v3 pipeline engine. The core purpose is
unchanged: orchestrate multi-step analysis pipelines with scoped aggregation,
priority scheduling, and warm worker management. What changes is the removal of
every axis of complexity that v3 introduced but did not earn its keep.

Removed in v4:

- GPU/CPU device category and GPU slot management
- "Local" execution (engine never runs step code)
- Isolation modes (minimal/maximal)
- Two-axis scope (spatial/temporal)

What remains is a clean four-method API, a single scope axis, subprocess-only
execution, and per-environment worker pools with dynamic scaling.

The engine derives its behavior from two inputs:

1. **YAML** -- step order, parameters, scopes (workflow definition)
2. **METADATA** -- environment and max_workers (execution requirements)

---

## Step Interface

### METADATA

Each step file declares a `METADATA` dict with two optional fields:

```python
METADATA = {
    "environment": "SMART--segment",  # conda env (default: orchestrator's env)
    "max_workers": 1,                 # max parallel workers allowed (default: 1)
}
```

Both fields are optional. A step with no METADATA runs in the orchestrator's
conda environment with a single worker. The orchestrator's environment is
auto-detected at engine startup from `CONDA_DEFAULT_ENV` (or `sys.executable`
as fallback). No manual configuration is needed.

The engine reads METADATA via AST parsing -- no step code is ever imported or
executed by the engine process. This is safe regardless of what the step imports,
because the engine never loads the module.

There is no `device` field. GPU vs CPU is not a concept the engine manages.
If a step needs a GPU, it lives in a conda environment that has GPU libraries
installed. The `max_workers` field controls how many parallel instances are
allowed, which is how the step author expresses resource constraints (e.g.,
`max_workers: 1` for a GPU model that cannot share VRAM).

### run() function

Every step exposes a single `run()` function:

```python
def run(pipeline_data, state, **params):
    ...
    return pipeline_data
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline_data` | dict | Input data. For immediate steps, this is the output of the previous step. For scoped steps, this contains a `"results"` list of accumulated outputs. |
| `state` | dict | Persistent per-step per-worker dict. Empty on first call, retained across calls. Garbage collected when the worker shuts down. |
| `**params` | keyword args | Parameters from the YAML configuration. |

The function must return a dict (the updated `pipeline_data`).

### Imports

Steps use top-level imports -- normal Python, one convention:

```python
from cellpose.models import CellposeModel
import numpy as np

METADATA = {"environment": "SMART--segment", "max_workers": 1}

def run(pipeline_data, state, **params):
    if "model" not in state:
        state["model"] = CellposeModel(gpu=True)
    masks, flows, styles = state["model"].eval(pipeline_data["image"])
    pipeline_data["masks"] = masks
    return pipeline_data
```

Top-level imports are safe because:

1. **Routing uses AST** -- the engine parses METADATA without executing the file.
   Import errors in step files never affect the engine process.
2. **Workers run in the correct conda environment** -- when the worker loads the
   module, the imports resolve against the target environment's packages.

There are no special import conventions (no imports inside functions, no lazy
import wrappers). Steps are normal Python files.

### Cold and warm execution

The first time a worker executes a step, it is a **cold start**: the module is
loaded (imports execute), and the `state` dict is empty. The step populates
`state` with whatever it needs (model weights, lookup tables, compiled kernels).

Subsequent calls to the same step on the same worker are **warm**: the module is
already cached in memory, and `state` retains its contents. There is no import
cost and no re-initialization.

When the worker shuts down (idle timeout or engine shutdown), `state` is garbage
collected. This frees GPU memory, file handles, and any other resources held in
the dict. There is no explicit cleanup protocol -- Python's garbage collector
handles it.

### state isolation

The `state` dict is **per-step per-worker**. A worker that executes both
`preprocess` and `segment` maintains two separate state dicts -- one for each
step. They do not interfere.

This means a worker running in `SMART--segment` can hold a Cellpose model in
`segment`'s state and a preprocessing cache in `preprocess`'s state
simultaneously, without either step knowing about the other.

---

## Pipeline YAML

The YAML defines workflow structure only: step order, parameters, and scopes.
There is no execution configuration in the YAML -- no environment, no isolation
mode, no device hints. Those belong in the step's METADATA.

```yaml
metadata:
  functions_dir: "../steps"
  verbose: 2                   # optional, 0=silent, 1=errors, 2=steps, 3=debug

workflow-name:
  - preprocess:
      sigma: 1.0

  - segment:
      model: cpsam

  - stitch:
      scope: group

  - normalize:
      scope: all
```

### Step entries

Each entry in the workflow list is a single-key dict. The key is the step name
(maps to `<functions_dir>/<name>.py`). The value is a dict of parameters passed
as `**params` to `run()`.

A step with no parameters can be written as:

```yaml
  - feedback:
```

### Scope

Scope is a single string value attached to a step. It controls when that step
is allowed to run.

- **No scope** = immediate. The step runs as soon as the previous step completes.
  This is the default.
- **Scope present** = the step waits until the named scope level is signaled
  complete via the `complete` parameter on `submit()`.

Scope values (`group`, `carrier`, `compartment`, `all`, etc.) are user-defined
strings. The engine does not interpret their meaning -- it matches them against
completion signals. The calling layer decides what scope levels mean for its
domain.

### Scope matching

When `complete="X"` is signaled on a submit call, the engine triggers any step
with `scope: X` in the YAML. To determine which jobs belong to the completing
scope group, the engine applies one rule:

- If `"X"` is a key in the job's scope dict, match by value. Only jobs where
  `scope["X"]` equals the completing job's `scope["X"]` are collected.
- If `"X"` is not a key in any job's scope dict, collect all results from the
  previous phase with no filtering.

This means `"all"` is not a special keyword -- it works because no job has
`"all"` as a scope key, so the engine collects everything. Any string not used
as a scope key on `submit()` behaves the same way.

Example:

```python
engine.submit("a", d1, scope={"group": "R3"})           # group=R3
engine.submit("a", d2, scope={"group": "R3"})           # group=R3
engine.submit("a", d3, scope={"group": "R4"})           # group=R4

# complete="group" with scope["group"]="R3" --> collects d1, d2 only
engine.submit("a", d4, scope={"group": "R3"}, complete="group")

# complete="all" --> "all" is not a scope key --> collects everything
engine.submit("a", d5, scope={"group": "R4"}, complete=["group", "all"])
```

### Scope only narrows

Same as v3: scopes form a one-way funnel. Once data is aggregated at a scope
boundary, it stays aggregated or narrows further. It never fans back out.

```
immediate    -->    group    -->    all
(per job)        (per region)    (everything)
```

Aggregation is lossy. A scoped step reduces many results into one (e.g., 9 tile
masks into 1 stitched image). You cannot un-aggregate. Each scope boundary
produces a single output that flows forward.

### Data at scope boundaries

When a scope completes, the engine collects the output of the last immediate
step from every job in that scope group and passes them as a list (in submission
order) to the scoped step:

```yaml
workflow-name:
  - segment:              # immediate, runs per job
  - stitch:               # scoped, gets ALL segment outputs for the group
      scope: group
```

```python
# stitch.py receives:
def run(pipeline_data, state, **params):
    # pipeline_data["results"] is a list of segment outputs
    all_masks = [r["masks"] for r in pipeline_data["results"]]
    stitched = stitch_masks(all_masks)
    return {"stitched": stitched}
```

Submission order is preserved in the results list. Failed jobs are excluded
from the results list but reported separately (see Error Handling).

### Phases

The engine internally splits the step list into phases at scope boundaries:

```
Phase 0 (immediate):   preprocess --> segment      [runs per job]
-- scope: group --
Phase 1 (scoped):      stitch --> features          [runs once per scope group]
-- scope: all --
Phase 2 (scoped):      normalize                    [runs once when all complete]
```

Phase 0 has no scope trigger -- it runs immediately on submission. Each
subsequent phase waits for its scope to be signaled complete.

### functions_dir

The `metadata.functions_dir` field specifies the directory containing step
files, relative to the YAML file's location. Defaults to `"../steps"`.

### verbose

The optional `metadata.verbose` field controls output verbosity:

- `0` -- silent (no output)
- `1` -- errors only
- `2` -- step start/finish (default)
- `3` -- debug (full detail)

The value is passed to workers via `pipeline_data["metadata"]["verbose"]`. Steps
can check this to control their own output level.

---

## API

The engine exposes four core methods plus shutdown:

```python
engine = Engine()

# 1. Register a pipeline
engine.register("overview", "path/to/overview_pipeline.yaml")

# 2. Submit jobs
engine.submit("overview", data, scope={"group": "R3"}, priority=10)
engine.submit("overview", data, scope={"group": "R3"}, complete="group")

# 3. Check status
engine.status("overview")

# 4. Get results
engine.results("overview")

# Cleanup
engine.shutdown()
```

### register(name, yaml_path)

Register a pipeline by name. Parses the YAML, resolves functions_dir, reads
METADATA from all step files (via AST), and builds the internal phase structure.

This is the only time the engine touches step files -- and it only parses them,
never executes them.

```python
engine.register("overview", "pipelines/overview.yaml")
engine.register("target", "pipelines/target.yaml")
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Short name for the pipeline. Used in all subsequent calls. |
| `yaml_path` | str or Path | Path to the pipeline YAML file. |

A name can only be registered once. Re-registering raises an error.

### submit(name, data, scope=None, priority=None, complete=None)

Submit a job to a registered pipeline. Non-blocking.

```python
engine.submit("overview", tile_data, scope={"group": "R3"})
engine.submit("overview", tile_data, scope={"group": "R3"}, complete="group")
engine.submit("overview", tile_data, priority=10)
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Registered pipeline name. |
| `data` | dict | Input data for the pipeline. |
| `scope` | dict, optional | Labels which scope group this job belongs to. Keys are scope level names, values identify the group. E.g., `{"group": "R3", "carrier": "plate1"}`. |
| `priority` | int, optional | Higher = more urgent. Default is FIFO (submission order). |
| `complete` | str or list, optional | Signals that one or more scope levels are complete for this job's scope group. Pass a string for a single level or a list for multiple (e.g., `complete=["group", "all"]`). |

Phase 0 steps execute immediately upon submission. The job flows through all
immediate steps in order.

When `complete` is provided, the engine checks whether any scoped step is
waiting on that scope level. If so, and all preceding jobs in the matching scope
group have finished their immediate steps, the scoped phase is triggered.

The `complete` parameter is typically passed on the last submission for a scope
group:

```python
engine.submit("overview", tile1, scope={"group": "R3"})
engine.submit("overview", tile2, scope={"group": "R3"})
engine.submit("overview", tile3, scope={"group": "R3"}, complete="group")
```

The calling layer always knows when a scope is complete. The microscope knows
when a region is done scanning. The script knows when the last file has been
loaded. The engine does not count or guess -- it trusts the caller's signal.

### status(name=None)

Query the current state of a pipeline or the entire engine.

```python
# Status for one pipeline
engine.status("overview")

# Status for all pipelines
engine.status()
```

Returns a dict per pipeline:

```python
{
    "overview": {
        "pending": 3,
        "running": 2,
        "completed": 14,
        "failed": 1,
        "failures": [
            {
                "scope": {"group": "R3"},
                "step": "preprocess",
                "error": "ValueError: empty image"
            }
        ]
    }
}
```

When called with a name, returns only that pipeline's status dict (not nested).
When called without arguments, returns the full dict keyed by pipeline name.

The `failures` list includes scope labels, the step that failed, and the error
message. This gives the caller enough information to diagnose problems without
digging through logs.

### results(name)

Retrieve completed results for a pipeline.

```python
results = engine.results("overview")
```

Returns completed pipeline_data dicts. Failed jobs include error information
rather than result data. The caller can distinguish successes from failures.

Results are consumed on retrieval -- calling `results()` again returns only
new results accumulated since the last call.

### shutdown()

Shut down the engine, all worker pools, and all worker subprocesses.

```python
engine.shutdown()
```

Worker state dicts are garbage collected when subprocesses exit, freeing GPU
memory and other resources. The engine also supports context manager usage:

```python
with Engine() as engine:
    engine.register("analysis", "pipeline.yaml")
    engine.submit("analysis", data)
    # ...
# Workers cleaned up automatically
```

---

## Worker Management

### Workers are per-environment

A worker is a subprocess running in a specific conda environment. It can execute
any step that matches its environment -- it is not tied to a single step file.

```
Worker (orchestrator env):    preprocess --> feedback --> preprocess --> ...
Worker (SMART--segment):      segment --> segment --> segment --> ...
```

All steps declaring `"environment": "SMART--segment"` in their METADATA share
the same worker pool, regardless of which pipeline they belong to. This means
two registered pipelines that both use a `segment` step in `SMART--segment`
share workers for that environment.

### Why environments matter

Steps that share a worker process share `sys.modules`. Once a library is
imported, it stays loaded for the life of the process. This is the warm worker
benefit -- but it also means conflicting libraries cannot coexist.

A concrete example: on Windows, importing scipy or scikit-image corrupts the
DLL search state, which breaks PyTorch in the same process. There is no
workaround within a single process. The only clean fix is separate environments
(separate processes, separate `sys.modules`).

The rule is simple: steps that play nice together share an environment and
benefit from warm workers. Steps with conflicting dependencies go in different
environments, with data serialized at the boundary.

### Dynamic scaling

Worker pools scale dynamically based on load:

1. **Start with 1** -- the first job for an environment spawns one worker.
2. **Scale up** -- as jobs queue up and the single worker cannot keep pace,
   additional workers are spawned, up to `max_workers`.
3. **Idle timeout** -- workers that have been idle longer than the timeout
   (default: 300 seconds) are shut down. The pool can scale back to zero.

This means the system starts lightweight and grows only as needed. A pipeline
with 20 jobs and `max_workers: 4` might use 1 worker if jobs complete fast
enough, or scale to 4 if they are slow.

### max_workers is per-step

`max_workers` is a per-step concurrency limit, not a per-environment pool size.
The engine tracks how many instances of each step are running and never exceeds
that step's `max_workers`.

```python
# preprocess.py:  METADATA = {"max_workers": 5}
# segment.py:     METADATA = {"environment": "SMART--segment", "max_workers": 1}
```

When 20 preprocess jobs arrive, the engine scales up to 5 workers and drains
the queue. When segment jobs arrive with `max_workers: 1`, only one runs at a
time -- the other workers sit idle or handle different work.

The pool grows and shrinks based on demand. Workers are spawned when a job is
queued and all existing workers for that environment are busy, up to the
requesting step's `max_workers`. Idle workers are reused before new ones are
spawned.

### Module caching

Each worker maintains a module cache. When a step is executed for the first
time, the module is loaded (via `exec()`, same as v3). On subsequent calls
to the same step, the cached module is reused -- no reimport, no recompile.

A worker executing a different step in the same environment loads a fresh module
but stays in the same process. No subprocess spawn, no serialization cost
between steps in the same worker.

### State dict management

Each worker maintains a dict of state dicts, keyed by step name:

```
Worker process (SMART--segment):
  state_dicts = {
      "preprocess": {"cache": <LRUCache>},
      "segment":    {"model": <CellposeModel>},
  }
```

When the worker receives a job for step `segment`, it passes
`state_dicts["segment"]` to the `run()` function. The step reads and writes
to this dict freely. The worker never inspects or modifies it.

When the worker shuts down, all state dicts are garbage collected. GPU tensors,
open files, and other resources held in state are freed by Python's normal
cleanup mechanisms.

---

## Worker Communication

Communication between the engine and workers uses the same protocol as v3:

- **TCP sockets** with `multiprocessing.connection` (Listener/Client)
- **Pickle serialization** for messages
- **Authkey handshake** for security (random 32-byte key per worker)
- **Parent alive check** in workers to prevent orphans

### Spawn protocol

1. Engine allocates a random TCP port and creates a Listener with an authkey.
2. Engine spawns the worker subprocess via `conda run -n <env> python worker_script.py --port PORT --authkey HEX`.
3. Worker connects back to the engine as a Client.
4. Authkey handshake validates the connection.

### Message protocol

The v4 worker script receives messages in the form:

```
(step_path, pipeline_data, params)
```

And responds with:

```
("ok", result)          # success
("error", {"message": str, "traceback": str})  # failure
```

The shutdown sentinel is `None`.

### Changes from v3

The v4 worker script adds one thing: the `state` dict. The worker maintains
`state_dicts` internally and passes the appropriate one to each `run()` call.
The state dict is never serialized across the wire -- it lives entirely within
the worker process.

---

## Scheduling

When a step becomes ready to execute, the engine evaluates four conditions:

### 1. Order

All preceding steps in the pipeline must be complete for the job (or scope
group, for scoped steps). A step never runs before its predecessor.

### 2. Scope

If the step has a scope, the scope must be signaled complete via the `complete`
parameter on `submit()`. Steps without a scope run immediately.

For a step with `scope: group`, the engine waits until some `submit()` call
for the matching scope group includes `complete="group"`. At that point, all
jobs in the scope group must have completed their preceding immediate steps
before the scoped step is dispatched.

### 3. Priority

If multiple jobs are waiting for workers in the same pool, higher priority
jobs are dispatched first. Priority is an integer: higher values = more urgent.

### 4. FIFO

Within the same priority level, submission order determines execution order.
The first job submitted is the first to run.

No priority specified = pure FIFO. Priority is opt-in. Most pipelines will
never use it.

---

## Error Handling

### Graceful failure

If one job out of twenty fails, the other nineteen continue to completion. A
single failure does not crash the pipeline or block other work.

```
Job 1:  preprocess --> segment --> OK
Job 2:  preprocess --> FAIL (ValueError: empty image)
Job 3:  preprocess --> segment --> OK
...
Job 20: preprocess --> segment --> OK
```

Job 2's failure is recorded. Jobs 1 and 3-20 proceed normally.

### Failures at scope boundaries

When a scoped step triggers, it receives the results from all **successful**
jobs in the scope group. Failed jobs are excluded from the results list but
reported in metadata so the step author can handle them if needed:

```python
def run(pipeline_data, state, **params):
    results = pipeline_data["results"]      # successful results only
    failures = pipeline_data["failures"]    # list of failure info dicts
    # Step can log failures, adjust behavior, or ignore them
    ...
```

This lets the step author decide what partial failure means for their analysis.
A stitching step might tolerate a missing tile. A normalization step might
require all data and raise if any failed.

### Failure reporting

Failures surface in two places:

1. **status()** -- the `failures` list in the status dict includes scope
   labels, step name, and error message for every failed job.
2. **results()** -- failed jobs are included with error information rather than
   result data.

### Worker crashes

If a worker subprocess crashes (segfault, OOM kill, etc.), the engine detects
the broken connection and reports a `WorkerCrashedError` for the affected job.
Other workers in the pool are unaffected. The pool can spawn a replacement
worker for subsequent jobs.

### Step exceptions

If a step's `run()` function raises an exception, the worker catches it,
formats the traceback, and sends it back to the engine as an error response.
The worker remains alive and available for the next job. The exception is
surfaced to the caller via `status()` and `results()`.

---

## Architecture Layers

```
Calling layer              Knows the acquisition plan.
(smart-microscopy or       Calls submit() with scope info.
 post-acq script)          Signals completion via complete parameter.
       |
       v
Engine                     Central orchestrator. Registers pipelines,
                           manages submissions, tracks scopes,
                           schedules work, reports status/results.
       |
       v
Worker Pool                Per-environment pools. Dynamic scaling
                           up to max_workers. Queue-based job
                           distribution. Idle timeout cleanup.
       |
       v
Workers                    Per-environment subprocesses. Execute any
                           step matching their environment. Cache
                           modules and state dicts. Communicate via
                           TCP sockets with pickle serialization.
```

### Engine responsibilities

- Parse and register pipelines (YAML + step METADATA via AST)
- Accept job submissions and route them through phases
- Track scope groups and trigger scoped phases on completion signals
- Maintain per-environment worker pools
- Schedule jobs with priority and FIFO ordering
- Report status and deliver results

### What the engine does NOT do

- Execute step code (all execution is in worker subprocesses)
- Interpret scope semantics (scope values are opaque strings)
- Manage GPU/CPU resources (no device concept)
- Choose isolation strategies (every step is a subprocess)

---

## Use Cases

### Live adaptive microscopy

The primary use case. The microscope acquires tiles and submits them for
real-time analysis. Results feed back into acquisition decisions.

```python
engine = Engine()
engine.register("overview", "overview_pipeline.yaml")
engine.register("target", "target_pipeline.yaml")

# Tiles come in from microscope -- submit as they arrive
engine.submit("overview", tile1_data, scope={"group": "R3"})
engine.submit("overview", tile2_data, scope={"group": "R3"})
engine.submit("overview", tile3_data, scope={"group": "R3"}, complete="group")
# complete="group" triggers stitch step for R3

# Poll for results
status = engine.status("overview")
results = engine.results("overview")
# Results say: interesting cell at position X

# Submit targeted acquisition
engine.submit("target", target_data, scope={"group": "R3"})
```

The overview pipeline might look like:

```yaml
metadata:
  functions_dir: "../steps"

overview:
  - preprocess:
      sigma: 1.0
  - segment:
      model: fast
  - feedback:
  - stitch:
      scope: group
```

Tiles 1-3 each run preprocess, segment, and feedback immediately (Phase 0).
When `complete="group"` is signaled, stitch receives all three feedback outputs
and produces the stitched result.

### Post-acquisition batch analysis

Submit all images at once, optionally with scope grouping.

```python
engine = Engine()
engine.register("analysis", "analysis_pipeline.yaml")

# Submit all images with scope grouping
for i, img in enumerate(images):
    is_last = (i == len(images) - 1)
    engine.submit(
        "analysis", img,
        scope={"group": "batch1"},
        complete="group" if is_last else None,
    )

# Wait for completion
import time
while True:
    status = engine.status("analysis")
    if status["pending"] == 0 and status["running"] == 0:
        break
    time.sleep(1)

results = engine.results("analysis")
```

### Simple pipeline (no scopes, no priority)

Everything works without scopes or priority. Submit jobs, steps run in order:

```python
engine = Engine()
engine.register("simple", "pipeline.yaml")
engine.submit("simple", data)
```

The pipeline YAML:

```yaml
metadata:
  functions_dir: "../steps"

simple:
  - preprocess:
      sigma: 1.0
  - segment:
      model: fast
  - measure:
```

All three steps run immediately in sequence. No scopes, no completion signals,
no priority. This is the baseline -- everything else is opt-in.

### Multiple scope levels

A pipeline can use multiple scope levels for progressive aggregation:

```yaml
metadata:
  functions_dir: "../steps"

multi-scope:
  - preprocess:
  - segment:
  - stitch:
      scope: group
  - normalize:
      scope: all
```

```python
engine = Engine()
engine.register("analysis", "multi-scope.yaml")

# Submit group R3
engine.submit("analysis", t1, scope={"group": "R3"})
engine.submit("analysis", t2, scope={"group": "R3"}, complete="group")

# Submit group R4
engine.submit("analysis", t3, scope={"group": "R4"})
engine.submit("analysis", t4, scope={"group": "R4"}, complete="group")

# Last group -- signal both group and all complete
engine.submit("analysis", t5, scope={"group": "R5"}, complete=["group", "all"])
```

When `complete="group"` is signaled for R3, the stitch step runs with R3's
segment results. Same for R4. When `complete="all"` is signaled, the normalize
step runs with all stitch outputs.

---

## Results

### What results() returns

Results appear at every natural completion point:

- **Phase 0 (immediate):** each job's final immediate step output is a result.
  These appear as soon as each job finishes Phase 0, before any scope triggers.
- **Scoped phases:** the scoped step's output is a result. These appear when the
  scope completes and the scoped phase finishes.

Each result includes metadata so the caller can distinguish them:

```python
results = engine.results("overview")
for r in results:
    r["_phase"]         # 0, 1, 2, ...
    r["_scope"]         # None for Phase 0, or {"group": "R3"} etc.
    r["_scope_level"]   # None for Phase 0, or "group" etc.
```

This means the adaptive microscopy loop gets per-tile feedback immediately
(Phase 0 results) AND the stitch result later (Phase 1 result). The caller
filters by phase or scope as needed.

### In-memory by default

Results are stored in memory for fast access. The engine maintains a results
queue per pipeline. `results()` drains the queue and returns accumulated
results since the last call.

### Disk persistence is optional

The engine does not mandate disk-based storage. If a step needs to persist
data to disk (for large results, crash recovery, or downstream consumption),
the step author writes that logic in their `run()` function:

```python
def run(pipeline_data, state, **params):
    result = heavy_computation(pipeline_data["image"])
    # Step decides to save to disk
    np.save(pipeline_data["output_path"], result)
    pipeline_data["result_path"] = pipeline_data["output_path"]
    return pipeline_data
```

The engine does not force either approach. In-memory is the default for speed.
Disk persistence is a step-level concern.

---

## Backwards Compatibility

v4 changes the API surface from v3. The key differences:

| v3 | v4 |
|----|-----|
| `engine.create_run(yaml, priority)` | `engine.register(name, yaml)` |
| `run.submit(label, data, spatial, temporal)` | `engine.submit(name, data, scope, priority, complete)` |
| `run.scope_complete(spatial, temporal)` | `complete` parameter on `submit()` |
| `engine.status()` returns workers + runs | `engine.status(name)` returns per-pipeline info |
| Two-axis scope dict | Single scope string |
| Isolation modes (minimal/maximal) | Always subprocess |
| Device field (gpu/cpu) | No device concept |
| Local execution for same-env steps | All steps run in workers |

Migration is straightforward: replace `create_run` + `run.submit` +
`run.scope_complete` with `register` + `submit(complete=...)`.

The simplest usage remains simple:

```python
engine = Engine()
engine.register("pipeline", "pipeline.yaml")
engine.submit("pipeline", data)
```

---

## Thread Safety

The engine is designed for concurrent use from multiple threads (e.g., a
microscope control thread and a results polling thread).

- **Engine** -- internal lock protects pipeline registry and submission state.
- **Worker Pool** -- lock protects pool membership during scaling.
- **Job queue** -- thread-safe priority queue for pending work.
- **Results** -- thread-safe queue per pipeline.
- **Status** -- reads a consistent snapshot under locks.

The calling layer can safely call `submit()` from one thread and `status()` or
`results()` from another.

---

## Configuration Summary

All configuration lives in exactly two places:

### YAML (workflow definition)

```yaml
metadata:
  functions_dir: "../steps"    # where step files live (default: "../steps")

workflow-name:
  - step_name:
      param: value             # passed as **params to run()
      scope: level             # optional, triggers on scope completion
```

### Step METADATA (execution requirements)

```python
METADATA = {
    "environment": "env-name",  # conda env (default: orchestrator's env)
    "max_workers": 1,           # parallel workers (default: 1)
}
```

There is no third configuration surface. No engine constructor flags for
isolation modes. No device declarations. No pipeline-level execution overrides.

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| No device field | GPU/CPU distinction added complexity for a problem solved by environments. A GPU step lives in a GPU-capable environment with `max_workers: 1`. |
| No local execution | Consistent execution model. The engine never loads step code. Every step runs in a subprocess, giving real parallelism (no GIL) and process isolation. |
| No isolation modes | Everything is a subprocess. No minimal/maximal toggle, no routing logic, no dual code paths. One execution model to understand and debug. |
| Single scope value | Simpler than two-axis spatial/temporal. A scope is a string, not a dict. The caller defines what scope levels mean for their domain. |
| scope on submit | Labels where a job belongs in the scope hierarchy. The engine groups jobs by scope for aggregation. |
| complete on submit | The caller always knows when a scope is done. Bundling completion with the last submit avoids a separate `scope_complete()` call. No separate method to learn. |
| max_workers in METADATA | The step author knows resource constraints (VRAM budget, CPU core count). The engine scales dynamically within those bounds. |
| State dict per-step per-worker | Warm models without top-level side effects. First call populates, subsequent calls reuse. Garbage collected on worker shutdown -- no explicit cleanup protocol. |
| Top-level imports only | One convention, normal Python. Safe because the engine uses AST (never executes files) and workers run in the correct conda environment. |
| register + submit API | Two methods to learn. Pipeline name maps to YAML once. Submit uses the name. No Run objects to manage. |
| Graceful failure | One failed job out of twenty does not crash the pipeline. Scoped steps receive successful results plus failure info. Errors surface in status() and results(). |
| Workers scale dynamically | Start with 1, add as needed up to max_workers. Idle timeout cleans up unused workers. No pre-allocation, no waste. |
| max_workers is per-step | Each step has its own concurrency limit. The pool scales dynamically but never runs more of a specific step than its max_workers allows. No cross-step resolution needed. |
| Results consumed on retrieval | results() drains the queue. Prevents unbounded memory growth for long-running pipelines. Caller processes results as they arrive. |
| No GPU slot management | v3's GPU mutual exclusion was complex (priority queue, dedicated thread, environment switching). In v4, `max_workers: 1` achieves the same constraint without special machinery. |
| AST-only METADATA reading | The engine process never imports step files. Safe regardless of what a step imports. No risk of import side effects or dependency conflicts in the engine. |
