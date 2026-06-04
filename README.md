# Smart Analysis

[![tests](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml/badge.svg?branch=v4-engine)](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Smart Analysis is a local Python workflow engine for scientific image
analysis, built around live adaptive microscopy. It accepts work while a
microscope is acquiring images, keeps heavy analysis workers warm, and
returns results quickly enough for acquisition software to decide what to
image next.

The same API also runs saved datasets after acquisition. A workflow that
processes live tiles can usually be run as a batch pipeline without
rewriting the analysis steps.

## Why use it

- **Live feedback loops.** Submit each tile as it arrives, aggregate
  results when a region is complete, and feed the result back into the
  microscope controller.
- **Per-step environments.** Run incompatible libraries in separate conda
  environments while keeping one Python API at the orchestration layer.
- **Warm workers.** Load expensive objects such as Cellpose models once
  per worker and reuse them across submissions.
- **Subprocess isolation.** Every step runs outside the engine process, so
  a crashing or misconfigured analysis step fails its own job without
  taking down the orchestrator.
- **YAML recipes, Python steps.** Workflows are declared in YAML; each
  step is a small Python file with a `run()` function.

## Install

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
conda create -n SMART--analysis-dev python=3.12 -y
conda activate SMART--analysis-dev
python -m pip install -e ".[test]"
pytest -m "not cellpose and not slow"
```

Python 3.10 or newer is required. Use conda for local development and
microscope deployments; the engine's environment switching also relies
on conda when a step declares a separate environment in its `METADATA`.

## Minimal workflow

A workflow has three pieces: a YAML recipe, one or more step files, and a
runner that submits jobs. This complete workflow doubles a submitted
number.

`pipeline.yaml`:

```yaml
metadata:
  functions_dir: "./steps"

hello:
  - double_it:
```

`steps/double_it.py`:

```python
def run(pipeline_data, state, **params):
    n = pipeline_data["input"]["n"]
    pipeline_data["doubled"] = n * 2
    return pipeline_data
```

`run.py`:

```python
import time
from engine import Engine


with Engine() as engine:
    engine.register("hello", "pipeline.yaml")
    engine.submit("hello", {"n": 21})

    while not (results := engine.results("hello")):
        time.sleep(0.05)

print(results[0]["doubled"])  # 42
```

See [`examples/01_hello_world/`](examples/01_hello_world/) for this
workflow as runnable files, [`examples/`](examples/) for more engine
patterns, and [`workflows/`](workflows/) for larger microscopy workflows.

## Scoped aggregation

Live acquisition often has per-tile work followed by per-region work. For
example, segment every tile immediately, then stitch or summarize a whole
region once the microscope finishes acquiring it.

Declare the aggregating step with a `scope`:

```yaml
overview:
  - segment_tile:
  - stitch_region:
      scope: group
```

Submit tiles with matching scope labels. The acquisition layer signals
when the group is complete:

```python
engine.submit("overview", tile_1, scope={"group": "R3"})
engine.submit("overview", tile_2, scope={"group": "R3"})
engine.submit("overview", tile_3, scope={"group": "R3"}, complete="group")
```

`segment_tile` runs once per submitted tile. `stitch_region` runs once
for `R3` after the `complete="group"` submission and receives the
accumulated tile results in `pipeline_data["results"]`.

The engine does not guess completion. The caller knows when acquisition
for a region is done and tells the engine explicitly.

## Per-step environments

Steps run in worker subprocesses. By default, a worker uses the same
Python environment as the orchestrator. Add an `environment` entry to a
step's `METADATA` to run that step in another conda environment:

```python
METADATA = {
    "environment": "SMART--target_acquisition--main",
    "max_workers": 1,
}


def run(pipeline_data, state, **params):
    from cellpose import models

    if "model" not in state:
        state["model"] = models.CellposeModel(gpu=params.get("gpu", False))
    ...
```

The engine reads `METADATA` with `ast.literal_eval`; it does not import
step modules in the orchestrator process. Heavy imports happen only
inside the worker environment.

## Project structure

| Path | Purpose |
|---|---|
| `engine/` | The engine package and unit tests. Public API is re-exported from `engine/__init__.py`. |
| `examples/` | Small runnable workflows showing basic submission, scoped aggregation, environment isolation, and adaptive feedback. |
| `workflows/basic_test/` | Synthetic workflows used for integration, robustness, and adversarial tests. |
| `workflows/rare_event_selection/` | Cellpose and scikit-image workflow for rare-event selection. |
| `workflows/cell_analysis/` | Generic preprocess, segment, extract, and select workflow. |
| `workflows/object_analysis/` | Object-centered analysis with classical features and optional DINOv2 embeddings. |
| `workflows/target_discovery/` | Selects and clusters revisit targets from object tables and tile geometry. |
| `workflows/target_acquisition/` | Combined target-acquisition workflow: per-tile Cellpose segmentation plus coordinate conversion. |
| `docs/` | Usage guide and v4 design rationale. |
| `.github/workflows/` | Cross-platform pytest CI. |

## Testing

```bash
pytest -m "not cellpose and not slow"
pytest
pytest -m "not slow"
pytest engine/
pytest workflows/object_analysis/tests/ -m "not cellpose and not deep"
pytest workflows/target_discovery/tests/
pytest workflows/target_acquisition/tests/ -m "not cellpose"
```

Tests marked `cellpose` require Cellpose and Torch in the active
environment and skip cleanly when those imports fail. Tests marked
`deep` require a Torch/DINO-capable environment. Tests marked
`cluster` require the target-discovery clustering environment. Tests marked
`pooch` use public scikit-image sample data downloaded/cached by pooch.
Tests marked `conda_env` require the small test environments created by
`workflows/basic_test/environments/setup_env.py`.

## Limitations

Smart Analysis is intentionally local and small. It is not a distributed
scheduler, a queue service, a notebook framework, or a result-storage
system. Step files are trusted Python code controlled by the workflow
author.

For background and deeper examples, see
[`docs/Usage_Guide.md`](docs/Usage_Guide.md) and
[`docs/Engine_v4_Design.md`](docs/Engine_v4_Design.md).

## License

MIT. See [LICENSE](LICENSE).
