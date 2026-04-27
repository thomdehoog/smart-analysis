# Smart Analysis

A Python pipeline engine for scientific image analysis workflows, with
first-class support for **live adaptive microscopy**: streaming work
tile-by-tile from an acquiring microscope, aggregating into regions, and
feeding results back into acquisition decisions in real time. The same
API also runs as a plain post-acquisition batch tool.

[![tests](https://img.shields.io/badge/tests-165%20passing-brightgreen)](https://github.com/thomdehoog/smart-analysis)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> **Status**: v4 engine, MIT-licensed, used in active microscopy
> research. The public API (`Engine.register`, `submit`, `status`,
> `results`, `shutdown`) is stable.

---

## TL;DR

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
pip install -e .[test]
pytest
```

The engine is one class. Workflows are YAML + a folder of step files.
Each step gets a fresh subprocess (in the conda env you tell it to use).
A shared `pipeline_data` dict carries data between steps.

```python
from engine import Engine

with Engine() as e:
    e.register("analysis", "path/to/pipeline.yaml")
    e.submit("analysis", {"data_source": "skimage.human_mitosis"})
    while not (results := e.results("analysis")):
        time.sleep(0.1)

print(results[0]["feedback"]["n_selected"])
```

That's the whole thing. Read on for the *why*.

---

## Why does this exist?

Scientific Python pipelines have a dependency-conflict problem. A
typical workflow needs scikit-image for preprocessing, PyTorch for
segmentation, and specialized packages for feature extraction. These
ship native libraries (BLAS, MKL, fbgemm) that fight at the OS loader
level. The two common workarounds are both unhappy:

1. **One environment to rule them all.** Find a conda env where every
   package coexists. Time-consuming, sometimes impossible, breaks every
   time someone updates anything.
2. **Save and reload between steps.** Each step is a separate script,
   intermediate state goes to disk. Fragile, hard to reproduce, painful
   to modify, and useless for live acquisition where you need
   sub-second feedback.

Smart Analysis takes a third path: **the engine spawns a subprocess in
whichever conda env each step needs, and ferries `pipeline_data`
between them automatically.** No manual file I/O between steps. No
shared dependency stack to maintain.

For live microscopy, the same engine accepts work tile-by-tile from
the acquiring scope and runs scoped aggregation when a region finishes:

```python
# microscope acquires tile 1 of region R3
engine.submit("overview", tile1, scope={"group": "R3"})
engine.submit("overview", tile2, scope={"group": "R3"})
engine.submit("overview", tile3, scope={"group": "R3"}, complete="group")
# stitch step automatically runs on all three tiles
```

---

## Install

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
pip install -e .                # runtime only (just pyyaml)
pip install -e .[test]          # adds pytest + psutil for development
```

**Requirements**: Python 3.10+, conda (only if you want per-step
environment switching; the engine works fine without it for in-env
pipelines).

---

## A complete hello-world (~30 lines)

Three files: a step, a YAML, a runner.

**`steps/double_it.py`**
```python
"""A step receives pipeline_data, returns it (possibly modified).

METADATA at module level is read by AST -- the engine never imports
this file in the orchestrator process. It only lives in workers.
"""

METADATA = {
    "description": "Double the input number.",
    "max_workers": 1,
}


def run(pipeline_data, state, **params):
    pipeline_data["doubled"] = pipeline_data["input"]["n"] * 2
    return pipeline_data
```

**`pipeline.yaml`**
```yaml
metadata:
  functions_dir: "./steps"

example:
  - double_it:
```

**`run.py`**
```python
import time
from engine import Engine

with Engine() as e:
    e.register("example", "pipeline.yaml")
    e.submit("example", {"n": 21})

    while not (results := e.results("example")):
        time.sleep(0.05)

    print(results[0]["doubled"])   # -> 42
```

That is the whole programming model. Everything else is composition.

---

## Concepts

| Term | What it is |
|---|---|
| **Pipeline** | A registered workflow, defined by a YAML file, with a name. |
| **Step** | A `.py` file at `<functions_dir>/<name>.py` exposing `run(pipeline_data, state, **params) -> dict` and an optional `METADATA` dict. |
| **`pipeline_data`** | A dict carried through every step in a single submission. Each step reads what it needs and adds its outputs. Often abbreviated `pd` in step bodies — Python convention from pandas, **not** an actual `pandas.DataFrame`. |
| **`state`** | A per-step per-worker dict that survives across calls on the same worker. Use it to cache heavy objects (e.g. a Cellpose model) so they load once and stay hot. Reset on worker shutdown. |
| **Scope** | An aggregation boundary. `scope={"group": "R3"}` tags a submission as belonging to region `R3`; the next phase aggregates all `R3` submissions when you signal `complete="group"`. |
| **Phase** | A contiguous run of steps with the same scope level. Phase 0 is the per-submission steps; subsequent phases are scoped. |
| **METADATA** | A literal dict at module level in a step file: `{"environment": "...", "max_workers": ...}`. Read via AST; step code is never imported in the orchestrator. |

---

## How conda environment switching works

Step files declare which env they need:

```python
# steps/segment.py
from cellpose import models

METADATA = {"environment": "SMART--my_workflow--main", "max_workers": 1}

def run(pipeline_data, state, **params):
    if "model" not in state:
        state["model"] = models.CellposeModel(gpu=True)
    masks, _, _ = state["model"].eval(pipeline_data["preprocess"]["image"])
    pipeline_data["segment"] = {"masks": masks, "n_cells": int(masks.max())}
    return pipeline_data
```

The engine spawns one worker subprocess per env, in that env, and
reuses it across calls. The Cellpose model loads once on cold start and
sticks around in `state["model"]` until the worker is reaped (default:
5 minutes idle).

You import normally at the top of the step file. The engine never
imports the file in the orchestrator process — it parses METADATA via
`ast.literal_eval`. So an `import cellpose` at module level is safe
even if the orchestrator's env doesn't have cellpose.

### Naming convention for envs

```
SMART--<workflow>--<step>
SMART--rare_event_selection--main       # default env for the workflow
SMART--rare_event_selection--segment    # isolated env for one step
SMART--basic_test--env_a                # one of the engine test envs
```

The `SMART--` prefix is just a namespace so `clean_env.py` knows what
to remove without touching unrelated envs.

### Set up an env for a workflow

```bash
cd workflows/rare_event_selection/environments
python setup_env.py        # auto-detects GPU, builds the env via pip
python clean_env.py        # tears it down
```

The setup script picks the right PyTorch wheel for your machine
(CUDA / MPS / CPU), installs cellpose / scikit-image / etc. via pip
(avoiding conda+pip DLL conflicts on Windows), and runs diagnostics.

---

## Real-world example: rare event selection

The `workflows/rare_event_selection/` workflow is a complete cell
analysis pipeline:

1. **preprocess** — load image (skimage), Gaussian blur, CLAHE.
2. **segment** — Cellpose v4 (CPSAM) instance segmentation.
3. **extract_features** — `skimage.measure.regionprops_table`, select
   top N% of cells by feature value.
4. **feedback** — write JSON of selected cells (label, centroid,
   intensity) to disk for downstream tools to consume.

Run it on a sample image:

```bash
cd workflows/rare_event_selection/environments
python setup_env.py
cd ..
SMART_ENV=SMART--rare_event_selection--main python run_pipeline.py
```

Or run the test suite that exercises it end-to-end:

```bash
pytest workflows/rare_event_selection/tests/ -v
```

Tests marked `@pytest.mark.cellpose` skip cleanly if cellpose isn't
loadable in the active env; they run for real if it is.

---

## Project structure

```
smart-analysis/
├── AGENTS.md                          # brief for AI agents
├── README.md                          # you are here
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml                     # project metadata + pytest markers
├── requirements.txt                   # runtime deps (just pyyaml)
├── conftest.py                        # shared pytest fixtures
│
├── engine/                            # the engine package
│   ├── __init__.py                    # public surface
│   ├── _pipeline.py                   # Engine class, public API
│   ├── _pool.py                       # worker pool, per-env reuse
│   ├── _worker.py                     # one worker subprocess
│   ├── _run.py                        # YAML parsing, phase splitting
│   ├── _loader.py                     # AST METADATA extraction
│   ├── _errors.py                     # exception hierarchy
│   ├── conda_utils.py                 # conda discovery, GPU detection
│   ├── worker_script.py               # script run inside workers
│   ├── test_engine.py                 # ~70 unit tests
│   └── test_conda_utils.py            # conda-utils unit tests
│
├── workflows/
│   ├── basic_test/                    # synthetic workflow for engine tests
│   │   ├── environments/
│   │   ├── pipelines/
│   │   ├── steps/
│   │   └── tests/
│   │       ├── test_integration.py    # 18 end-to-end tests
│   │       ├── test_robustness.py     # 14 edge-case tests
│   │       ├── test_adversarial.py    # 34 stress / corruption tests
│   │       └── run_benchmarks.py      # performance benchmarks
│   │
│   └── rare_event_selection/          # real workflow: cellpose + skimage
│       ├── environments/
│       ├── pipelines/
│       ├── steps/
│       ├── tests/test_rare_event.py   # 9 tests (mock + real cellpose)
│       └── run_pipeline.py            # CLI runner
│
└── docs/
    ├── Engine_v4_Design.md            # full design rationale (~1000 lines)
    └── Usage_Guide.md                 # tutorial walkthrough
```

---

## Testing

One command:

```bash
pytest                        # everything runnable in the active env
pytest -m "not slow"          # fast subset
pytest -m cellpose            # only real-cellpose tests
pytest -m adversarial         # only stress / race / corruption
pytest engine/                # engine unit tests only
```

Tests marked `@pytest.mark.cellpose` auto-skip when cellpose isn't
loadable, so the suite is portable across envs. From a cellpose-capable
env (e.g. one created by `workflows/rare_event_selection/environments/setup_env.py`),
you get full end-to-end coverage including real Cellpose segmentation
of `skimage.human_mitosis`.

| Suite | Where | Count |
|---|---|---|
| Engine unit | `engine/test_engine.py` | ~70 |
| Conda utils | `engine/test_conda_utils.py` | 21 |
| Integration | `workflows/basic_test/tests/test_integration.py` | 18 |
| Robustness | `workflows/basic_test/tests/test_robustness.py` | 14 |
| Adversarial | `workflows/basic_test/tests/test_adversarial.py` | 34 |
| Rare event workflow | `workflows/rare_event_selection/tests/test_rare_event.py` | 9 |

---

## When *not* to use this

This is not a general workflow orchestrator. If you need:

- Distributed execution across machines → **Airflow / Prefect / Dagster**.
- Scheduled cron-like batch jobs → **Airflow / Prefect**.
- Long-lived persistent state with a database → not this.
- Notebook-style interactive analysis → use a notebook directly.

Smart Analysis is for **a single host running multi-step Python image
analysis with conflicting dependencies and a need for low-latency
feedback to instruments**. That's a narrow but real niche.

---

## Documentation

- [`AGENTS.md`](AGENTS.md) — fast onramp for AI agents
- [`docs/Engine_v4_Design.md`](docs/Engine_v4_Design.md) — full design rationale
- [`docs/Usage_Guide.md`](docs/Usage_Guide.md) — tutorial walkthrough
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to develop and submit changes
- [`CHANGELOG.md`](CHANGELOG.md) — version history

---

## License

MIT. See [LICENSE](LICENSE).
