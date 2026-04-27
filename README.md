# Smart Analysis

[![tests](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml/badge.svg?branch=v4-engine)](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python pipeline engine for scientific image analysis, built for
real-time adaptive feedback microscopy.

## Why it exists

- **Live adaptive feedback.** Submit each tile to the engine as the
  microscope acquires it. The engine processes each tile, aggregates
  when a region is done, and you use the results to decide what to
  acquire next.
- **Per-step conda environments.** Each step runs in its own conda
  environment. Libraries that clash at the OS level (PyTorch,
  scikit-image, Cellpose) coexist in the same recipe without conflict.
- **Recipes in YAML.** A recipe is a YAML file listing the steps to
  run. Each step is a small Python function. Reuse the same step
  across many recipes.
- **Warm workers.** Heavy objects like a Cellpose model load once and
  stay in memory across submissions. Per-tile latency stays low during
  live acquisition.
- **Same API for batch.** Recipes written for live microscopy also
  process saved datasets without changes.

## What a recipe looks like

A recipe has three parts: a YAML file listing the steps, the step
files themselves, and a short Python runner that submits jobs and
reads results.

**1. `pipeline.yaml`** — the recipe. A typical microscopy workflow:
preprocess an image, detect objects, select the ones of interest, and
write feedback for the microscope.

```yaml
metadata:
  functions_dir: "./steps"

cell_analysis:
  - preprocess:
      sigma: 1.0
      clip_limit: 0.03

  - detect_objects:
      threshold: 0.5

  - select_objects:
      feature: "area"
      percentile: 95

  - feedback:
      output_dir: "./feedback"
```

**2. `steps/select_objects.py`** — one of the four steps. Each step is
a Python file with a `run` function. It reads from `pipeline_data`,
adds its own outputs under its own key, and returns it.

```python
"""Pick the top N% of detected objects by a chosen feature."""

METADATA = {"max_workers": 1}

def run(pipeline_data, state, **params):
    feature = params.get("feature", "area")
    percentile = params.get("percentile", 95)

    # Read from the previous step's output.
    objects = pipeline_data["detect_objects"]["objects"]

    # Pick the top by feature.
    values = sorted(obj[feature] for obj in objects)
    cutoff = values[int(len(values) * percentile / 100)]
    selected = [obj for obj in objects if obj[feature] >= cutoff]

    # Add output under this step's name; later steps read it the same way.
    pipeline_data["select_objects"] = {
        "selected": selected,
        "n_selected": len(selected),
    }
    return pipeline_data
```

**3. `run.py`** — submit jobs and collect results. These are the four
engine methods you'll use most:

```python
import time
from engine import Engine

with Engine() as e:
    # Register the recipe under a name you'll use to refer to it.
    e.register("cell_analysis", "pipeline.yaml")

    # Submit a job. Returns immediately; the engine runs it in the
    # background. Submit as many jobs as you like.
    e.submit("cell_analysis", {"image_path": "tile_001.tif"})

    # Check progress at any time. Returns counts of pending, running,
    # completed, and failed jobs, plus details on any failures.
    print(e.status("cell_analysis"))

    # Drain finished results. Each call returns whatever has completed
    # since the last call (and removes it from the queue).
    while not (results := e.results("cell_analysis")):
        time.sleep(0.05)

print(results[0]["select_objects"]["n_selected"])
```

For a fully runnable version of this shape, see
[`workflows/rare_event_selection/`](workflows/rare_event_selection/) (the
real workflow this example mirrors) or
[`examples/`](examples/) (smaller self-contained examples).

## Install and test

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
pip install -e .[test]
pytest
```

Requires Python 3.10+. Conda is optional (only needed for per-step
environment isolation).

## Where to go next

- [`examples/`](examples/) — four runnable workflows: hello world,
  scoped aggregation, environment isolation, adaptive microscopy.
- [`AGENTS.md`](AGENTS.md) — repo brief with concepts, file map, and
  design philosophy. Useful for fast onboarding (humans and AI).
- [`docs/Engine_v4_Design.md`](docs/Engine_v4_Design.md) — full design
  rationale.
- [`docs/Usage_Guide.md`](docs/Usage_Guide.md) — tutorial walkthrough.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development setup.
- [`CHANGELOG.md`](CHANGELOG.md) — version history.

## License

MIT. See [LICENSE](LICENSE).
