# Smart Analysis

[![tests](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml/badge.svg?branch=v4-engine)](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python pipeline engine for scientific image analysis, built for
real-time adaptive feedback microscopy.

## Why it exists

- **Live adaptive feedback.** Submit work tile-by-tile as the
  microscope acquires; aggregate by region as scans complete; feed
  results back into the next acquisition decision in real time.
- **Per-step conda environments.** Each step runs in its own conda env
  via subprocess, so PyTorch, scikit-image, Cellpose, and other clashing
  native libraries coexist in one workflow without DLL conflicts.
- **Composable recipes.** Define multi-step workflows in YAML; write
  each step as a plain Python function. Reuse and remix steps across
  recipes without rewriting plumbing.
- **Warm workers for heavy models.** Cellpose, segmentation, and other
  expensive objects load once and stay hot across submissions, keeping
  per-tile latency sub-second during live acquisition.
- **Same API for batch.** Recipes you write for live microscopy also
  process archived datasets unchanged.

## What a recipe looks like

**`steps/double_it.py`**
```python
METADATA = {"max_workers": 1}

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
