# Smart Analysis

[![tests](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml/badge.svg?branch=v4-engine)](https://github.com/thomdehoog/smart-analysis/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python pipeline engine for scientific image analysis. Define multi-step
workflows in YAML; the engine runs each step in a worker subprocess (in
the conda env you choose) and passes a shared `pipeline_data` dict
between them. First-class support for **live adaptive microscopy** —
tile-by-tile streaming with scoped aggregation that feeds results back
into acquisition decisions in real time. The same API also runs as a
plain post-acquisition batch tool.

## Install

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
pip install -e .[test]
```

Requires Python 3.10+. Conda is optional (only needed if you want to
isolate steps in different environments).

## A 30-line example

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

## Where to go next

- [`examples/`](examples/) — four runnable workflows: hello world,
  scoped aggregation, environment isolation, adaptive microscopy.
- [`AGENTS.md`](AGENTS.md) — repo brief: concepts, file map, design
  philosophy, anti-patterns. Useful for fast onboarding (humans and AI).
- [`docs/Engine_v4_Design.md`](docs/Engine_v4_Design.md) — full design
  rationale.
- [`docs/Usage_Guide.md`](docs/Usage_Guide.md) — tutorial walkthrough.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to develop and submit changes.
- [`CHANGELOG.md`](CHANGELOG.md) — version history.

## Testing

```bash
pytest                        # everything runnable in the active env
pytest -m "not slow"          # fast subset
pytest -m cellpose            # only real-cellpose tests
pytest engine/                # engine unit tests only
```

## License

MIT. See [LICENSE](LICENSE).
