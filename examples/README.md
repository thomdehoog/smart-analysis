# Examples

Self-contained, runnable examples of the smart-analysis pipeline engine.
Most examples are directories with their own `pipeline.yaml`, `steps/`, and
`run.py`. The notebook example runs the public object-analysis and
target-discovery workflows on a real microscopy image and plots the results
inline. Copy any example into your own project as a starting point.

Each `run.py` adds the repo root to `sys.path`, so the examples work
straight from a clone -- no install needed beyond `pyyaml`.

| #  | Directory                 | What it teaches                                |
|----|---------------------------|------------------------------------------------|
| 01 | `01_hello_world`          | Engine basics: register, submit, results       |
| 02 | `02_scoped_aggregation`   | Tile-and-aggregate pattern via `scope`         |
| 03 | `03_environment_isolation`| Per-step conda env via `METADATA`              |
| 04 | `04_adaptive_microscopy`  | Live feedback loop: cheap survey -> targeted   |
| 05 | `05_object_analysis_target_discovery.ipynb` | Object analysis -> clustering target discovery |

## Run one

```bash
python examples/01_hello_world/run.py
```

## Run all

From the repo root:

```bash
python examples/01_hello_world/run.py
python examples/02_scoped_aggregation/run.py
python examples/03_environment_isolation/run.py
python examples/04_adaptive_microscopy/run.py
```

Example 03 requires the conda env `SMART--basic_test--env_a`. If it is
not present the example prints a skip message and exits cleanly.

Example 05 is a notebook. It uses `skimage.data.immunohistochemistry()` and runs the
dedicated object-analysis and target-discovery conda environments:
`SMART--object_analysis--vision`, `SMART--object_analysis--classical`,
`SMART--target_discovery--main`, and `SMART--target_discovery--cluster`.
