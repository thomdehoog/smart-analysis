# `cell_analysis` workflow

Generic four-step cell analysis pipeline built on the smart-analysis
engine:

```
preprocess  ->  segment  ->  extract_features  ->  select_features
```

| Step | Does | Key params |
|---|---|---|
| `preprocess` | Loads an image, applies Gaussian blur and CLAHE. | `sigma`, `clip_limit` |
| `segment` | Runs Cellpose v4 (CPSAM) instance segmentation. Warm-loads the model in `state["model"]`. | `diameter`, `gpu` |
| `extract_features` | `skimage.measure.regionprops_table` over the masks against the original image. | `properties` |
| `select_features` | Picks cells by criterion: percentile, threshold, or top-N. | `feature`, `mode`, `direction`, `percentile` / `threshold` / `top_n` |

The output of the pipeline (under `pipeline_data["select_features"]`) is:

```python
{
    "selected_labels": [int, ...],   # cell IDs in the segmentation mask
    "n_selected": int,
    "n_total": int,
    "feature": str,                  # which feature was used
    "mode": str,                     # percentile | threshold | top_n
    "direction": str,                # high | low
    "cutoff": float | None,          # the actual threshold applied
}
```

Downstream steps (or your acquisition code) can consume `selected_labels`
to drive feedback into the microscope, mask out non-selected cells, or
export to CSV.

## Run it

The pipeline needs `cellpose`, `scikit-image`, and `pyyaml` (plus the
engine itself) in the active Python environment. The
[`workflows/rare_event_selection/environments/setup_env.py`](../rare_event_selection/environments/setup_env.py)
script installs the same dependency set under
`SMART--rare_event_selection--main`, which works for this workflow too.
On a developer machine that already has a cellpose-capable conda env
(e.g. `dino3_test`), use that.

```bash
# from a cellpose-capable conda env:
python workflows/cell_analysis/run_pipeline.py
python workflows/cell_analysis/run_pipeline.py --source skimage.human_mitosis --feature area --percentile 95
python workflows/cell_analysis/run_pipeline.py --source path/to/image.tif --feature mean_intensity --top-n 10
```

## Tune the selection

Three modes via `select_features` parameters in the YAML or the CLI:

```yaml
# Top 5% by area (default)
- select_features:
    feature: "area"
    mode: "percentile"
    percentile: 95
    direction: "high"

# Cells above a fixed area threshold
- select_features:
    feature: "area"
    mode: "threshold"
    threshold: 200
    direction: "high"

# Top 10 cells by mean intensity
- select_features:
    feature: "mean_intensity"
    mode: "top_n"
    top_n: 10
    direction: "high"

# Smallest 5% of cells (anti-selection)
- select_features:
    feature: "area"
    mode: "percentile"
    percentile: 5
    direction: "low"
```

## Test it

```bash
pytest workflows/cell_analysis/tests/                      # everything in env
pytest workflows/cell_analysis/tests/ -m "not cellpose"    # fast subset
pytest workflows/cell_analysis/tests/ -m cellpose          # real Cellpose (slow)
```

The test file covers `select_features` directly (six unit tests over
the three modes and both directions), an end-to-end smoke test using
real preprocess + extract + select with a stubbed segment (no cellpose
required), and two real-cellpose tests that auto-skip when the
runtime is unavailable.
