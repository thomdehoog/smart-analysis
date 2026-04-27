"""Select Features -- pick cells matching a criterion from regionprops.

Reads the ``extract_features`` output and produces a list of selected
cell labels. Three selection modes are supported:

- ``percentile`` -- top (or bottom) N% of cells by ``feature``.
- ``threshold``  -- cells with ``feature`` >= (or <=) a fixed value.
- ``top_n``      -- the top (or bottom) N cells by ``feature``.

Inputs (from previous step)
    pipeline_data["extract_features"]["properties"] : regionprops dict
        Must contain "label" and the chosen feature.

Parameters (via YAML / **params)
    feature : str, default "area"
        The regionprops key to rank cells by.
    mode : str, default "percentile"
        One of ``percentile`` / ``threshold`` / ``top_n``.
    direction : str, default "high"
        ``high`` selects the largest, ``low`` selects the smallest.
    percentile : float, used when mode == "percentile"
        E.g. 95 selects the top 5% (with direction="high").
    threshold : float, used when mode == "threshold"
        Cells with feature >= threshold (high) or <= threshold (low).
    top_n : int, used when mode == "top_n"
        Pick exactly this many cells.

Outputs (under pipeline_data["select_features"])
    selected_labels : list[int]
    n_selected, n_total : int
    feature, mode, direction, cutoff : echoed configuration + the
        threshold value that was actually applied (None for top_n
        when no cells were selected).
"""

import numpy as np


METADATA = {
    "description": "Select cells by regionprops feature criterion",
    "version": "1.0",
}


def _select_percentile(values, percentile, direction):
    if direction == "high":
        cutoff = float(np.percentile(values, percentile))
        return values >= cutoff, cutoff
    cutoff = float(np.percentile(values, 100 - percentile))
    return values <= cutoff, cutoff


def _select_threshold(values, threshold, direction):
    cutoff = float(threshold)
    if direction == "high":
        return values >= cutoff, cutoff
    return values <= cutoff, cutoff


def _select_top_n(values, top_n, direction):
    order = np.argsort(values)
    if direction == "high":
        order = order[::-1]
    keep_idx = order[: int(top_n)]
    mask = np.zeros(len(values), dtype=bool)
    mask[keep_idx] = True
    cutoff = float(values[keep_idx[-1]]) if len(keep_idx) > 0 else None
    return mask, cutoff


_MODES = {
    "percentile": _select_percentile,
    "threshold": _select_threshold,
    "top_n": _select_top_n,
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    feature = params.get("feature", "area")
    mode = params.get("mode", "percentile")
    direction = params.get("direction", "high")

    if mode not in _MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Expected one of {sorted(_MODES)}."
        )
    if direction not in ("high", "low"):
        raise ValueError(
            f"Unknown direction '{direction}'. Expected 'high' or 'low'."
        )

    props = pipeline_data["extract_features"]["properties"]
    if feature not in props:
        raise KeyError(
            f"Feature '{feature}' not found in extracted properties: "
            f"{sorted(props)}"
        )
    if "label" not in props:
        raise KeyError(
            "extract_features must produce a 'label' key for selection."
        )

    values = np.asarray(props[feature], dtype=float)
    labels = np.asarray(props["label"])

    if mode == "percentile":
        mask, cutoff = _select_percentile(
            values, params.get("percentile", 95), direction
        )
    elif mode == "threshold":
        threshold = params.get("threshold")
        if threshold is None:
            raise ValueError("mode='threshold' requires a 'threshold' param.")
        mask, cutoff = _select_threshold(values, threshold, direction)
    else:  # top_n
        top_n = params.get("top_n")
        if top_n is None:
            raise ValueError("mode='top_n' requires a 'top_n' param.")
        mask, cutoff = _select_top_n(values, top_n, direction)

    selected_labels = labels[mask].tolist()

    verbose = pipeline_data["metadata"].get("verbose", 0)
    if verbose >= 2:
        print(f"  [select_features] {feature} {direction} {mode}: "
              f"{len(selected_labels)}/{len(labels)} cells "
              f"(cutoff={cutoff})")

    pipeline_data["select_features"] = {
        "selected_labels": selected_labels,
        "n_selected": len(selected_labels),
        "n_total": int(len(labels)),
        "feature": feature,
        "mode": mode,
        "direction": direction,
        "cutoff": cutoff,
    }
    return pipeline_data
