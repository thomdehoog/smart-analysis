"""segment_tile -- v0 stub.

Real cellpose segmentation lands when Step 4 of the notebook is
implemented. This stub registers cleanly so preflight can succeed
and Step 4 of the notebook can wire up the engine submit / drain
loop against a no-op pipeline.
"""

METADATA = {"max_workers": 1}


def run(pipeline_data, state, **params):
    pipeline_data["segment_tile"] = {
        "masks": None,
        "n_cells": 0,
    }
    return pipeline_data
