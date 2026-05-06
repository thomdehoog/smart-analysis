"""pick_targets -- v0 stub.

Returns an empty picks list so Step 5 short-circuits cleanly.
Real top-N picking + pixel-to-stage conversion land when Step 4
of the notebook is implemented (design 6.3).
"""

METADATA = {"max_workers": 1}


def run(pipeline_data, state, **params):
    pipeline_data["pick_targets"] = {"picks": []}
    return pipeline_data
