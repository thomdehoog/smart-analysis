"""Scoped step: pick the 'interesting' overview tiles.

Receives every overview_tile result for the scope group. Returns a list
of (row, col) coordinates above a brightness threshold -- the caller
will turn these into target submissions.
"""


def run(pipeline_data, state, **params):
    threshold = params.get("threshold", 0)
    picks = []
    for r in pipeline_data["results"]:
        ov = r["overview"]
        if ov["brightness"] >= threshold:
            picks.append((ov["row"], ov["col"]))
    pipeline_data["picks"] = picks
    return pipeline_data
