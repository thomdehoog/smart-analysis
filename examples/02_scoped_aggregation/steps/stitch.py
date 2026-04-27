"""Scoped aggregation step.

Fires once when the caller signals complete='group'. The engine fills
pipeline_data['results'] with every per-tile pipeline_data dict that
shared the same scope group.
"""


def run(pipeline_data, state, **params):
    results = pipeline_data["results"]
    total = sum(r["tile_result"]["v"] for r in results)
    pipeline_data["stitched"] = {
        "n_tiles": len(results),
        "sum": total,
    }
    return pipeline_data
