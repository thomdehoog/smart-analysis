"""Per-tile step. Wraps the input value in a tile_result dict."""


def run(pipeline_data, state, **params):
    value = pipeline_data["input"]["value"]
    pipeline_data["tile_result"] = {"v": value}
    return pipeline_data
