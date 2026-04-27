"""Cheap overview tile. Records the tile's coords + 'brightness' value."""


def run(pipeline_data, state, **params):
    inp = pipeline_data["input"]
    pipeline_data["overview"] = {
        "row": inp["row"],
        "col": inp["col"],
        "brightness": inp["brightness"],
    }
    return pipeline_data
