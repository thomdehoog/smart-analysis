"""Expensive target tile. Pretends to acquire a high-res image."""


def run(pipeline_data, state, **params):
    inp = pipeline_data["input"]
    pipeline_data["target"] = {
        "row": inp["row"],
        "col": inp["col"],
        "high_res": True,
    }
    return pipeline_data
