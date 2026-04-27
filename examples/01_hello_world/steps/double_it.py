"""Trivial step: doubles the number passed in via pipeline_data['input']."""


def run(pipeline_data, state, **params):
    n = pipeline_data["input"]["n"]
    pipeline_data["doubled"] = n * 2
    return pipeline_data
