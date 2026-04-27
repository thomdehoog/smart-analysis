"""Step with no METADATA -- runs in the orchestrator's conda env."""


def run(pipeline_data, state, **params):
    import sys
    pipeline_data["here"] = {"executable": sys.executable}
    return pipeline_data
