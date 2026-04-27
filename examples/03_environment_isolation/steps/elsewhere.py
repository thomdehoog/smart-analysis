"""Step that requests a different conda env via METADATA.

The engine reads METADATA via AST at register() time, then spawns a
worker subprocess in the named env. The 'import' lines below run inside
that subprocess, not in the orchestrator -- so this step's deps don't
have to be installed alongside the orchestrator's deps.
"""

METADATA = {
    "environment": "SMART--basic_test--env_a",
}


def run(pipeline_data, state, **params):
    import sys
    pipeline_data["elsewhere"] = {"executable": sys.executable}
    return pipeline_data
