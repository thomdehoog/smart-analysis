"""
Exception hierarchy for the pipeline engine.

    WorkerError (base for all subprocess issues)
    +-- WorkerSpawnError      subprocess failed to start or connect
    +-- WorkerCrashedError    subprocess died during execution
    +-- StepExecutionError    step's run() raised an exception
                              (includes .remote_traceback from subprocess)

    ScopeError                invalid scope configuration or completion

All step execution goes through worker subprocesses. StepExecutionError
covers any step failure. WorkerSpawnError and WorkerCrashedError cover
infrastructure issues with the subprocess itself.
"""


class WorkerError(Exception):
    """Base exception for all worker subprocess errors."""


class WorkerSpawnError(WorkerError):
    """Worker subprocess failed to start or connect back."""


class WorkerCrashedError(WorkerError):
    """Worker process died unexpectedly during execution."""


class StepExecutionError(WorkerError):
    """Step's run() raised an exception inside the worker subprocess."""

    def __init__(self, message, remote_traceback=None):
        super().__init__(message)
        self.remote_traceback = remote_traceback


class ScopeError(Exception):
    """Invalid scope configuration, missing results, or bad completion signal."""
