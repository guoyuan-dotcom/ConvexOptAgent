__all__ = ["ExecutionPolicy", "LocalExecutionRunner", "WORKER_MODE_ARG", "run_worker_mode"]


def __getattr__(name: str):
    if name in __all__:
        from convexopt_tutor_agent.execution.local_runner import (
            ExecutionPolicy,
            LocalExecutionRunner,
            WORKER_MODE_ARG,
            run_worker_mode,
        )

        exports = {
            "ExecutionPolicy": ExecutionPolicy,
            "LocalExecutionRunner": LocalExecutionRunner,
            "WORKER_MODE_ARG": WORKER_MODE_ARG,
            "run_worker_mode": run_worker_mode,
        }
        return exports[name]
    raise AttributeError(name)
