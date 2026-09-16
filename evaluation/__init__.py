"""TwinGuard Evaluation Module.

Benchmarks task success, execution duration, and hardware performance metrics.
"""

from evaluation.executor import (
    WORKED_EXAMPLE_INSTRUCTION,
    execute_primitive_step,
    resolve_step_target,
    run_plan,
)

__all__ = [
    "WORKED_EXAMPLE_INSTRUCTION",
    "execute_primitive_step",
    "resolve_step_target",
    "run_plan",
]
