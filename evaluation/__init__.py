"""TwinGuard Evaluation Module.

Benchmarks task success, execution duration, and hardware performance metrics.
"""

from evaluation.benchmark import run_benchmark
from evaluation.executor import (
    WORKED_EXAMPLE_INSTRUCTION,
    execute_primitive_step,
    resolve_step_target,
    run_plan,
)
from evaluation.openvino_export import export_to_openvino

__all__ = [
    "WORKED_EXAMPLE_INSTRUCTION",
    "execute_primitive_step",
    "export_to_openvino",
    "resolve_step_target",
    "run_benchmark",
    "run_plan",
]
