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
from evaluation.report import generate_report
from evaluation.runner import run_evaluation

__all__ = [
    "WORKED_EXAMPLE_INSTRUCTION",
    "execute_primitive_step",
    "export_to_openvino",
    "generate_report",
    "resolve_step_target",
    "run_benchmark",
    "run_evaluation",
    "run_plan",
]

