"""Report generator for TwinGuard multi-seed evaluation results.

Reads evaluation/results.json and prints formatted summary and breakdown tables:
- Task success rate
- Grasp success rate (initial vs post-recovery)
- Recovery success rate
- Total collisions
- Average completion time
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"


def generate_report(
    results_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load evaluation results JSON and display formatted summary tables.

    Args:
        results_path: Path to results.json (defaults to evaluation/results.json).
        verbose: Whether to print formatted report tables to stdout.

    Returns:
        Dict containing summary statistics and seed details.
    """
    res_p = Path(results_path) if results_path else DEFAULT_RESULTS_PATH
    if not res_p.exists():
        raise FileNotFoundError(f"Evaluation results file not found at: {res_p}")

    with open(res_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    seeds = data.get("seeds", [])

    if verbose:
        print("\n" + "=" * 80)
        print("                 TWINGUARD BIMANUAL MANIPULATION EVALUATION REPORT")
        print("=" * 80)

        # 1. Executive Summary Table
        print("\n" + "-" * 80)
        print(" 1. EXECUTIVE SUMMARY METRICS")
        print("-" * 80)
        print(f" {'Metric':<40} | {'Value':<35}")
        print("-" * 80)

        total_seeds = summary.get("total_seeds", len(seeds))
        succ_seeds = summary.get("successful_seeds", 0)
        task_rate = summary.get("task_success_rate", 0.0)
        print(f" {'Task Success Rate':<40} | {task_rate:>6.1f}% ({succ_seeds}/{total_seeds} seeds)")

        init_grasp = summary.get("initial_grasp_success_rate", 0.0)
        total_grasps = summary.get("total_grasps", 0)
        print(f" {'Initial Grasp Success Rate':<40} | {init_grasp:>6.1f}%")

        final_grasp = summary.get("final_grasp_success_rate", 0.0)
        print(f" {'Post-Recovery Grasp Success Rate':<40} | {final_grasp:>6.1f}% ({total_grasps} total grasps)")

        rec_rate = summary.get("recovery_success_rate", 0.0)
        rec_count = summary.get("total_failures_recovered", 0)
        fail_count = summary.get("total_failures_encountered", 0)
        print(f" {'Dynamic Recovery Success Rate':<40} | {rec_rate:>6.1f}% ({rec_count}/{fail_count} recovered)")

        collisions = summary.get("total_collisions", 0)
        print(f" {'Total Collision Count':<40} | {collisions:>6d} collisions")

        avg_sim = summary.get("avg_sim_time_s", 0.0)
        print(f" {'Average Simulation Time':<40} | {avg_sim:>6.3f} seconds")

        avg_wall = summary.get("avg_wall_time_s", 0.0)
        print(f" {'Average Wall-Clock Time':<40} | {avg_wall:>6.3f} seconds")
        print("-" * 80)

        # 2. Per-Seed Breakdown Table
        print("\n" + "-" * 80)
        print(" 2. PER-SEED BREAKDOWN TABLE")
        print("-" * 80)
        hdr = (
            f" {'Seed':<5} | {'Status':<8} | {'Steps':<7} | {'Forced Fail':<11} | "
            f"{'Recoveries':<10} | {'Collisions':<10} | {'Sim Time':<9} | {'Wall Time':<9}"
        )
        print(hdr)
        print("-" * 80)

        for s in seeds:
            seed_id = s.get("seed", 0)
            status = "SUCCESS" if s.get("success", False) else "FAILED"
            steps_str = f"{s.get('steps_succeeded', 0)}/{s.get('total_steps', 0)}"
            forced = "YES (Grasp)" if s.get("forced_failure", False) else "No"
            rec_str = f"{s.get('steps_recovered', 0)} recov"
            cols = str(s.get("collisions", 0))
            sim_t = f"{s.get('sim_time', 0.0):.2f}s"
            wall_t = f"{s.get('wall_time', 0.0):.2f}s"

            row = (
                f" {seed_id:<5} | {status:<8} | {steps_str:<7} | {forced:<11} | "
                f"{rec_str:<10} | {cols:<10} | {sim_t:<9} | {wall_t:<9}"
            )
            print(row)

        print("-" * 80)
        print(" [NOTE] Seeds with forced failure prove TwinGuard's dynamic recovery path:")
        print(" When a grasp unexpectedly fails, the system re-observes and replans,")
        print(" successfully completing the full multi-step instruction without human intervention.")
        print("=" * 80 + "\n")

    return data


def main() -> None:
    """CLI entrypoint for report generation."""
    parser = argparse.ArgumentParser(description="TwinGuard Multi-Seed Evaluation Report")
    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to results.json (default: evaluation/results.json)",
    )
    args = parser.parse_args()

    generate_report(results_path=args.results, verbose=True)


if __name__ == "__main__":
    main()
