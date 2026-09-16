"""Closed-loop task execution pipeline for TwinGuard bimanual manipulation.

Orchestrates:
1. Perception: Ground-truth scene state extraction (perception/scene_state.py).
2. Planning: LLM task decomposition with schema validation (planning/planner.py).
3. Robotics: Primitive execution on dual SO-101 arms (robotics/primitives.py).
4. Safety & Verification: Physical verification and collision checks (safety/verifier.py).
5. Dynamic Recovery: Closed-loop re-observation and replanning upon step failure
   (retries up to 2 attempts per step before continuing).
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import mujoco

from simulation.simulator import TwinGuardSim
from perception.scene_state import format_scene_state, get_ground_truth_scene_state
from planning.planner import PlanStep, plan
from robotics.primitives import MotionPrimitives
from safety.verifier import SafetyVerifier, VerificationResult


# Configure module logger
logger = logging.getLogger("twinguard.executor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


WORKED_EXAMPLE_INSTRUCTION = (
    "Open the top drawer, pick up the plate with arm A, place it on the table, "
    "pick up the mug with arm B, pour water into the mug with arm A."
)


def resolve_step_target(sim: TwinGuardSim, step: PlanStep) -> Optional[List[float]]:
    """Resolve 3D target coordinates for a step ensuring kinematic reachability and clearance."""
    arm = step.arm
    action = step.action
    obj_name = step.object.lower()

    if obj_name == "table":
        # Destination on table reachable by the designated arm
        if arm == "left_arm":
            return [0.20, 0.10, 0.50]
        else:
            return [0.20, -0.10, 0.50]

    # If target was provided by step
    if step.target is not None:
        target = list(step.target)
        # Ensure z clearance above table for approach and transport
        if action in ("approach", "transport") and target[2] < 0.49:
            target[2] = 0.50
        # If transport/approach x coordinate exceeds reach:
        if action in ("approach", "transport") and target[0] > 0.28:
            target[0] = 0.24
        # If pouring, clamp to arm reach limits
        if action == "pour":
            if arm == "left_arm" and target[1] < -0.05:
                target[1] = -0.05
            elif arm == "right_arm" and target[1] > 0.05:
                target[1] = 0.05
        return target

    # Resolve from object geometry in scene
    if "drawer" in obj_name:
        handle_sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site")
        if handle_sid != -1:
            return [round(float(v), 3) for v in sim.data.site_xpos[handle_sid]]
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        if bid != -1:
            return [round(float(v), 3) for v in sim.data.xpos[bid]]

    # Freejoint objects: plate, mug
    for candidate in ("plate", "mug"):
        if candidate in obj_name:
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid != -1:
                pos = list(sim.data.xpos[bid])
                if action in ("approach", "transport"):
                    pos[2] = max(pos[2] + 0.04, 0.50)
                elif action == "pour":
                    pos[2] = max(pos[2] + 0.10, 0.52)
                    if arm == "left_arm" and pos[1] < -0.05:
                        pos[1] = -0.05
                    elif arm == "right_arm" and pos[1] > 0.05:
                        pos[1] = 0.05
                return [round(float(v), 3) for v in pos]

    return None


def execute_primitive_step(
    primitives: MotionPrimitives,
    sim: TwinGuardSim,
    step: PlanStep,
) -> bool:
    """Execute a single PlanStep using the matching deterministic motion primitive."""
    action = step.action
    arm = step.arm
    target = resolve_step_target(sim, step)

    if action == "approach":
        if target is None:
            target = resolve_step_target(sim, step) or [0.25, 0.0, 0.50]
        return bool(primitives.approach(arm, target))

    elif action == "grasp":
        return bool(primitives.grasp(arm))

    elif action == "lift":
        return bool(primitives.lift(arm, height=0.06))

    elif action == "transport":
        if target is None:
            target = resolve_step_target(sim, step) or [0.25, 0.0, 0.50]
        return bool(primitives.transport(arm, target))

    elif action == "release":
        return bool(primitives.release(arm))

    elif action == "open_drawer":
        return bool(primitives.open_drawer(arm, drawer_body=step.object))

    elif action == "pour":
        return bool(primitives.pour(arm, target_container=step.object))

    else:
        logger.error(f"Unrecognized action '{action}'")
        return False


def run_plan(
    instruction: str,
    sim: Optional[TwinGuardSim] = None,
    config_path: Optional[Union[str, Path]] = None,
    llm_caller: Optional[Callable] = None,
    max_recovery_attempts: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute natural language instruction through perception, planning, control, and safety verification.

    Args:
        instruction: High-level manipulation instruction.
        sim: Optional TwinGuardSim instance (created if not provided).
        config_path: Optional path to sim_config.yaml.
        llm_caller: Optional LLM callable for mocking or live model calls.
        max_recovery_attempts: Maximum recovery replanning attempts per failed step (default: 2).
        verbose: Whether to log execution details to console.

    Returns:
        Dict containing execution metrics, logs, and overall status.
    """
    owns_sim = sim is None
    if sim is None:
        sim = TwinGuardSim(config_path=str(config_path) if config_path else None)
        sim.reset()

    primitives = MotionPrimitives(sim, config_path=config_path)
    verifier = SafetyVerifier(sim)

    if verbose:
        logger.info(f"Starting execution for instruction: '{instruction}'")

    # 1. Perception: Read current scene state
    initial_scene_state = format_scene_state(sim)
    if verbose:
        logger.info("Perception extracted initial scene state.")

    # 2. Planning: Decompose instruction into validated primitive steps
    initial_steps: List[PlanStep] = plan(
        instruction,
        initial_scene_state,
        config_path=config_path,
        llm_caller=llm_caller,
    )

    if verbose:
        logger.info(f"Planner generated {len(initial_steps)} steps:")
        for idx, s in enumerate(initial_steps, 1):
            logger.info(f"  Step {idx}: {s.action} [{s.arm}] target={s.target} obj={s.object}")

    execution_log: List[Dict[str, Any]] = []
    overall_success = True

    # 3. Execution & Closed-Loop Verification Loop
    for step_idx, step in enumerate(initial_steps, 1):
        if verbose:
            logger.info(f"\n--- Executing Step {step_idx}/{len(initial_steps)}: {step.action} ({step.arm}) ---")

        # Execute the primitive action
        prim_target = resolve_step_target(sim, step)
        prim_ok = execute_primitive_step(primitives, sim, step)

        # Safety & physical verification check
        verification = verifier.verify_step(step, primitive_returned=prim_ok, target=prim_target)

        step_record: Dict[str, Any] = {
            "step_index": step_idx,
            "step": step.model_dump(),
            "initial_success": verification["success"],
            "initial_reason": verification["reason"],
            "recovered": False,
            "recovery_attempts": 0,
            "recovery_log": [],
            "final_success": verification["success"],
        }

        # 4. Dynamic Recovery on Failure
        if not verification["success"]:
            logger.warning(
                f"Step {step_idx} [{step.action}] failed verification: {verification['reason']}. "
                f"Initiating dynamic recovery (max {max_recovery_attempts} attempts)..."
            )

            step_recovered = False
            for attempt in range(1, max_recovery_attempts + 1):
                step_record["recovery_attempts"] = attempt

                # Re-observe scene via perception
                updated_scene_state = format_scene_state(sim)

                # Formulate recovery instruction with failure context
                recovery_instruction = (
                    f"Recovery Attempt {attempt}: Step '{step.action}' failed because: {verification['reason']}. "
                    f"Perform a recovery action to continue achieving: {instruction}"
                )

                try:
                    recovery_steps = plan(
                        recovery_instruction,
                        updated_scene_state,
                        config_path=config_path,
                        llm_caller=llm_caller,
                    )
                except Exception as plan_err:
                    logger.error(f"Recovery planning failed on attempt {attempt}: {plan_err}")
                    step_record["recovery_log"].append({
                        "attempt": attempt,
                        "success": False,
                        "reason": f"Recovery planning error: {plan_err}",
                    })
                    continue

                # Execute recovery step(s)
                rec_all_ok = True
                for rec_step in recovery_steps:
                    if verbose:
                        logger.info(f"  Executing recovery action: {rec_step.action} ({rec_step.arm})")
                    rec_target = resolve_step_target(sim, rec_step)
                    r_ok = execute_primitive_step(primitives, sim, rec_step)
                    r_ver = verifier.verify_step(rec_step, primitive_returned=r_ok, target=rec_target)
                    if not r_ver["success"]:
                        rec_all_ok = False
                        verification = r_ver
                        logger.warning(f"  Recovery action '{rec_step.action}' failed: {r_ver['reason']}")
                        break

                step_record["recovery_log"].append({
                    "attempt": attempt,
                    "success": rec_all_ok,
                    "reason": "Recovery verified successfully" if rec_all_ok else verification["reason"],
                })

                if rec_all_ok:
                    logger.info(f"Recovery succeeded on attempt {attempt}!")
                    step_recovered = True
                    break

            step_record["recovered"] = step_recovered
            step_record["final_success"] = step_recovered

            if not step_recovered:
                logger.error(
                    f"Step {step_idx} [{step.action}] failed after {max_recovery_attempts} recovery attempts. "
                    "Logging failure and continuing with remaining pipeline."
                )
                overall_success = False
        else:
            if verbose:
                logger.info(f"Step {step_idx} [{step.action}] SUCCESS: {verification['reason']}")

        execution_log.append(step_record)

    results = {
        "instruction": instruction,
        "success": overall_success,
        "total_steps": len(initial_steps),
        "steps_succeeded": sum(1 for r in execution_log if r["final_success"]),
        "steps_failed": sum(1 for r in execution_log if not r["final_success"]),
        "sim_time": round(sim.get_time(), 3),
        "execution_log": execution_log,
    }

    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info(f"EXECUTION SUMMARY: Overall {'SUCCESS' if overall_success else 'FAILED'}")
        logger.info(f"Steps: {results['steps_succeeded']}/{results['total_steps']} succeeded")
        logger.info(f"Simulation Time: {results['sim_time']}s")
        logger.info("=" * 60)

    if owns_sim:
        sim.close()

    return results


def main() -> None:
    """CLI entrypoint to execute the worked example or custom instruction."""
    parser = argparse.ArgumentParser(description="TwinGuard End-to-End Pipeline Executor")
    parser.add_argument(
        "--instruction",
        type=str,
        default=WORKED_EXAMPLE_INSTRUCTION,
        help="Natural language instruction to execute (defaults to brief's worked example)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.yaml",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max recovery replanning attempts per failed step (default: 2)",
    )
    args = parser.parse_args()

    results = run_plan(
        instruction=args.instruction,
        config_path=args.config,
        max_recovery_attempts=args.max_retries,
        verbose=True,
    )

    print("\nSTEP BY STEP EXECUTION STATUS:")
    for item in results["execution_log"]:
        st = item["step"]
        status = "PASSED" if item["final_success"] else "FAILED"
        print(f"  Step {item['step_index']:02d} [{status}] {st['action']:<12} {st['arm']:<10} obj={st['object']}")
        if not item["final_success"]:
            print(f"      Failure reason: {item['initial_reason']}")

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
