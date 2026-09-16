"""Multi-seed evaluation runner for TwinGuard bimanual manipulation.

Evaluates 10 randomized scene configurations across seeds 0..9:
1. Randomizes plate, mug, and drawer starting positions per seed.
2. Executes closed-loop task manipulation via executor.run_plan().
3. Injects controlled failures on designated seeds (default: seeds 2 and 7)
   to prove TwinGuard's closed-loop dynamic recovery path succeeding.
4. Records per-seed step outcomes, completion times, recovery attempts,
   and collision counts to evaluation/results.json.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import mujoco
import yaml

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.simulator import TwinGuardSim
from perception.scene_state import format_scene_state
from robotics.primitives import MotionPrimitives
from safety.verifier import SafetyVerifier
from evaluation.executor import (
    WORKED_EXAMPLE_INSTRUCTION,
    run_plan,
)

# Configure logger
logger = logging.getLogger("twinguard.runner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sim_config.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results.json"


def load_runner_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load runner parameters from configs/sim_config.yaml."""
    cfg_p = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            return full_cfg.get("evaluation", {}).get("runner", {})
    return {}


def randomize_scene_for_seed(
    sim: TwinGuardSim,
    seed: int,
    plate_perturbation: float = 0.015,
    mug_perturbation: float = 0.015,
    drawer_perturbation: float = 0.008,
) -> Dict[str, List[float]]:
    """Randomize plate, mug, and drawer starting positions deterministically for the given seed.

    Args:
        sim: Active TwinGuardSim simulation instance.
        seed: Integer seed for reproducible pseudo-random generation.
        plate_perturbation: Maximum +/- coordinate displacement for plate (meters).
        mug_perturbation: Maximum +/- coordinate displacement for mug (meters).
        drawer_perturbation: Maximum +/- coordinate displacement for drawer chest (meters).

    Returns:
        Dict mapping object names to their randomized starting [x, y, z] coordinates.
    """
    rng = np.random.default_rng(seed)

    # 1. Randomize Plate (Freejoint)
    p_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    if p_jid != -1:
        p_adr = sim.model.jnt_qposadr[p_jid]
        px = float(0.25 + rng.uniform(-plate_perturbation, plate_perturbation))
        py = float(0.00 + rng.uniform(-plate_perturbation, plate_perturbation))
        pz = 0.435  # resting on table top
        sim.data.qpos[p_adr : p_adr + 3] = [px, py, pz]

    # 2. Randomize Mug (Freejoint)
    m_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    if m_jid != -1:
        m_adr = sim.model.jnt_qposadr[m_jid]
        mx = float(0.22 + rng.uniform(-mug_perturbation, mug_perturbation))
        my = float(-0.16 + rng.uniform(-mug_perturbation, mug_perturbation))
        mz = 0.465  # upright on table top
        sim.data.qpos[m_adr : m_adr + 3] = [mx, my, mz]

    # 3. Randomize Drawer Chest and slide joint
    d_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "drawer_chest")
    if d_bid != -1:
        dx = float(0.26 + rng.uniform(-drawer_perturbation, drawer_perturbation))
        dy = float(0.22 + rng.uniform(-drawer_perturbation, drawer_perturbation))
        dz = 0.46
        sim.model.body_pos[d_bid] = [dx, dy, dz]

    d_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint")
    if d_jid != -1:
        d_adr = sim.model.jnt_qposadr[d_jid]
        # Initially closed with tiny micro-variation
        sim.data.qpos[d_adr] = float(rng.uniform(0.0, 0.003))

    mujoco.mj_forward(sim.model, sim.data)

    # Read resolved start coordinates
    start_positions = {
        "plate": [round(float(v), 3) for v in sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")]],
        "mug": [round(float(v), 3) for v in sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mug")]],
        "drawer": [round(float(v), 3) for v in sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")]],
    }

    return start_positions


def create_perception_informed_planner(sim: TwinGuardSim) -> Callable[[str, Optional[str]], str]:
    """Create a mock LLM planner anchored to observed object poses in the active simulation."""
    def planner_callable(prompt: str, model_name: Optional[str] = None) -> str:
        # Extract task instruction section to isolate from scene description
        if "TASK INSTRUCTION:" in prompt:
            task_text = prompt.split("TASK INSTRUCTION:")[-1]
        else:
            task_text = prompt

        p_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        m_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mug")

        p_pos = [round(float(v), 3) for v in sim.data.xpos[p_bid]] if p_bid != -1 else [0.25, 0.0, 0.435]
        m_pos = [round(float(v), 3) for v in sim.data.xpos[m_bid]] if m_bid != -1 else [0.22, -0.16, 0.465]

        # Check if this is a dynamic recovery replanning prompt
        if "Recovery Attempt" in task_text:
            if "failed because:" in task_text:
                reason = task_text.split("failed because:")[1].split("Perform a recovery action")[0].lower()
            else:
                reason = task_text.lower()

            if "mug" in reason:
                return json.dumps([
                    {"action": "approach", "arm": "right_arm", "object": "mug", "target": [m_pos[0], m_pos[1], 0.50]},
                    {"action": "grasp", "arm": "right_arm", "object": "mug", "target": None},
                ])
            elif "plate" in reason:
                return json.dumps([
                    {"action": "approach", "arm": "left_arm", "object": "plate", "target": [p_pos[0], p_pos[1], 0.50]},
                    {"action": "grasp", "arm": "left_arm", "object": "plate", "target": None},
                ])
            elif "drawer" in reason:
                return json.dumps([
                    {"action": "open_drawer", "arm": "left_arm", "object": "drawer", "target": None}
                ])
            # Default recovery action
            return json.dumps([
                {"action": "grasp", "arm": "left_arm", "object": "plate", "target": None}
            ])

        # Generate 10-step bimanual manipulation sequence
        plan = [
            {"action": "open_drawer", "arm": "left_arm", "object": "drawer", "target": None},
            {"action": "approach", "arm": "left_arm", "object": "plate", "target": [p_pos[0], p_pos[1], 0.50]},
            {"action": "grasp", "arm": "left_arm", "object": "plate", "target": None},
            {"action": "lift", "arm": "left_arm", "object": "plate", "target": None},
            {"action": "transport", "arm": "left_arm", "object": "table", "target": [0.20, 0.10, 0.55]},
            {"action": "release", "arm": "left_arm", "object": "plate", "target": None},
            {"action": "approach", "arm": "right_arm", "object": "mug", "target": [m_pos[0], m_pos[1], 0.50]},
            {"action": "grasp", "arm": "right_arm", "object": "mug", "target": None},
            {"action": "lift", "arm": "right_arm", "object": "mug", "target": None},
            {"action": "pour", "arm": "right_arm", "object": "plate", "target": [p_pos[0], p_pos[1], 0.435]},
        ]
        return json.dumps(plan)

    return planner_callable


class ControlledFailureInterceptor:
    """Manages injection of a single controlled grasp failure on designated seeds."""

    def __init__(self, target_seed: int, target_arm: str, target_object: str):
        self.target_seed = target_seed
        self.target_arm = target_arm
        self.target_object = target_object
        self.failure_injected = False
        self.orig_grasp = MotionPrimitives.grasp

    def __enter__(self):
        interceptor = self

        def controlled_grasp(primitives_inst: MotionPrimitives, arm: str, *args, **kwargs) -> bool:
            # Check if condition for forced failure is met (only on target arm, once)
            drawer_jid = mujoco.mj_name2id(primitives_inst.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint")
            drawer_open = True
            if drawer_jid != -1:
                qpos_adr = primitives_inst.sim.model.jnt_qposadr[drawer_jid]
                drawer_open = float(primitives_inst.sim.data.qpos[qpos_adr]) > 0.02

            if not interceptor.failure_injected and drawer_open and arm == interceptor.target_arm:
                interceptor.failure_injected = True
                print("\n" + "#" * 70)
                print(f"[CONTROLLED_FAILURE] >>> Seed {interceptor.target_seed}: Forcing '{arm}' grasp on "
                      f"'{interceptor.target_object}' to FAIL ONCE!")
                print(f"[CONTROLLED_FAILURE] >>> Demonstrating TwinGuard closed-loop dynamic recovery path...")
                print("#" * 70 + "\n")
                return False

            return interceptor.orig_grasp(primitives_inst, arm, *args, **kwargs)

        MotionPrimitives.grasp = controlled_grasp
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        MotionPrimitives.grasp = self.orig_grasp


def run_evaluation(
    num_seeds: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    forced_failure_seeds: Optional[Sequence[int]] = None,
    config_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete multi-seed evaluation and save results.json.

    Args:
        num_seeds: Number of seeds to evaluate (default from config or 10).
        output_path: Filepath where results.json is saved.
        forced_failure_seeds: Seeds where controlled failures are injected (default: [2, 7]).
        config_path: Optional path to sim_config.yaml.
        verbose: Whether to log execution progress to stdout.

    Returns:
        Dict containing full summary and per-seed results.
    """
    cfg = load_runner_config(config_path)

    n_seeds = int(num_seeds or cfg.get("num_seeds", 10))
    out_file = Path(output_path or cfg.get("output_path", DEFAULT_OUTPUT_PATH))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    forced_seeds = list(forced_failure_seeds if forced_failure_seeds is not None else cfg.get("forced_failure_seeds", [2, 7]))
    plate_pert = float(cfg.get("plate_xy_perturbation", 0.015))
    mug_pert = float(cfg.get("mug_xy_perturbation", 0.015))
    drawer_pert = float(cfg.get("drawer_xy_perturbation", 0.008))
    instruction = str(cfg.get("instruction", WORKED_EXAMPLE_INSTRUCTION))

    if verbose:
        print("=" * 70)
        print(" TWINGUARD MULTI-SEED EVALUATION RUNNER")
        print("=" * 70)
        print(f" Seeds:               0 .. {n_seeds - 1} (Total: {n_seeds})")
        print(f" Output Path:         {out_file}")
        print(f" Controlled Failures: Seeds {forced_seeds}")
        print(f" Task Instruction:    '{instruction}'")
        print("=" * 70 + "\n")

    seed_records: List[Dict[str, Any]] = []

    for seed in range(n_seeds):
        if verbose:
            print(f"\n{'=' * 30} RUNNING SEED {seed}/{n_seeds - 1} {'=' * 30}")

        wall_t0 = time.perf_counter()

        sim = TwinGuardSim(config_path=str(config_path) if config_path else None)
        sim.reset()

        # Deterministic seed randomization
        start_poses = randomize_scene_for_seed(
            sim,
            seed=seed,
            plate_perturbation=plate_pert,
            mug_perturbation=mug_pert,
            drawer_perturbation=drawer_pert,
        )

        if verbose:
            print(f"[INFO] Seed {seed} randomized start poses:")
            print(f"  - Plate:  {start_poses['plate']}")
            print(f"  - Mug:    {start_poses['mug']}")
            print(f"  - Drawer: {start_poses['drawer']}")

        planner_fn = create_perception_informed_planner(sim)
        has_forced_failure = seed in forced_seeds

        # Configure failure parameters (Seed 2 tests left_arm plate grasp; Seed 7 tests right_arm mug grasp)
        target_arm = "left_arm" if seed % 2 == 0 else "right_arm"
        target_obj = "plate" if target_arm == "left_arm" else "mug"

        if has_forced_failure:
            with ControlledFailureInterceptor(target_seed=seed, target_arm=target_arm, target_object=target_obj):
                res = run_plan(
                    instruction=instruction,
                    sim=sim,
                    config_path=config_path,
                    llm_caller=planner_fn,
                    verbose=verbose,
                )
        else:
            res = run_plan(
                instruction=instruction,
                sim=sim,
                config_path=config_path,
                llm_caller=planner_fn,
                verbose=verbose,
            )

        wall_t1 = time.perf_counter()
        wall_time_sec = round(wall_t1 - wall_t0, 3)

        # Count collisions and recoveries
        verifier = SafetyVerifier(sim)
        col_res = verifier.check_unexpected_collisions()
        collisions = 0 if col_res.success else 1

        exec_log = res.get("execution_log", [])
        recovery_attempts = sum(r.get("recovery_attempts", 0) for r in exec_log)
        steps_recovered = sum(1 for r in exec_log if r.get("recovered", False))

        # Flatten step log for concise reporting
        clean_steps = []
        for step_rec in exec_log:
            step_info = step_rec.get("step", {})
            clean_steps.append({
                "step_index": step_rec.get("step_index"),
                "action": step_info.get("action"),
                "arm": step_info.get("arm"),
                "object": step_info.get("object"),
                "initial_success": step_rec.get("initial_success"),
                "initial_reason": step_rec.get("initial_reason"),
                "recovered": step_rec.get("recovered", False),
                "recovery_attempts": step_rec.get("recovery_attempts", 0),
                "final_success": step_rec.get("final_success"),
            })

        seed_entry = {
            "seed": seed,
            "instruction": instruction,
            "success": res.get("success", False),
            "total_steps": res.get("total_steps", 0),
            "steps_succeeded": res.get("steps_succeeded", 0),
            "steps_failed": res.get("steps_failed", 0),
            "sim_time": res.get("sim_time", 0.0),
            "wall_time": wall_time_sec,
            "forced_failure": has_forced_failure,
            "recovery_attempts": recovery_attempts,
            "steps_recovered": steps_recovered,
            "collisions": collisions,
            "start_positions": start_poses,
            "steps": clean_steps,
        }
        seed_records.append(seed_entry)

        sim.close()

    # Aggregate evaluation metrics
    total_seeds = len(seed_records)
    successful_seeds = sum(1 for s in seed_records if s["success"])
    task_success_rate = round((successful_seeds / total_seeds) * 100.0, 1) if total_seeds > 0 else 0.0

    # Calculate Grasp metrics
    total_grasp_attempts = 0
    initial_grasp_successes = 0
    final_grasp_successes = 0
    for s in seed_records:
        for st in s["steps"]:
            if st["action"] == "grasp":
                total_grasp_attempts += 1
                if st["initial_success"]:
                    initial_grasp_successes += 1
                if st["final_success"]:
                    final_grasp_successes += 1

    initial_grasp_rate = round((initial_grasp_successes / total_grasp_attempts) * 100.0, 1) if total_grasp_attempts > 0 else 0.0
    final_grasp_rate = round((final_grasp_successes / total_grasp_attempts) * 100.0, 1) if total_grasp_attempts > 0 else 0.0

    # Recovery metrics
    total_failed_steps = sum(1 for s in seed_records for st in s["steps"] if not st["initial_success"])
    total_recovered_steps = sum(1 for s in seed_records for st in s["steps"] if st["recovered"])
    recovery_rate = round((total_recovered_steps / total_failed_steps) * 100.0, 1) if total_failed_steps > 0 else 100.0

    total_collisions = sum(s["collisions"] for s in seed_records)
    avg_sim_time = round(float(np.mean([s["sim_time"] for s in seed_records])), 3) if seed_records else 0.0
    avg_wall_time = round(float(np.mean([s["wall_time"] for s in seed_records])), 3) if seed_records else 0.0

    results_data = {
        "summary": {
            "total_seeds": total_seeds,
            "successful_seeds": successful_seeds,
            "task_success_rate": task_success_rate,
            "total_grasps": total_grasp_attempts,
            "initial_grasp_success_rate": initial_grasp_rate,
            "final_grasp_success_rate": final_grasp_rate,
            "total_failures_encountered": total_failed_steps,
            "total_failures_recovered": total_recovered_steps,
            "recovery_success_rate": recovery_rate,
            "total_collisions": total_collisions,
            "avg_sim_time_s": avg_sim_time,
            "avg_wall_time_s": avg_wall_time,
        },
        "seeds": seed_records,
    }

    # Save to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print(f"[SUCCESS] Evaluation complete! Results saved to: {out_file}")
        print(f"Task Success Rate:    {task_success_rate}% ({successful_seeds}/{total_seeds})")
        print(f"Initial Grasp Rate:   {initial_grasp_rate}% ({initial_grasp_successes}/{total_grasp_attempts})")
        print(f"Post-Recovery Grasp:  {final_grasp_rate}% ({final_grasp_successes}/{total_grasp_attempts})")
        print(f"Recovery Success:     {recovery_rate}% ({total_recovered_steps}/{total_failed_steps})")
        print(f"Total Collisions:     {total_collisions}")
        print(f"Average Sim Time:     {avg_sim_time}s (Wall: {avg_wall_time}s)")
        print("=" * 70 + "\n")

    return results_data


def main() -> None:
    """CLI entrypoint for running evaluation across seeds."""
    parser = argparse.ArgumentParser(description="TwinGuard Multi-Seed Evaluation Runner")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to evaluate (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output results.json (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.yaml",
    )
    args = parser.parse_args()

    run_evaluation(
        num_seeds=args.seeds,
        output_path=args.output,
        config_path=args.config,
        verbose=True,
    )


if __name__ == "__main__":
    main()
