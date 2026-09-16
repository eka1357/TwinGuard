"""Automated integration tests for the TwinGuard task execution and safety pipeline.

Verifies:
- End-to-end execution of the challenge brief's worked example instruction.
- Perception -> Planning -> Control -> Safety Verification pipeline integration.
- Dynamic recovery when a primitive is forced to fail (monkeypatched failure):
  confirms re-observation of scene state, failure reason propagation, and recovery replanning.
- Retry exhaustion behavior (max 2 recovery attempts before logging failure and continuing).
- SafetyVerifier physical post-condition, collision, and workspace boundary checks.
"""

import json
from typing import Any, Dict, List
import pytest
import mujoco

from simulation.simulator import TwinGuardSim
from robotics.primitives import MotionPrimitives
from safety.verifier import SafetyVerifier, VerificationResult
from evaluation.executor import (
    WORKED_EXAMPLE_INSTRUCTION,
    execute_primitive_step,
    resolve_step_target,
    run_plan,
)
from planning.planner import PlanStep


@pytest.fixture
def sim():
    """Fixture providing a fresh TwinGuardSim instance."""
    sim_inst = TwinGuardSim()
    sim_inst.reset()
    yield sim_inst
    sim_inst.close()


def fixed_worked_example_mock_llm(prompt: str, model_name: str) -> str:
    """Mock LLM returning a valid, kinematically achievable plan for the brief's worked example."""
    # Worked example plan tuned to reachable dual-arm coordinate zones
    plan = [
        {"action": "open_drawer", "arm": "left_arm", "object": "drawer", "target": None},
        {"action": "approach", "arm": "left_arm", "object": "plate", "target": [0.25, 0.0, 0.50]},
        {"action": "grasp", "arm": "left_arm", "object": "plate", "target": None},
        {"action": "lift", "arm": "left_arm", "object": "plate", "target": None},
        {"action": "transport", "arm": "left_arm", "object": "table", "target": [0.20, 0.10, 0.55]},
        {"action": "release", "arm": "left_arm", "object": "plate", "target": None},
        {"action": "approach", "arm": "right_arm", "object": "mug", "target": [0.22, -0.16, 0.50]},
        {"action": "grasp", "arm": "right_arm", "object": "mug", "target": None},
        {"action": "lift", "arm": "right_arm", "object": "mug", "target": None},
        {"action": "pour", "arm": "right_arm", "object": "plate", "target": [0.25, 0.0, 0.435]},
    ]
    return json.dumps(plan)


# ------------------------------------------------------------------------------
# 1. Full Pipeline on Worked Example
# ------------------------------------------------------------------------------

def test_executor_worked_example_full_pipeline(sim):
    """Verify full end-to-end execution of the brief's worked example."""
    results = run_plan(
        instruction=WORKED_EXAMPLE_INSTRUCTION,
        sim=sim,
        llm_caller=fixed_worked_example_mock_llm,
        verbose=False,
    )

    assert results["success"] is True, "Worked example execution failed"
    assert results["total_steps"] == 10
    assert results["steps_succeeded"] == 10
    assert results["steps_failed"] == 0
    assert results["sim_time"] > 0.0

    # Verify each step's record in execution log
    for item in results["execution_log"]:
        assert item["final_success"] is True
        assert len(item["initial_reason"]) > 0


# ------------------------------------------------------------------------------
# 2. Dynamic Failure and Recovery Path Test
# ------------------------------------------------------------------------------

def test_executor_recovery_on_primitive_failure(sim, monkeypatch):
    """Confirm the recovery path retries and logs correctly when a primitive fails."""
    call_counts = {"grasp": 0, "planner_recovery": 0}
    recorded_recovery_prompts: List[str] = []

    orig_grasp = MotionPrimitives.grasp

    def failing_first_grasp(self, arm, *args, **kwargs):
        call_counts["grasp"] += 1
        if call_counts["grasp"] == 1:
            # Force primitive execution to fail on first attempt
            return False
        return orig_grasp(self, arm, *args, **kwargs)

    monkeypatch.setattr(MotionPrimitives, "grasp", failing_first_grasp)

    def recovery_mock_llm(prompt: str, model_name: str) -> str:
        if "Recovery Attempt" in prompt:
            call_counts["planner_recovery"] += 1
            recorded_recovery_prompts.append(prompt)
            # Return recovery step to retry grasp
            return json.dumps([{"action": "grasp", "arm": "left_arm", "object": "plate", "target": None}])

        # Initial plan: approach and grasp plate
        return json.dumps([
            {"action": "approach", "arm": "left_arm", "object": "plate", "target": [0.25, 0.0, 0.50]},
            {"action": "grasp", "arm": "left_arm", "object": "plate", "target": None},
        ])

    instruction = "Approach and grasp the plate."
    results = run_plan(
        instruction=instruction,
        sim=sim,
        llm_caller=recovery_mock_llm,
        max_recovery_attempts=2,
        verbose=False,
    )

    # Grasp was called twice (initial failing call + recovery call)
    assert call_counts["grasp"] == 2
    assert call_counts["planner_recovery"] == 1
    assert len(recorded_recovery_prompts) == 1

    # Verify recovery prompt contained failure reason
    rec_prompt = recorded_recovery_prompts[0]
    assert "Recovery Attempt 1" in rec_prompt
    assert "failed because:" in rec_prompt

    # Overall execution must succeed after recovery
    assert results["success"] is True
    assert results["steps_succeeded"] == 2

    # Verify step 2 log shows initial failure followed by successful recovery
    step2_log = results["execution_log"][1]
    assert step2_log["initial_success"] is False
    assert step2_log["recovered"] is True
    assert step2_log["recovery_attempts"] == 1
    assert step2_log["final_success"] is True
    assert len(step2_log["recovery_log"]) == 1
    assert step2_log["recovery_log"][0]["success"] is True


# ------------------------------------------------------------------------------
# 3. Maximum Retries Exhaustion
# ------------------------------------------------------------------------------

def test_executor_max_retries_exhaustion(sim, monkeypatch):
    """Verify max 2 attempts are made when primitive persistently fails, then continues."""
    call_counts = {"lift": 0}

    def always_failing_lift(self, arm, *args, **kwargs):
        call_counts["lift"] += 1
        return False

    monkeypatch.setattr(MotionPrimitives, "lift", always_failing_lift)

    def mock_llm(prompt: str, model_name: str) -> str:
        if "Recovery Attempt" in prompt:
            # Recovery tries lifting again
            return json.dumps([{"action": "lift", "arm": "left_arm", "object": "plate", "target": None}])

        return json.dumps([
            {"action": "lift", "arm": "left_arm", "object": "plate", "target": None},
            {"action": "release", "arm": "left_arm", "object": "plate", "target": None},
        ])

    results = run_plan(
        instruction="Lift and release the plate.",
        sim=sim,
        llm_caller=mock_llm,
        max_recovery_attempts=2,
        verbose=False,
    )

    # 1 initial call + 2 recovery attempts = 3 total lift calls
    assert call_counts["lift"] == 3

    # Step 1 should be logged as failed after exhausting retries
    step1_log = results["execution_log"][0]
    assert step1_log["initial_success"] is False
    assert step1_log["recovered"] is False
    assert step1_log["recovery_attempts"] == 2
    assert step1_log["final_success"] is False

    # Pipeline continued to Step 2 (release)
    assert results["total_steps"] == 2
    assert results["execution_log"][1]["step"]["action"] == "release"
    assert results["success"] is False


# ------------------------------------------------------------------------------
# 4. Safety Verifier Unit Checks
# ------------------------------------------------------------------------------

def test_safety_verifier_detects_table_drop(sim):
    """Verify SafetyVerifier catches objects falling below table height (z < 0.40)."""
    verifier = SafetyVerifier(sim)
    assert verifier.check_global_safety().success is True

    # Artificially drop plate below table height
    plate_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    sim.data.xpos[plate_bid][2] = 0.35

    check = verifier.check_global_safety()
    assert check.success is False
    assert "fell below table height" in check.reason


def test_safety_verifier_detects_unexpected_arm_collision(sim):
    """Verify SafetyVerifier catches arm-arm collisions."""
    verifier = SafetyVerifier(sim)

    # Simulate contact between left and right arm
    sim.data.ncon = 1
    sim.data.contact[0].geom1 = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "left_arm_forearm_link")
    sim.data.contact[0].geom2 = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "right_arm_forearm_link")

    check = verifier.check_unexpected_collisions()
    assert check.success is False
    assert "Unexpected arm collision" in check.reason


def test_safety_verifier_grasp_zone_validation(sim):
    """Verify SafetyVerifier rejects grasp when gripper is far from target object."""
    verifier = SafetyVerifier(sim)

    # Gripper is at home position (~[0, 0.22, 0.925]), far from plate ([0.25, 0, 0.435])
    step = PlanStep(action="grasp", arm="left_arm", object="plate")
    check = verifier.verify_step(step, primitive_returned=True)

    assert check["success"] is False
    assert "not within gripper grasp zone" in check["reason"]
