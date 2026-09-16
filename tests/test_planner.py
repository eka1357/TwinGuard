"""Automated test suite for the TwinGuard LLM-based task planner.

Verifies:
- Production of schema-valid plans for 3 distinct natural-language instructions
  including the challenge brief's worked example.
- Every planned action corresponds to an actual motion primitive implemented in robotics/primitives.py.
- Target arms adhere to the dual-arm topology ('left_arm' or 'right_arm').
- Valid 3D coordinate validation for step targets.
- Single-retry recovery mechanism when LLM output initially violates schema.
- Exception handling (PlanValidationError) when output fails validation after retry.
- Model name configuration loading from configs/sim_config.yaml (no hardcoded model strings).
- End-to-end integration with perception/scene_state.py and TwinGuardSim.
"""

import json
from pathlib import Path
from typing import List
import pytest
import yaml

from simulation.simulator import TwinGuardSim
from perception.scene_state import format_scene_state, get_ground_truth_scene_state
from planning.planner import (
    ActionType,
    ArmType,
    LLMPlanner,
    PlanResponse,
    PlanStep,
    PlanValidationError,
    plan,
)
import robotics.primitives as prim_module
from robotics.primitives import MotionPrimitives


# Ground truth primitives that physically exist in the robotics module
EXISTING_PRIMITIVES = {
    "approach",
    "grasp",
    "lift",
    "transport",
    "release",
    "open_drawer",
    "pour",
}


@pytest.fixture
def sim():
    """Fixture providing a fresh TwinGuardSim instance."""
    sim_inst = TwinGuardSim()
    sim_inst.reset()
    yield sim_inst
    sim_inst.close()


@pytest.fixture
def scene_state_text(sim):
    """Fixture providing compact ground-truth scene state text."""
    return format_scene_state(sim)


def assert_valid_plan_step(step: PlanStep) -> None:
    """Helper asserting an individual PlanStep conforms to schema and physical primitives."""
    # 1. Action must exist in robotics.primitives
    assert step.action in EXISTING_PRIMITIVES, f"Action '{step.action}' is not in known primitives"
    assert hasattr(MotionPrimitives, step.action), f"MotionPrimitives has no method '{step.action}'"
    assert hasattr(prim_module, step.action), f"robotics.primitives has no function '{step.action}'"

    # 2. Arm must be left_arm or right_arm
    assert step.arm in ("left_arm", "right_arm"), f"Invalid arm '{step.arm}'"

    # 3. Object must be non-empty string
    assert isinstance(step.object, str) and len(step.object.strip()) > 0

    # 4. Target, if specified, must be 3D coordinate [x, y, z]
    if step.target is not None:
        assert isinstance(step.target, list), f"Expected list target, got {type(step.target)}"
        assert len(step.target) == 3, f"Expected 3D coordinate, got length {len(step.target)}"
        assert all(isinstance(coord, (int, float)) for coord in step.target)


# ------------------------------------------------------------------------------
# 1. Three Target Instructions (including Brief Worked Example)
# ------------------------------------------------------------------------------

def test_planner_brief_worked_example(scene_state_text):
    """Test 1: Brief worked example.

    'Open the top drawer, pick up the plate with arm A, place it on the table,
    pick up the mug with arm B, pour water into the mug with arm A.'
    """
    instruction = (
        "Open the top drawer, pick up the plate with arm A, place it on the table, "
        "pick up the mug with arm B, pour water into the mug with arm A."
    )

    steps: List[PlanStep] = plan(instruction, scene_state_text)

    assert isinstance(steps, list)
    assert len(steps) >= 5, f"Expected multi-step plan, got {len(steps)} steps"

    for step in steps:
        assert_valid_plan_step(step)

    # Validate high-level logical requirements from the brief
    actions = [s.action for s in steps]
    assert "open_drawer" in actions, "Plan must include opening the drawer"
    assert "grasp" in actions, "Plan must include grasping objects"
    assert "pour" in actions, "Plan must include pour action"

    # Verify arm mapping: arm A maps to left_arm, arm B maps to right_arm
    # Initial drawer and plate manipulation should use left_arm (arm A)
    drawer_steps = [s for s in steps if s.action == "open_drawer"]
    assert all(s.arm == "left_arm" for s in drawer_steps), "Drawer should be opened by arm A (left_arm)"

    # Mug pick up should use right_arm (arm B)
    mug_grasp_steps = [s for s in steps if s.action == "grasp" and s.object == "mug"]
    assert all(s.arm == "right_arm" for s in mug_grasp_steps), "Mug should be grasped by arm B (right_arm)"


def test_planner_plate_pick_and_place(scene_state_text):
    """Test 2: Plate pick and place instruction.

    'Pick up the plate with left_arm and place it on the table.'
    """
    instruction = "Pick up the plate with left_arm and place it on the table."

    steps: List[PlanStep] = plan(instruction, scene_state_text)

    assert isinstance(steps, list)
    assert len(steps) >= 3

    for step in steps:
        assert_valid_plan_step(step)

    # Verify key steps in pick-and-place sequence
    actions = [s.action for s in steps]
    assert "approach" in actions
    assert "grasp" in actions
    assert "release" in actions

    # All steps should operate with left_arm as requested
    assert all(s.arm == "left_arm" for s in steps)


def test_planner_drawer_mug_and_pour(scene_state_text):
    """Test 3: Complex multi-step coordination instruction.

    'Open the drawer with left_arm, pick up the mug with right_arm, and pour into the plate with left_arm.'
    """
    instruction = (
        "Open the drawer with left_arm, pick up the mug with right_arm, and pour into the plate with left_arm."
    )

    steps: List[PlanStep] = plan(instruction, scene_state_text)

    assert isinstance(steps, list)
    assert len(steps) >= 3

    for step in steps:
        assert_valid_plan_step(step)

    actions = [s.action for s in steps]
    assert "open_drawer" in actions
    assert "pour" in actions


# ------------------------------------------------------------------------------
# 2. Schema Validation and Retry Recovery Tests
# ------------------------------------------------------------------------------

def test_planner_retries_on_schema_failure_and_recovers(scene_state_text):
    """Verify planner retries when LLM outputs invalid schema on first attempt and recovers on second."""
    call_count = 0
    recorded_prompts: List[str] = []

    def mock_flaky_llm(prompt: str, model_name: str) -> str:
        nonlocal call_count
        call_count += 1
        recorded_prompts.append(prompt)

        if call_count == 1:
            # First attempt: Return invalid schema (unsupported action and bad arm name)
            invalid_plan = [
                {"action": "teleport_to_moon", "arm": "arm_3", "object": "plate", "target": [1, 2]}
            ]
            return json.dumps(invalid_plan)
        else:
            # Second attempt: Correct schema output
            valid_plan = [
                {"action": "approach", "arm": "left_arm", "object": "plate", "target": [0.25, 0.0, 0.435]},
                {"action": "grasp", "arm": "left_arm", "object": "plate", "target": None},
            ]
            return json.dumps(valid_plan)

    planner = LLMPlanner(llm_caller=mock_flaky_llm)
    steps = planner.plan("Pick up the plate", scene_state_text)

    # Must have called LLM twice (initial attempt + retry)
    assert call_count == 2
    assert len(steps) == 2
    assert steps[0].action == "approach"
    assert steps[1].action == "grasp"

    # Verify that the retry prompt contained feedback about the validation error
    retry_prompt = recorded_prompts[1]
    assert "VALIDATION ERROR ON PREVIOUS ATTEMPT" in retry_prompt
    assert "teleport_to_moon" in retry_prompt or "arm_3" in retry_prompt or "target" in retry_prompt


def test_planner_raises_clear_error_if_retry_fails(scene_state_text):
    """Verify PlanValidationError is raised if the LLM output remains invalid after retry."""
    call_count = 0

    def mock_broken_llm(prompt: str, model_name: str) -> str:
        nonlocal call_count
        call_count += 1
        # Consistently return non-JSON garbage
        return "I am an LLM and I refuse to speak in JSON!"

    planner = LLMPlanner(llm_caller=mock_broken_llm)

    with pytest.raises(PlanValidationError) as exc_info:
        planner.plan("Pick up the plate", scene_state_text)

    # Verify it attempted retry before raising
    assert call_count == 2
    assert "schema validation after" in str(exc_info.value).lower()


def test_planner_rejects_invalid_target_dimension(scene_state_text):
    """Verify Pydantic validator catches invalid target dimensions (e.g. 2D instead of 3D)."""
    with pytest.raises(ValueError, match="target must be a 3D coordinate"):
        PlanStep(action="approach", arm="left_arm", object="plate", target=[0.1, 0.2])


# ------------------------------------------------------------------------------
# 3. Configuration & Model Name Introspection Tests
# ------------------------------------------------------------------------------

def test_planner_model_loaded_from_config():
    """Verify planner retrieves model name strictly from sim_config.yaml without hardcoding."""
    config_path = Path("configs/sim_config.yaml")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expected_model = cfg["planning"]["llm"]["model"]
    planner = LLMPlanner()
    assert planner.model_name == expected_model
    assert len(planner.model_name) > 0


def test_planner_missing_model_config_raises():
    """Verify ValueError is raised if sim_config.yaml is missing planning.llm.model."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump({"simulation": {}}, f)
        temp_cfg = f.name

    try:
        with pytest.raises(ValueError, match="LLM model name is not configured"):
            LLMPlanner(config_path=temp_cfg)
    finally:
        Path(temp_cfg).unlink(missing_ok=True)


# ------------------------------------------------------------------------------
# 4. Perception & Scene State Integration Test
# ------------------------------------------------------------------------------

def test_scene_state_perception_integration(sim):
    """Verify ground truth scene state extractor generates valid compact description for planner."""
    state = get_ground_truth_scene_state(sim)
    assert "plate" in state.objects
    assert "mug" in state.objects
    assert "drawer" in state.objects
    assert "left_arm" in state.arms
    assert "right_arm" in state.arms

    text = format_scene_state(sim)
    assert "CURRENT SCENE STATE:" in text
    assert "plate: pos=" in text
    assert "mug: pos=" in text
    assert "drawer: pos=" in text
    assert "left_arm (arm A):" in text
    assert "right_arm (arm B):" in text

    # Plan with live formatted scene state
    steps = plan("Pick up the plate with left_arm and place it on the table.", text)
    assert len(steps) > 0
    for s in steps:
        assert_valid_plan_step(s)


def test_planner_visual_detections_grounding(scene_state_text):
    """Verify visual detections from neural detector are grounded into LLM prompt."""
    planner = LLMPlanner()
    mock_detections = [
        {"label": "mug", "confidence": 0.94, "box": [10, 20, 50, 60], "coords": [0.22, -0.16, 0.465]},
        {"label": "plate", "confidence": 0.98, "box": [100, 110, 180, 190], "coords": [0.25, 0.0, 0.435]},
    ]
    prompt = planner.build_prompt(
        instruction="Pick up the plate",
        scene_state=scene_state_text,
        visual_detections=mock_detections,
    )

    assert "VISUAL PERCEPTION (CAMERA DETECTIONS):" in prompt
    assert "Detected 'mug': confidence=0.94" in prompt
    assert "Detected 'plate': confidence=0.98" in prompt

    # Generating a plan with visual detections returns valid steps
    steps = planner.plan(
        instruction="Pick up the plate with left arm and place it on the table",
        scene_state=scene_state_text,
        visual_detections=mock_detections,
    )
    assert len(steps) > 0
    for s in steps:
        assert_valid_plan_step(s)


def test_planner_openrouter_fallback(scene_state_text):
    """Verify openrouter provider falls back safely or processes prompt."""
    planner = LLMPlanner()
    planner.provider = "openrouter"
    # Should execute without uncaught network exceptions, falling back to deterministic mock if needed
    steps = planner.plan("Pick up the plate with left arm", scene_state_text)
    assert len(steps) > 0
    for s in steps:
        assert_valid_plan_step(s)
