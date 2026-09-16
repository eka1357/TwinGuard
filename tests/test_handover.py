"""Tests for coordinated bimanual handover primitive in TwinGuard."""

import pytest
import numpy as np
import mujoco

from simulation.simulator import TwinGuardSim
from robotics.primitives import MotionPrimitives, handover
from safety.verifier import SafetyVerifier


@pytest.fixture
def sim():
    sim = TwinGuardSim()
    sim.reset()
    yield sim
    sim.close()


def test_handover_primitive_execution(sim):
    """Verify that source arm can present plate and destination arm can grasp it."""
    primitives = MotionPrimitives(sim)
    verifier = SafetyVerifier(sim)

    # 1. Left arm approaches and grasps plate
    p_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    p_pos = sim.data.xpos[p_bid]

    app_ok = primitives.approach("left_arm", [p_pos[0], p_pos[1], 0.44])
    assert app_ok, "Left arm approach failed"

    grasp_ok = primitives.grasp("left_arm", object_name="plate")
    assert grasp_ok, "Left arm grasp failed"

    lift_ok = primitives.lift("left_arm", height=0.06)
    assert lift_ok, "Left arm lift failed"

    # 2. Coordinated bimanual handover to right_arm
    handover_ok = primitives.handover("left_arm", "right_arm", object_name="plate")
    assert handover_ok, "Bimanual handover failed"

    # 3. Verify handover outcome
    ver = verifier.verify_step(
        {"action": "handover", "arm": "left_arm", "object": "plate"},
        primitive_returned=handover_ok,
    )
    assert ver["success"], f"Handover verification failed: {ver['reason']}"
    assert sim.is_stable(), "Simulation unstable after handover"


def test_handover_functional_api(sim):
    """Verify the top-level functional handover API."""
    primitives = MotionPrimitives(sim)

    # Move left arm to plate
    p_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    p_pos = sim.data.xpos[p_bid]
    primitives.approach("left_arm", [p_pos[0], p_pos[1], 0.44])
    primitives.grasp("left_arm", object_name="plate")

    ok = handover("left_arm", "right_arm", object_name="plate", sim=sim)
    assert ok, "Functional handover failed"
