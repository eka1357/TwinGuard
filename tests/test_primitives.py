"""Automated tests for TwinGuard deterministic motion primitives.

Verifies:
- approach(arm, target_xyz): reaches target within tolerance using Jacobian IK
- grasp(arm): closes gripper reliably without physics destabilization
- lift(arm, height): vertical translation within tolerance
- transport(arm, target_xyz): spatial translation to target pose
- release(arm): opens gripper to commanded width
- open_drawer(arm, drawer_body): grasps handle and pulls drawer along slide axis
- pour(arm, target_container): geometric proxy tilting held object over container
- Physics stability: verifies finite qpos/qvel and absence of numerical explosion
- Support for both functional and MotionPrimitives class interfaces
"""

import pytest
import numpy as np
import mujoco

from simulation.simulator import TwinGuardSim
from robotics.primitives import (
    MotionPrimitives,
    approach,
    grasp,
    lift,
    transport,
    release,
    open_drawer,
    pour,
)


@pytest.fixture
def sim():
    """Fixture providing a fresh TwinGuardSim dual-arm instance."""
    sim_instance = TwinGuardSim()
    sim_instance.reset()
    yield sim_instance
    sim_instance.close()


@pytest.fixture
def primitives(sim):
    """Fixture providing MotionPrimitives controller initialized with active sim."""
    return MotionPrimitives(sim)


def assert_stable_physics(sim: TwinGuardSim) -> None:
    """Helper verifying finite state, no NaNs/Infs, and bounded velocity."""
    assert sim.is_stable(), "Simulation state reported unstable"
    assert not np.any(np.isnan(sim.data.qpos)), "qpos contains NaNs"
    assert not np.any(np.isinf(sim.data.qpos)), "qpos contains Infs"
    assert not np.any(np.isnan(sim.data.qvel)), "qvel contains NaNs"
    assert not np.any(np.isinf(sim.data.qvel)), "qvel contains Infs"
    assert np.all(np.abs(sim.data.qvel) < 50.0), (
        f"Excessive velocities detected: max={np.max(np.abs(sim.data.qvel))}"
    )


def test_approach_primitive_both_arms(primitives, sim):
    """Verify approach primitive drives end-effector to target within tolerance for both arms."""
    pos_tol = float(primitives._get_param("position_tolerance", 0.02))

    # Test left arm approach
    target_left = np.array([0.20, 0.20, 0.55])
    success_left = primitives.approach("left_arm", target_left)
    assert success_left, "Left arm approach failed"
    assert_stable_physics(sim)

    ee_left = primitives.get_ee_position("left_arm")
    err_left = np.linalg.norm(target_left - ee_left)
    assert err_left <= pos_tol, f"Left arm error {err_left:.4f}m exceeded tolerance {pos_tol}m"

    # Test right arm approach
    target_right = np.array([0.20, -0.20, 0.55])
    success_right = primitives.approach("right_arm", target_right)
    assert success_right, "Right arm approach failed"
    assert_stable_physics(sim)

    ee_right = primitives.get_ee_position("right_arm")
    err_right = np.linalg.norm(target_right - ee_right)
    assert err_right <= pos_tol, f"Right arm error {err_right:.4f}m exceeded tolerance {pos_tol}m"


def test_grasp_and_release_primitives(primitives, sim):
    """Verify grasp closes the gripper and release reopens it stably."""
    # Initially open gripper
    primitives.release("left_arm")
    assert_stable_physics(sim)
    open_pos = sim.get_arm_gripper_position("left_arm", in_degrees=True)
    assert open_pos > 25.0, f"Expected gripper to open, got {open_pos:.1f} deg"

    # Grasp (close)
    grasp_success = primitives.grasp("left_arm")
    assert grasp_success, "Grasp primitive returned False"
    assert_stable_physics(sim)
    closed_pos = sim.get_arm_gripper_position("left_arm", in_degrees=True)
    assert closed_pos < 10.0, f"Expected gripper to close, got {closed_pos:.1f} deg"

    # Release (open)
    release_success = primitives.release("left_arm")
    assert release_success, "Release primitive returned False"
    assert_stable_physics(sim)
    reopened_pos = sim.get_arm_gripper_position("left_arm", in_degrees=True)
    assert reopened_pos > 25.0, f"Expected gripper to reopen, got {reopened_pos:.1f} deg"


def test_lift_primitive(primitives, sim):
    """Verify lift moves the end-effector upwards by specified height."""
    pos_tol = float(primitives._get_param("position_tolerance", 0.02))
    initial_target = [0.20, 0.22, 0.50]
    primitives.approach("left_arm", initial_target)
    assert_stable_physics(sim)

    pos_before = primitives.get_ee_position("left_arm")
    lift_h = 0.08
    success = primitives.lift("left_arm", height=lift_h)
    assert success, "Lift primitive failed"
    assert_stable_physics(sim)

    pos_after = primitives.get_ee_position("left_arm")
    expected_z = pos_before[2] + lift_h
    assert abs(pos_after[2] - expected_z) <= pos_tol, (
        f"Lift height error: expected z={expected_z:.4f}, got {pos_after[2]:.4f}"
    )


def test_transport_primitive(primitives, sim):
    """Verify transport moves the arm end-effector to destination coordinates."""
    pos_tol = float(primitives._get_param("position_tolerance", 0.02))

    # Move to starting point
    start_xyz = [0.20, 0.22, 0.52]
    primitives.approach("left_arm", start_xyz)

    # Transport to distinct destination
    dest_xyz = [0.15, 0.12, 0.58]
    success = primitives.transport("left_arm", dest_xyz)
    assert success, "Transport primitive failed"
    assert_stable_physics(sim)

    final_pos = primitives.get_ee_position("left_arm")
    err = np.linalg.norm(np.array(dest_xyz) - final_pos)
    assert err <= pos_tol, f"Transport error {err:.4f}m exceeded tolerance {pos_tol}m"


def test_open_drawer_primitive(primitives, sim):
    """Verify open_drawer grasps handle, pulls along slide joint, and releases."""
    # Ensure drawer slide joint exists
    assert "drawer_joint" in sim.joint_names

    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint")
    qpos_adr = sim.model.jnt_qposadr[jid]
    initial_drawer_pos = float(sim.data.qpos[qpos_adr])

    # Left arm opens drawer
    success = primitives.open_drawer("left_arm", drawer_body="drawer")
    assert success, "open_drawer primitive failed"
    assert_stable_physics(sim)

    final_drawer_pos = float(sim.data.qpos[qpos_adr])
    # Verify drawer moved open
    assert final_drawer_pos > initial_drawer_pos + 0.005, (
        f"Drawer did not slide open: initial={initial_drawer_pos:.4f}, final={final_drawer_pos:.4f}"
    )


def test_pour_geometric_proxy(primitives, sim):
    """Verify pour primitive hovers above target container and tilts wrist."""
    assert "mug" in [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(sim.model.nbody)]

    # Right arm approaches mug and pours
    success = primitives.pour("right_arm", target_container="mug", tilt_angle_deg=45.0, hold_steps=40)
    assert success, "pour primitive failed"
    assert_stable_physics(sim)

    # Verify wrist orientation returned to neutral upright state
    wrist_roll = sim.get_arm_joint_positions("right_arm", in_degrees=True).get("joint_wrist_roll", 0.0)
    assert abs(wrist_roll) < 15.0, f"Wrist roll should be returned to near zero, got {wrist_roll:.1f} deg"


def test_functional_api_delegation(sim):
    """Verify module-level standalone functions work identically to class methods."""
    target = [0.20, 0.20, 0.55]
    ok = approach("left_arm", target, sim=sim)
    assert ok
    assert_stable_physics(sim)

    ok = grasp("left_arm", sim=sim)
    assert ok
    assert_stable_physics(sim)

    ok = lift("left_arm", height=0.06, sim=sim)
    assert ok
    assert_stable_physics(sim)

    ok = transport("left_arm", [0.18, 0.18, 0.58], sim=sim)
    assert ok
    assert_stable_physics(sim)

    ok = release("left_arm", sim=sim)
    assert ok
    assert_stable_physics(sim)


def test_sequential_bimanual_pipeline(primitives, sim):
    """Verify a complete multi-step manipulation sequence completes with full stability."""
    # 1. Left arm approaches and opens drawer
    assert primitives.open_drawer("left_arm", drawer_body="drawer")
    assert_stable_physics(sim)

    # 2. Right arm approaches mug on table
    mug_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    mug_pos = sim.data.xpos[mug_bid]
    assert primitives.approach("right_arm", [mug_pos[0], mug_pos[1], mug_pos[2] + 0.04])
    assert_stable_physics(sim)

    # 3. Right arm grasps mug
    assert primitives.grasp("right_arm")
    assert_stable_physics(sim)

    # 4. Right arm lifts mug
    assert primitives.lift("right_arm", height=0.07)
    assert_stable_physics(sim)

    # 5. Right arm pours towards plate
    assert primitives.pour("right_arm", target_container="plate", tilt_angle_deg=40.0, hold_steps=30)
    assert_stable_physics(sim)

    # 6. Right arm releases
    assert primitives.release("right_arm")
    assert_stable_physics(sim)
