"""Automated test suite for TwinGuard dual-arm MuJoCo simulation environment.

Verifies:
- Dual-arm scene loading (left_arm and right_arm with independent namespaces)
- Table and plate placeholder object
- Independent arm motion commands without cross-coupling
- Independent gripper actuation (open/close)
- Physical stability and absence of numerical explosion / interpenetration
- Camera rendering from multiple viewpoints
- Configuration-driven parameter binding
"""

import pytest
import numpy as np
from simulation.simulator import TwinGuardSim


@pytest.fixture
def sim():
    """Fixture providing a fresh TwinGuardSim dual-arm instance."""
    sim_instance = TwinGuardSim()
    sim_instance.reset()
    yield sim_instance
    sim_instance.close()


def test_dual_arm_model_loads(sim):
    """Verify that dual-arm scene loads cleanly with expected arms, DOFs, and objects."""
    assert sim.model is not None
    assert sim.data is not None

    # Verify configured arms
    assert "left_arm" in sim.arm_names
    assert "right_arm" in sim.arm_names
    assert len(sim.arm_names) == 2

    # Verify 6 joints per arm
    assert len(sim.arm_joints["left_arm"]) == 6
    assert len(sim.arm_joints["right_arm"]) == 6

    # Verify 6 actuators per arm (12 total actuators)
    assert len(sim.arm_actuators["left_arm"]) == 6
    assert len(sim.arm_actuators["right_arm"]) == 6
    assert sim.model.nu == 12, f"Expected 12 actuators (6 per arm), got {sim.model.nu}"

    # Verify scene objects exist and have appropriate joints
    assert "plate_joint" in sim.joint_names
    assert "mug_joint" in sim.joint_names
    assert "drawer_joint" in sim.joint_names

    # Total DOFs: 6 (left) + 6 (right) + 6 (plate) + 6 (mug) + 1 (drawer slide) = 25 DOFs (nv)
    # Total qpos: 6 (left) + 6 (right) + 7 (plate) + 7 (mug) + 1 (drawer slide) = 27 qpos (nq)
    assert sim.model.nv == 25, f"Expected 25 DOFs, got {sim.model.nv}"
    assert sim.model.nq == 27, f"Expected 27 qpos values, got {sim.model.nq}"



def test_independent_namespaces(sim):
    """Verify that joint and actuator names between left_arm and right_arm do not collide."""
    left_joints = set(sim.arm_joints["left_arm"])
    right_joints = set(sim.arm_joints["right_arm"])
    assert left_joints.isdisjoint(right_joints), "Left and right arm joint namespaces must be disjoint"

    left_actuators = set(sim.arm_actuators["left_arm"])
    right_actuators = set(sim.arm_actuators["right_arm"])
    assert left_actuators.isdisjoint(right_actuators), "Left and right actuator namespaces must be disjoint"


def test_independent_arm_motion(sim):
    """Confirm both arms can be commanded to different target poses independently."""
    # Command left arm in positive direction
    sim.set_arm_joint_targets("left_arm", {
        "joint_base_yaw": 30.0,
        "joint_shoulder_pitch": -20.0,
        "joint_elbow_pitch": 25.0,
    })

    # Command right arm in opposite / different direction
    sim.set_arm_joint_targets("right_arm", {
        "joint_base_yaw": -30.0,
        "joint_shoulder_pitch": 20.0,
        "joint_elbow_pitch": -25.0,
    })

    # Step simulation
    sim.step(200)

    assert sim.is_stable(), "Simulation must remain stable during independent dual arm motion"

    left_pos = sim.get_arm_joint_positions("left_arm")
    right_pos = sim.get_arm_joint_positions("right_arm")

    # Verify left arm moved towards positive base yaw and elbow
    assert left_pos["joint_base_yaw"] > 1.0, f"Left base yaw should have moved positive, got {left_pos['joint_base_yaw']}"
    assert left_pos["joint_elbow_pitch"] > 1.0, f"Left elbow should have moved positive, got {left_pos['joint_elbow_pitch']}"

    # Verify right arm moved towards negative base yaw and elbow
    assert right_pos["joint_base_yaw"] < -1.0, f"Right base yaw should have moved negative, got {right_pos['joint_base_yaw']}"
    assert right_pos["joint_elbow_pitch"] < -1.0, f"Right elbow should have moved negative, got {right_pos['joint_elbow_pitch']}"

    # Verify divergence between the two arms (opposing signs)
    assert np.sign(left_pos["joint_base_yaw"]) != np.sign(right_pos["joint_base_yaw"])
    assert np.sign(left_pos["joint_elbow_pitch"]) != np.sign(right_pos["joint_elbow_pitch"])


def test_command_isolation(sim):
    """Verify that commanding only left arm does not command right arm actuators."""
    initial_right_pos = sim.get_arm_joint_positions("right_arm")

    sim.set_arm_joint_targets("left_arm", {
        "joint_base_yaw": 35.0,
        "joint_shoulder_pitch": -25.0,
    })
    sim.step(150)

    left_pos = sim.get_arm_joint_positions("left_arm")
    right_pos = sim.get_arm_joint_positions("right_arm")

    # Left arm moved significantly
    assert abs(left_pos["joint_base_yaw"]) > 2.0

    # Right arm remained near initial resting state
    for joint, init_val in initial_right_pos.items():
        assert pytest.approx(right_pos[joint], abs=0.5) == init_val, (
            f"Right arm joint {joint} should remain unaffected when only commanding left arm"
        )


def test_independent_gripper_control(sim):
    """Verify that grippers for each arm can be opened and closed independently."""
    sim.open_gripper("left_arm")
    sim.close_gripper("right_arm")

    sim.step(150)

    assert sim.is_stable()
    left_grip = sim.get_arm_gripper_position("left_arm")
    right_grip = sim.get_arm_gripper_position("right_arm")

    # Left gripper should have opened significantly towards 40.0
    assert left_grip > 5.0, f"Left gripper expected to open, got {left_grip}"
    # Right gripper should stay close to closed position 0.0
    assert right_grip < 1.0, f"Right gripper expected to stay closed, got {right_grip}"


def test_physics_stability_and_finite_states(sim):
    """Verify simulation remains finite and physically stable over 500 frames of motion."""
    dt = sim.get_timestep()
    assert dt > 0.0

    # Command dynamic targets
    sim.set_arm_joint_targets("left_arm", {"joint_base_yaw": 20.0, "joint_shoulder_pitch": -15.0})
    sim.set_arm_joint_targets("right_arm", {"joint_base_yaw": -20.0, "joint_shoulder_pitch": 15.0})

    for step_idx in range(500):
        sim.step(1)
        assert sim.is_stable(), f"Simulation became unstable at step {step_idx}"

    # Verify no NaN or Inf in qpos/qvel
    assert not np.any(np.isnan(sim.data.qpos)), "qpos contains NaNs"
    assert not np.any(np.isinf(sim.data.qpos)), "qpos contains Infs"
    assert not np.any(np.isnan(sim.data.qvel)), "qvel contains NaNs"
    assert not np.any(np.isinf(sim.data.qvel)), "qvel contains Infs"

    # Verify velocities are reasonable (no explosive interpenetration velocities)
    assert np.all(np.abs(sim.data.qvel) < 50.0), f"Excessive velocities detected: {np.max(np.abs(sim.data.qvel))}"

    # Verify plate is resting stably on the table (z around 0.42m - 0.44m, within table x,y bounds)
    positions = sim.get_joint_positions()
    plate_state = positions["plate_joint"]
    assert isinstance(plate_state, list) and len(plate_state) == 7
    plate_x, plate_y, plate_z = plate_state[0], plate_state[1], plate_state[2]
    assert -0.2 <= plate_x <= 0.6, f"Plate x {plate_x} drifted outside table"
    assert -0.5 <= plate_y <= 0.5, f"Plate y {plate_y} drifted outside table"
    assert 0.41 <= plate_z <= 0.46, f"Plate z {plate_z} not resting on table surface"


def test_camera_rendering(sim):
    """Verify scene camera rendering works and produces valid images."""
    camera_names = sim.get_camera_names()
    assert "overview_cam" in camera_names
    assert "front_cam" in camera_names

    # Render overview camera (1280x720 from config)
    img_overview = sim.get_camera_image("overview_cam")
    assert isinstance(img_overview, np.ndarray)
    assert img_overview.dtype == np.uint8
    assert img_overview.shape == (720, 1280, 3)
    # Check image is not empty
    assert np.mean(img_overview) > 10.0, "Rendered image should not be black/blank"

    # Render front camera (640x480 from config)
    img_front = sim.get_camera_image("front_cam")
    assert isinstance(img_front, np.ndarray)
    assert img_front.shape == (480, 640, 3)
    assert np.mean(img_front) > 10.0


def test_config_driven_no_magic_numbers(sim):
    """Verify that configuration file drives arm names, cameras, and model paths."""
    assert "robot" in sim.config
    assert "arms" in sim.config["robot"]
    assert "left_arm" in sim.config["robot"]["arms"]
    assert "right_arm" in sim.config["robot"]["arms"]
    assert "cameras" in sim.config
    assert "overview_cam" in sim.config["cameras"]
    assert sim.config["simulation"]["model_path"] == "simulation/models/scene.xml"
