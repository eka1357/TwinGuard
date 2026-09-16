"""Unit and integration tests for the TwinGuard MuJoCo simulation environment."""

import pytest
import numpy as np
from simulation.simulator import TwinGuardSim


@pytest.fixture
def sim():
    """Fixture providing a fresh TwinGuardSim instance for the single-arm scene."""
    sim_instance = TwinGuardSim(model_path="simulation/models/scene_single.xml")
    sim_instance.reset()
    return sim_instance


def test_model_loads(sim):
    """Test that the scene XML loads without errors and has expected DOFs."""
    assert sim.model is not None
    assert sim.data is not None
    assert sim.model.nq == 6, f"Expected 6 joint positions, got {sim.model.nq}"
    assert sim.model.nv == 6, f"Expected 6 degrees of freedom, got {sim.model.nv}"
    assert sim.model.nu == 6, f"Expected 6 actuators, got {sim.model.nu}"


def test_joint_names(sim):
    """Verify that all 6 SO-101 joints are properly registered in the model."""
    expected_joints = [
        "joint_base_yaw",
        "joint_shoulder_pitch",
        "joint_elbow_pitch",
        "joint_wrist_pitch",
        "joint_wrist_roll",
        "joint_gripper",
    ]
    assert sim.joint_names == expected_joints


def test_actuator_names(sim):
    """Verify that all 6 actuators are properly registered."""
    expected_actuators = [
        "actuator_base_yaw",
        "actuator_shoulder_pitch",
        "actuator_elbow_pitch",
        "actuator_wrist_pitch",
        "actuator_wrist_roll",
        "actuator_gripper",
    ]
    assert sim.actuator_names == expected_actuators


def test_physics_step_stability(sim):
    """Verify that simulation steps advance time and remain numerically stable."""
    dt = sim.get_timestep()
    assert dt > 0.0

    initial_time = sim.get_time()
    sim.step(100)
    final_time = sim.get_time()

    assert final_time > initial_time
    assert pytest.approx(final_time - initial_time, abs=1e-5) == 100 * dt
    assert sim.is_stable(), "Simulation state must remain physically stable (no NaNs/Infs)"


def test_actuator_targets_apply(sim):
    """Verify that set_joint_targets sets control values on actuators."""
    targets = {
        "actuator_base_yaw": 30.0,
        "actuator_shoulder_pitch": -20.0,
        "actuator_gripper": 15.0,
    }
    sim.set_joint_targets(targets)

    # Step simulation so control acts on the system
    sim.step(10)

    assert sim.is_stable()
    positions = sim.get_joint_positions()
    for j_name in sim.joint_names:
        assert not np.isnan(positions[j_name])


def test_simulation_reset(sim):
    """Verify that reset returns simulation time to 0 and clears velocities."""
    sim.step(50)
    assert sim.get_time() > 0.0

    sim.reset()
    assert sim.get_time() == 0.0
    velocities = sim.get_joint_velocities()
    for j_name, vel in velocities.items():
        assert vel == 0.0, f"Joint velocity for {j_name} should be 0 after reset"


def test_left_gripper_position():
    """Verify that the left gripper site position is a 3D NumPy array."""
    sim = TwinGuardSim()
    try:
        position = sim.get_gripper_position("left_arm")
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
    finally:
        sim.close()


def test_right_gripper_position():
    """Verify that the right gripper site position is a 3D NumPy array."""
    sim = TwinGuardSim()
    try:
        position = sim.get_gripper_position("right_arm")
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
    finally:
        sim.close()


def test_invalid_gripper_arm_raises_value_error():
    """Verify that an unknown arm is rejected."""
    sim = TwinGuardSim()
    try:
        with pytest.raises(ValueError):
            sim.get_gripper_position("invalid_arm")
    finally:
        sim.close()


def test_invalid_site_raises_value_error():
    """Verify that an unknown site is rejected."""
    sim = TwinGuardSim()
    try:
        with pytest.raises(ValueError):
            sim.get_site_position("invalid_site")
    finally:
        sim.close()
