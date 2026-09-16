"""Tests for Hugging Face LeRobot and SmolVLA interface adapter."""

import pytest
import numpy as np

from simulation.simulator import TwinGuardSim
from planning.vla_interface import (
    CANONICAL_12DOF_JOINTS,
    CANONICAL_12DOF_ACTUATORS,
    get_lerobot_state,
    get_lerobot_velocity,
    get_lerobot_observation,
    apply_lerobot_action,
    SmolVLAPolicyAdapter,
)


@pytest.fixture
def sim():
    sim = TwinGuardSim()
    sim.reset()
    yield sim
    sim.close()


def test_lerobot_state_and_velocity_dimensions(sim):
    """Verify state and velocity match 12-DOF bi_so101_follower convention."""
    state = get_lerobot_state(sim)
    vel = get_lerobot_velocity(sim)

    assert len(CANONICAL_12DOF_JOINTS) == 12
    assert len(CANONICAL_12DOF_ACTUATORS) == 12
    assert isinstance(state, np.ndarray)
    assert state.shape == (12,)
    assert isinstance(vel, np.ndarray)
    assert vel.shape == (12,)


def test_lerobot_observation_dict(sim):
    """Verify observation dictionary conforms to SmolVLA policy input schema."""
    obs = get_lerobot_observation(sim, include_wrist_cameras=True)

    assert "observation.state" in obs
    assert "observation.velocity" in obs
    assert "observation.images.overview" in obs

    overview = obs["observation.images.overview"]
    assert isinstance(overview, np.ndarray)
    assert overview.ndim == 3
    assert overview.shape[2] == 3


def test_apply_lerobot_action(sim):
    """Verify 12-DOF continuous control vector is applied to actuators."""
    action = np.zeros(12, dtype=np.float64)
    action[0] = 0.1  # left yaw
    action[6] = -0.1  # right yaw

    apply_lerobot_action(sim, action)
    sim.step(5)
    assert sim.is_stable()


def test_smolvla_policy_adapter():
    """Verify SmolVLA adapter formats observation tensors and generates action chunks."""
    adapter = SmolVLAPolicyAdapter()
    dummy_obs = {
        "observation.state": np.zeros(12),
        "observation.images.overview": np.zeros((128, 128, 3), dtype=np.uint8),
    }

    inp = adapter.format_input(dummy_obs, "pick plate and handover to arm B")
    assert "text" in inp
    assert "state" in inp

    chunk = adapter.predict_action_chunk(dummy_obs, "pick plate", chunk_size=8)
    assert chunk.shape == (8, 12)
