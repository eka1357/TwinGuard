"""Automated test suite for the TwinGuard multi-seed evaluation runner and report generator.

Verifies:
- Seed-based scene randomization for plate, mug, and drawer start positions.
- Execution across randomized seeds and output of evaluation/results.json.
- Controlled failure injection and TwinGuard's dynamic recovery replanning.
- Report generator parsing and summary metric calculations.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from simulation.simulator import TwinGuardSim
from evaluation.runner import (
    randomize_scene_for_seed,
    run_evaluation,
)
from evaluation.report import generate_report


@pytest.fixture
def sim():
    """Fixture providing a fresh simulation instance."""
    sim_inst = TwinGuardSim()
    sim_inst.reset()
    yield sim_inst
    sim_inst.close()


def test_randomize_scene_for_seed(sim):
    """Verify that different seeds produce distinct, stable starting positions."""
    poses_seed_0 = randomize_scene_for_seed(sim, seed=0)
    assert sim.is_stable()
    assert "plate" in poses_seed_0
    assert "mug" in poses_seed_0
    assert "drawer" in poses_seed_0

    sim.reset()
    poses_seed_1 = randomize_scene_for_seed(sim, seed=1)
    assert sim.is_stable()

    # Different seeds should produce perturbed positions
    assert poses_seed_0["plate"] != poses_seed_1["plate"] or poses_seed_0["mug"] != poses_seed_1["mug"]

    # Positions should stay on table workspace bounds
    assert 0.20 <= poses_seed_0["plate"][0] <= 0.30
    assert -0.05 <= poses_seed_0["plate"][1] <= 0.05
    assert 0.18 <= poses_seed_0["mug"][0] <= 0.26
    assert -0.20 <= poses_seed_0["mug"][1] <= -0.10


def test_runner_subset_execution(tmp_path):
    """Verify run_evaluation executes on a small subset of seeds and saves valid results."""
    out_file = tmp_path / "test_results.json"

    results = run_evaluation(
        num_seeds=2,
        output_path=out_file,
        forced_failure_seeds=[],
        verbose=False,
    )

    assert out_file.exists()
    assert results["summary"]["total_seeds"] == 2
    assert results["summary"]["successful_seeds"] == 2
    assert results["summary"]["task_success_rate"] == 100.0
    assert len(results["seeds"]) == 2

    # Check first seed
    seed0 = results["seeds"][0]
    assert seed0["seed"] == 0
    assert seed0["success"] is True
    assert seed0["total_steps"] == 10
    assert seed0["steps_succeeded"] == 10
    assert len(seed0["steps"]) == 10


def test_controlled_failure_injection_and_recovery(tmp_path):
    """Verify that a forced failure triggers dynamic recovery and successfully recovers."""
    out_file = tmp_path / "forced_failure_results.json"

    # Run seed 0 with forced failure enabled
    results = run_evaluation(
        num_seeds=1,
        output_path=out_file,
        forced_failure_seeds=[0],
        verbose=False,
    )

    seed0 = results["seeds"][0]
    assert seed0["forced_failure"] is True
    assert seed0["success"] is True
    assert seed0["recovery_attempts"] >= 1
    assert seed0["steps_recovered"] >= 1

    # Locate the recovered step (grasp on plate)
    recovered_steps = [s for s in seed0["steps"] if s["recovered"]]
    assert len(recovered_steps) >= 1
    rec_step = recovered_steps[0]
    assert rec_step["action"] == "grasp"
    assert rec_step["initial_success"] is False
    assert rec_step["final_success"] is True
    assert rec_step["recovery_attempts"] == 1


def test_report_generation(tmp_path, capsys):
    """Verify generate_report reads results JSON and outputs summary table."""
    sample_data = {
        "summary": {
            "total_seeds": 2,
            "successful_seeds": 2,
            "task_success_rate": 100.0,
            "total_grasps": 4,
            "initial_grasp_success_rate": 75.0,
            "final_grasp_success_rate": 100.0,
            "total_failures_encountered": 1,
            "total_failures_recovered": 1,
            "recovery_success_rate": 100.0,
            "total_collisions": 0,
            "avg_sim_time_s": 4.5,
            "avg_wall_time_s": 2.1,
        },
        "seeds": [
            {
                "seed": 0,
                "success": True,
                "total_steps": 10,
                "steps_succeeded": 10,
                "forced_failure": True,
                "steps_recovered": 1,
                "collisions": 0,
                "sim_time": 4.5,
                "wall_time": 2.1,
            },
            {
                "seed": 1,
                "success": True,
                "total_steps": 10,
                "steps_succeeded": 10,
                "forced_failure": False,
                "steps_recovered": 0,
                "collisions": 0,
                "sim_time": 4.5,
                "wall_time": 2.1,
            },
        ],
    }

    dummy_results = tmp_path / "dummy_results.json"
    with open(dummy_results, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    res = generate_report(results_path=dummy_results, verbose=True)
    captured = capsys.readouterr().out

    assert "TWINGUARD BIMANUAL MANIPULATION EVALUATION REPORT" in captured
    assert "Task Success Rate" in captured
    assert "100.0%" in captured
    assert "Dynamic Recovery Success Rate" in captured
    assert res["summary"]["task_success_rate"] == 100.0
