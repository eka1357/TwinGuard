"""Smoke test for command execution against the real MuJoCo simulator."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from planning.command_planner import prepare_command
from robotics.command_executor import execute_command
from simulation.simulator import TwinGuardSim


def main() -> None:
    sim = TwinGuardSim()

    try:
        sim.reset()

        left_command = prepare_command({
            "arm": "A",
            "action": "open",
            "obj": "plate",
            "target": None,
        })

        execute_command(sim, left_command)
        sim.step(1)

        right_command = prepare_command({
            "arm": "B",
            "action": "open",
            "obj": "plate",
            "target": None,
        })

        execute_command(sim, right_command)
        sim.step(1)

        assert sim.is_stable()
        print("[SUCCESS] Real simulator command smoke test passed.")

    finally:
        sim.close()


if __name__ == "__main__":
    main()