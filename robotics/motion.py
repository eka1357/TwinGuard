from typing import Mapping


def move_arm_to_targets(
    sim,
    arm: str,
    targets: Mapping[str, float],
    steps: int = 100,
) -> None:
    if steps <= 0:
        raise ValueError("steps must be greater than zero")

    sim.set_arm_joint_targets(arm, dict(targets))

    for _ in range(steps):
        sim.step(1)

        if not sim.is_stable():
            raise RuntimeError(
                f"Simulation became unstable while moving {arm}."
            )
