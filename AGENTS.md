
# TwinGuard Development Rules

## Project Goal

Build a reliable bimanual Physical AI system using two simulated SO-101 arms in MuJoCo.

The system must eventually:
- Understand natural-language instructions.
- Use visual observations.
- Coordinate two robotic arms.
- Perform multi-step manipulation.
- Detect failures.
- Recover through re-observation and replanning.
- Support OpenVINO and Intel hardware benchmarking.

## Development Rules

1. Work incrementally. Do not build the entire project at once.
2. Read this file before making changes.
3. Preserve working code when adding features.
4. Do not claim that code works unless it has been executed and tested.
5. Add a test or runnable command for every major feature.
6. Keep robotics, planning, perception, safety, and evaluation modules separate.
7. Do not replace the required VLA/Physical AI component with only hardcoded robot movements.
8. Document installation and execution steps.
9. Avoid unnecessary dependencies.
10. Explain errors clearly and suggest practical fixes.

## Current Milestone

M1 — Verify the Python and MuJoCo development environment.

## Current Scope

Only:
- Inspect the repository.
- Set up the Python environment.
- Install required basic dependencies.
- Create a minimal MuJoCo test.
- Document how to run the test.

Do not implement:
- VLA integration.
- OpenVINO optimization.
- TwinGuard recovery logic.
- Complex manipulation.
