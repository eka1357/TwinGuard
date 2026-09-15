# Challenge brief — ground truth, do not deviate

Intel Physical AI Online Challenge — "Bimanual VLA Manipulation with Multi-Modal Reasoning"
Challenge option: Setting Up a Dinner Table (AI Infra Summit Hackathon, lablab.ai)
Deadline: Sep 17, 2026, 01:00 UTC

## Task
Two SO-101 arms in MuJoCo take a natural-language instruction, perceive the scene via
cameras, execute a multi-step dinner-table task (open drawer, retrieve cutlery, pick up
plate/cup, hand off between arms, pour). Worked example from the brief:
"Open the top drawer, pick up the plate with arm A, place it on the table, pick up the
mug with arm B, pour water into the mug with arm A."

## Hard constraint
Final inference/benchmark MUST run on Intel hardware (CPU/iGPU/NPU, OpenVINO) — no cloud
GPU on the graded path. Training hardware is unconstrained.

## Required deliverables
1. Reproducible GitHub repo  2. Reproducible MuJoCo sim package
3. Intel inference benchmark script  4. Demo video across 10 randomized seeds
5. Technical README / architecture summary

## Judging (100 pts)
30 end-to-end task completion & bimanual manipulation · 20 VLA/multi-modal reasoning ·
20 OpenVINO & Core Ultra optimization · 15 robustness (10 seeds) · 10 technical
quality/reproducibility · 5 innovation

## Locked technical decisions
- MJCF: TheRobotStudio/SO-ARM100 (Apache-2.0), instantiated twice for bimanual.
- Action/obs schema: LeRobot bi_so101_follower convention (12-dim = 6 joints x 2 arms), SmolVLA.
- No hardcoded paths/seeds/thresholds in code — everything in configs/*.yaml.