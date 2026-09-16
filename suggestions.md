# TwinGuard — Improvement Suggestions for the AI Infra Summit Hackathon

> **Context**: Intel Physical AI Online Challenge — "Bimanual VLA Manipulation with
> Multi-Modal Reasoning" (lablab.ai × AI Infra Summit, deadline Sep 17, 2026 01:00 UTC).
> Research based on official challenge brief, judging rubric, Intel's recommended
> technology stack, and lablab.ai submission guidelines.

---

## ✅ Current Project Health Check

| Area | Status | Notes |
|:---|:---:|:---|
| Test suite | **51/51 PASS** | 12.75s, zero failures |
| End-to-end pipeline | **Working** | executor.py runs full worked example |
| 10-seed evaluation | **80% success** | 8/10 seeds pass, 0 collisions |
| Dynamic recovery | **77.8%** | 7/9 failures recovered autonomously |
| OpenVINO export | **Working** | IR model exported and loads |
| OpenVINO benchmark | **Working** | 1.936ms mean, 516.5 FPS on CPU |
| Demo frame recorder | **Working** | GIF generation verified |
| Dependencies | **Clean** | requirements.txt verified |
| README | **Comprehensive** | Architecture, commands, results |
| NOTICE.md | **Complete** | All attributions present |

**Verdict**: The project is functional and complete. The suggestions below focus on
**maximizing your score** across the five judging categories.

---

## 🏆 Judging Rubric Alignment

The challenge scores on 100 points:

| Category | Points | TwinGuard Current Status | Gap |
|:---|:---:|:---|:---|
| End-to-end task completion & bimanual manipulation | 30 | Strong (80% success, bimanual pour) | Minor |
| **VLA / multi-modal reasoning** | **20** | **LLM mock planner, no real VLA** | **Critical** |
| **OpenVINO & Core Ultra optimization** | **20** | **Basic IR export, CPU only, no INT8** | **Significant** |
| Robustness (10 seeds) | 15 | Good (80%, controlled failures shown) | Minor |
| Technical quality / reproducibility | 10 | Excellent (51 tests, full README) | None |
| Innovation | 5 | Decent (recovery loop is novel) | Minor |

**The two biggest scoring gaps are VLA/multi-modal reasoning (20 pts) and OpenVINO optimization (20 pts).**

---

## 🔴 CRITICAL — VLA / Multi-Modal Reasoning (20 points at stake)

### Problem
The planner currently runs a mock LLM provider that returns hardcoded plan steps.
The `provider: "mock"` setting in `sim_config.yaml` means **no real multi-modal
reasoning is happening**. Judges will immediately spot this — it's the single
biggest threat to your score.

### Suggested Fixes (pick at least one)

#### Option A: Wire Up a Real LLM Provider (2-3 hours)
1. In `planning/planner.py`, implement the `"gemini"` provider path to actually call
   the Gemini API (you already have `gemini-2.5-flash` configured).
2. Set `GEMINI_API_KEY` in your `.env` file.
3. Change `provider: "mock"` → `provider: "gemini"` in `sim_config.yaml`.
4. The Pydantic validation already handles bad outputs, so this is low-risk.
5. **Mention in README** that the planner can run either in mock mode (offline,
   deterministic) or live-LLM mode (online, multi-modal reasoning demonstrated).

#### Option B: Integrate SmolVLA (Ambitious, 4-6 hours)
1. Download Hugging Face's SmolVLA (450M params) — it's the VLA model specifically
   referenced in the challenge brief alongside LeRobot.
2. Use it as the policy backbone (takes camera image + language instruction →
   action chunks) instead of the LLM planner.
3. Export SmolVLA to OpenVINO IR — this alone would be a massive differentiator
   because you'd be benchmarking the *actual VLA* on Intel hardware, not just
   the object detector.
4. Even if accuracy is poor (no fine-tuning time), **having it integrated and
   benchmarked** demonstrates the full Physical AI pipeline.

#### Option C: Camera-Grounded LLM Reasoning (Minimum Viable, 1-2 hours)
1. When calling the planner, render a camera frame, run it through the neural
   object detector (already trained!), and include the detected bounding boxes
   + labels in the LLM prompt alongside the scene state text.
2. This makes the system genuinely **multi-modal** — language instruction + visual
   perception → structured plan.
3. Mention this explicitly: "The planner receives both ground-truth spatial
   coordinates and neural-vision bounding boxes, enabling multi-modal reasoning."

---

## 🔴 CRITICAL — OpenVINO & Core Ultra Optimization (20 points at stake)

### Problem
The current benchmark runs FP32 on CPU. The challenge specifically calls out
"Core Ultra optimization" including iGPU and NPU, and Intel provides NNCF
(Neural Network Compression Framework) for quantization. You're leaving points
on the table.

### Suggested Fixes

#### 1. INT8 Quantization with NNCF (High Impact, 1-2 hours)
```python
# Add to evaluation/openvino_export.py:
import nncf

# After loading the OpenVINO model:
quantized_model = nncf.quantize(ov_model, calibration_dataset)
ov.save_model(quantized_model, "checkpoints/openvino/object_detector_int8.xml")
```
- Generate a small calibration dataset (50-100 synthetic frames from the sim).
- Report INT8 vs FP32 latency comparison in the benchmark table.
- Add `nncf` to requirements.txt.

#### 2. Performance Hints (Quick Win, 30 minutes)
```python
# In benchmark.py, when compiling:
compiled_model = core.compile_model(model, device,
    config={"PERFORMANCE_HINT": "LATENCY"})  # or "THROUGHPUT"
```
- Report both LATENCY and THROUGHPUT mode numbers.

#### 3. Model Caching (Quick Win, 15 minutes)
```python
core.set_property({"CACHE_DIR": "checkpoints/openvino/cache"})
```
- Eliminates cold-start compilation latency on repeated runs.
- Judges will recognize this as production-ready practice.

#### 4. Multi-Device Benchmark Table (High Impact, 30 minutes)
- Even if you're on AMD hardware, add `--device GPU` and `--device NPU`
  flags and handle the graceful fallback/error message.
- In README, add a table showing: "CPU: X ms | GPU: (requires Intel iGPU) |
  NPU: (requires Intel Core Ultra NPU)".
- This shows the judge you designed for Core Ultra even if you tested on AMD.

#### 5. Async Inference (Differentiator, 1 hour)
```python
infer_request = compiled_model.create_infer_request()
infer_request.start_async(input_tensor)
# ... do other work (e.g., physics step) ...
infer_request.wait()
```
- Demonstrates understanding of real-time robotics perception pipelines.
- Report async vs sync throughput comparison.

---

## 🟡 IMPORTANT — Anomalib Integration (Innovation + VLA points)

### Problem
The challenge brief lists **Anomalib v2.6.0** as part of Intel's recommended stack.
Not using it at all means missing a chance to align with what judges expect.

### Suggested Fix (2-3 hours)
1. Train a lightweight Anomalib model (e.g., PaDiM or EfficientAd) on "normal"
   manipulation frames rendered during successful task execution.
2. During the executor loop, run each post-step camera frame through the
   anomaly detector.
3. If the anomaly score exceeds a threshold, trigger the dynamic recovery path.
4. **This replaces or supplements your geometry-based safety verifier with a
   learned, vision-based anomaly detector** — which is exactly what Intel's
   Physical AI Studio promotes.
5. Export the Anomalib model to OpenVINO IR and benchmark it alongside the
   object detector.

---

## 🟡 IMPORTANT — Demo Video (Presentation: 5-10 points)

### Problem
lablab.ai requires a **max 5-minute MP4 video** with: intro, slide deck, and
functional demo. Judges will watch this *before* reading your code.

### Suggested Actions

1. **Record the demo video NOW** — don't leave it for the last hour.
2. Structure (5 minutes max):
   - **0:00-0:30** — Problem intro: "Setting a dinner table with bimanual robots"
   - **0:30-1:30** — Architecture slide (use the ASCII diagram from README)
   - **1:30-3:00** — Live terminal demo:
     - `python evaluation/executor.py` (show full pipeline executing)
     - `python evaluation/report.py` (show results table)
     - `python evaluation/benchmark.py` (show OpenVINO numbers)
   - **3:00-4:00** — Show the GIF animations of seeds 2 and 7 (recovery path!)
   - **4:00-4:30** — Show the test suite passing (`pytest tests/ -v`)
   - **4:30-5:00** — Summary: "80% success, 0 collisions, 1.9ms inference,
     autonomous recovery demonstrated on controlled failures"
3. **Use OBS Studio** or similar to screen-record the terminal + slides.
4. **Show the recovery path explicitly** — "Watch seed 2: the grasp fails,
   TwinGuard re-observes, replans, and succeeds autonomously."

---

## 🟡 IMPORTANT — Submission Checklist for lablab.ai

- [ ] **Public GitHub repo** — Make sure the repo is PUBLIC before submission.
- [ ] **Cover image** — 16:9 PNG/JPG. Generate a banner showing two robot arms
      over a dinner table.
- [ ] **Short description** (≤255 chars): "TwinGuard: bimanual Physical AI system
      with closed-loop recovery, neural perception, and OpenVINO edge inference
      for Intel Core Ultra — demonstrated across 10 randomized seeds."
- [ ] **Long description** (≥100 words): Use the README's architecture overview.
- [ ] **Video** — MP4, ≤5 minutes.
- [ ] **Slide deck** — PDF with architecture diagram, results table, benchmark.
- [ ] **Live demo URL** — Not strictly needed for robotics, but consider hosting
      a Streamlit dashboard showing `results.json` interactively.
- [ ] **Technology tags** — Intel, OpenVINO, MuJoCo, PyTorch, Physical AI.

---

## 🟢 NICE-TO-HAVE — Polish Items

### 1. Increase Task Success Rate (80% → 90%+)
- Seeds 4 and 5 fail. Analyze why:
  - Are the randomized object positions pushing objects out of kinematic reach?
  - Consider widening the IK convergence tolerance or adding a pre-grasp
    alignment step.
  - Even a 10% improvement (1 more seed passing) matters for the robustness score.

### 2. Multiple Task Instructions
- Currently all 10 seeds run the same instruction. Add 2-3 rotating instructions:
  - "Pick up the mug with the right arm and pour into the plate."
  - "Open the drawer, retrieve the plate, and place it center-table."
- This demonstrates generalization, not just memorization of one sequence.

### 3. Bimanual Coordination (Hand-Off)
- The challenge brief mentions "hand off between arms." If one arm can pass
  the plate to the other arm, this is a strong differentiator for the
  bimanual manipulation score (30 pts).
- Even a simple "arm A holds plate, arm B takes it" sequence would count.

### 4. Force/Torque Feedback
- Add simulated force sensing (MuJoCo contact forces) to the verifier.
- Instead of only checking geometric distance for grasp verification, also
  check that contact forces exceed a minimum threshold.
- This is more physically realistic and judges will appreciate it.

### 5. Depth Camera Integration
- MuJoCo can render depth maps alongside RGB. Feed both to the object
  detector (4-channel input: RGBD).
- Mention "multi-modal perception: RGB + depth" in the README.

### 6. Add a `Dockerfile` or `devcontainer.json`
- Makes reproducibility trivial for judges:
  ```dockerfile
  FROM python:3.13-slim
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  CMD ["pytest", "tests/", "-v"]
  ```

### 7. Add a `LICENSE` File
- The project references Apache-2.0 in NOTICE.md but there's no explicit
  LICENSE file. Add one for the project itself.

---

## 📋 Priority-Ordered Action Plan

If you have limited time before the deadline (Sep 17, 01:00 UTC):

| Priority | Task | Time | Impact |
|:---:|:---|:---:|:---|
| **P0** | Wire up real Gemini LLM calls (Option A above) | 2-3h | **+10-15 pts** (VLA category) |
| **P0** | Record & upload demo video | 1-2h | **Required for submission** |
| **P1** | INT8 quantization + performance hints | 1-2h | **+5-10 pts** (OpenVINO category) |
| **P1** | Include camera frames in planner prompt (Option C) | 1-2h | **+5 pts** (multi-modal) |
| **P1** | Make GitHub repo public, add cover image & slide deck | 30m | **Required for submission** |
| **P2** | Debug seeds 4 & 5 failures → push to 90%+ success | 1-2h | **+3-5 pts** (robustness) |
| **P2** | Add Anomalib anomaly detection | 2-3h | **+3-5 pts** (innovation) |
| **P3** | SmolVLA integration | 4-6h | **+10-15 pts** (if time permits) |
| **P3** | Bimanual hand-off primitive | 2-3h | **+3-5 pts** (manipulation) |
| **P3** | Dockerfile + LICENSE file | 30m | **+1-2 pts** (reproducibility) |

---

## 🎯 Bottom Line

**TwinGuard is a solid, working project.** The architecture, test suite,
documentation, and recovery mechanism are genuinely impressive. But the two
areas where you're losing the most points are:

1. **No real VLA/LLM calls** — the mock provider bypasses the entire multi-modal
   reasoning category (20 pts). Fix this first.
2. **No quantization or hardware-specific optimization** — basic FP32 CPU benchmark
   doesn't show Intel Core Ultra value (20 pts). Add INT8 + performance hints.

Fixing just these two gaps could improve your score by 15-25 points.
