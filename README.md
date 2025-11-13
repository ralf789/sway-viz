# Sway Visualiser — single‑file 

A small tool to **visualise postural sway** and compute a simple **Centre‑of‑Mass (CoM)** stability metric from either a **live webcam** or a **video file**. Everything runs from a single Python file: `sway_viz_v1.py`.

> ⚠️ Research/visual feedback only — not a medical device.

---

## What this does

- Uses **MediaPipe Pose** to estimate 3D world landmarks (metres).
- Approximates **Centre of Mass** as the **mean of all world landmarks** (simple proxy).
- Lets you choose the sway axis:
  - `lr` → mediolateral (world **X**) → labelled **ML**
  - `bf` → anteroposterior (world **Z**) → labelled **AP**
- Records that axis for a set duration, computes **SD (≈ RMS)** in **cm**, and compares it to stance‑specific **95th‑percentile cut‑offs**.
- Draws a **line during recording** so you can *see* the sway path.

---

## Requirements

- **Python** 3.9–3.12
- **pip** (comes with Python)
- A webcam (for live mode) or a readable video file (for file mode)

Install the libraries:
```bash
pip install opencv-python mediapipe numpy
