# Worker Activity Tracker

Measures how long each factory worker spends **Working** vs **Idle** in CCTV video, benchmarks the result
against human labels, and shows it live.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
# PyTorch must match the NVIDIA driver. This machine's driver supports CUDA 12.4:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Workflow

```bash
# 1. Detect + track + pose (GPU, cached in outputs/<video>/), classify, write the report and annotated video
python -m tracker.run data/*.mp4

# 2. Label ground truth (once per video). Mark state changes per worker with w / i; x = not a worker
python tools/label_tool.py data/video1.mp4

# 3. Tune thresholds on the dev split (first 60 % of each video), check them once on the test split
python -m tracker.tune data/*.mp4

# 4. Benchmark: report the held-out test split
python -m tracker.benchmark data/*.mp4 --config configs/tuned.yaml --split test

# 5. Final numbers per video with the tuned config
python -m tracker.run data/*.mp4 --config configs/tuned.yaml

# Live view (file, RTSP camera or webcam). Keys: q quit, space pause, d debug signals
python -m tracker.live data/video1.mp4 --config configs/tuned.yaml
python -m tracker.live rtsp://user:pass@camera/stream

# Tests
pytest -q
```

## Outputs (`outputs/<video>/`)

| File | Content |
|---|---|
| `summary.json` | total working / idle seconds, utilisation %, per-worker totals |
| `timeline.csv` | `track_id, start_s, end_s, state` segments |
| `annotated.mp4` | video with boxes, skeletons, states, running timers and debug signals |
| `detections.parquet`, `camera.parquet` | stage-1 cache (delete it or pass `--redetect` to recompute) |

## How it works

1. **YOLO11-pose** detects every person and their 17 body keypoints. **BoT-SORT** (with ReID and camera-motion
   compensation) gives each person a stable track ID.
2. Each frame, the global camera motion is estimated from the background and removed, then three signals are
   measured per worker:
   - **motion energy**: fraction of pixels inside the person's box that changed
   - **wrist speed** relative to the body (hand work at a station)
   - **body speed** (walking, carrying)
   All three are normalised by body size, so they work for near and far workers.
3. Signals are median-filtered over a 2 s window. A worker is *working* if any signal exceeds its threshold.
   A state machine with hysteresis (idle only after 3 s of low activity) removes flicker. Switches are applied
   retroactively, so totals aren't delayed by the hold time.
4. Tracks shorter than 1.5 s (passers-by, false detections) are ignored.

All thresholds are in [configs/default.yaml](configs/default.yaml). `tracker.tune` fits them to your labels.
