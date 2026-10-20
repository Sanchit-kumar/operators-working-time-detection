# Worker Activity Tracker

Measure per-worker **Working / Idle** time from factory CCTV video (several videos, up to 15 min each), benchmark
it against human-labelled ground truth, and show it live. See README.md for the workflow, PLAN.md for the roadmap.

## Environment
- Python 3.12, WSL2 Ubuntu, NVIDIA RTX 4060 Laptop (8 GB). Use the GPU (CUDA, FP16).
- Virtualenv at `.venv/` (`source .venv/bin/activate`). Dependencies in `requirements.txt`.
- The NVIDIA driver only supports CUDA 12.4: PyTorch must be the cu124 build (torch 2.6.0). A plain
  `pip install torch` pulls a CUDA 13 build that silently runs on CPU. Don't let pip upgrade torch.
- Network is slow (~0.8 MB/s): large downloads take tens of minutes, run them in the background.
- Raw videos live in `data/` and are never modified. Generated output goes in `outputs/` (git-ignored).

## Data facts
- Videos in `data/` are named `videoN.mp4` (`tools/rename_videos.py`); original names are in `videos.csv`.
- `video1.mp4`: factory floor, fixed CCTV, 1920x1080, **6 fps**, 17 s, ~9 workers in hi-vis at packing tables,
  small/far (40-60 px). Fine hand work seen from behind gives ~zero motion signal: motion rules under-count work.
- `video2.mp4`: parcel-sorting room, fixed CCTV, 640x360, 25 fps, 17.5 min, desk workers at computers + people
  walking between bins. Low-res: weak detections (conf 0.2-0.5), seated people missed, a blue sack pile on the left
  is sometimes detected as a person. Labelled 0-188 s only.
- `video3.mp4` (3840x2160) and `video4.mp4` (426x240): the **same 17.1 s warehouse clip** from a **moving** overhead
  camera (use `configs/moving_camera.yaml`); 240p is too low (tracks fragment). 2 workers + a passer-by.
- Labels exist for video1 and video2.

## Scope
- Individual workers are tracked by ID only (no identity recognition).
- States in scope: **Working** and **Idle**. Absent is out of scope for now.

## State definitions (source of truth — labels, rules and metrics all follow these)
- **Working**: worker is doing **any kind of task** (user decision, 2026-10-16): computer/desk work, packing,
  sorting, handling/carrying items, pulling a pallet jack, walking with purpose between stations, operating
  equipment, using a tablet/scanner (even when standing or sitting still).
- **Idle**: worker is visible but not doing a task for >= `classifier.idle_after_s` (standing with empty hands, sitting,
  on a personal phone, chatting, waiting).
- Ambiguous cases are decided here, in writing, before labelling. Update this section when a rule changes.

## Conventions
- All thresholds, model names and paths live in `configs/*.yaml`, not in code.
- Every pipeline stage can run offline on a file and writes its output (JSON/CSV/parquet) so stages are re-runnable.
- Ground truth (written by `tools/label_tool.py`): `labels/<video>.json` (change-points, editable) and
  `labels/<video>.csv` with one row per worker per second: `second,track_id,x1,y1,x2,y2,state`
  (`state` in working/idle/ignore). The benchmark matches GT to predictions by box IoU, not track id.
- Results: `outputs/<video>/timeline.csv` (`track_id,start_s,end_s,state`) and `summary.json`.
- Splits: first 60 % of each video's labelled time span = dev (tuning), last 40 % = test.
- An empty zone file means no restriction (whole frame). Classifier = rule-based thresholds (tracker/engine.py), tuned by grid search.
- `configs/live.yaml` / `moving_camera.yaml` extend `default.yaml` (configs support `extends:`).
- Work zones: `zones/<video stem or camera name>.json`, normalised polygons, drawn with `tools/zone_tool.py`.
  A person counts only when the bottom-centre of their box is inside a zone. No file = whole frame.
- Motion features are measured over `motion.interval_s` (0.2 s), never frame-to-frame, so they are fps-independent.
- Benchmark with `python -m tracker.benchmark` and report numbers, never "looks good".
- Never tune thresholds on the test split.
