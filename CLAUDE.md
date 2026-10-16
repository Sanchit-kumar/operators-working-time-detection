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
- Real factory videos (up to 15 min each) are still to come. Until then the only data is a warehouse sample:
- `data/warehouse_high_resolution.mp4` (3840x2160) and `data/warehouse_low_resolution.mp4` (426x240) are the
  **same 17.1 s clip** (427 frames @ 25 fps). 240p is too low: tracks fragment. Use the 4K file.
- The sample camera is **not fixed** (factory CCTV is expected to be): it is an overhead, slowly moving (handheld/drone-style) shot. Raw bbox/pixel motion
  includes camera motion, so motion features must be camera-compensated or body-relative (pose-based).
- Scene: 2 workers in uniform (one pulling a pallet jack and loading buckets, one using a tablet), plus a passer-by
  at the frame edge near the end.

## Scope
- Individual workers are tracked by ID only (no identity recognition).
- States in scope: **Working** and **Idle**. Absent is out of scope for now.

## State definitions (source of truth — labels, rules and metrics all follow these)
- **Working**: worker is doing a task: handling/carrying items, pulling the pallet jack, walking with purpose
  between stations, operating equipment, **using the tablet/scanner (even when standing still)**, packing.
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
- Splits: first 60 % of each video = dev (tuning), last 40 % = test.
- `configs/live.yaml` extends `default.yaml` (configs support `extends:`).
- Benchmark with `python -m tracker.benchmark` and report numbers, never "looks good".
- Never tune thresholds on the test split.
