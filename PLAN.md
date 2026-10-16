# Project Plan — Worker Activity Tracker

Goal: for each **factory worker** in CCTV videos (several videos, up to 15 min each), output a timeline and
totals of **Working / Idle**, prove its accuracy against ground truth, and render it live.
Absent and identity recognition are out of scope for now (workers are distinguished by track ID only).
Status: pipeline v1 (rule-based) implemented — see README.md for the workflow.

## Pipeline

```
video ─► person detection ─► multi-object tracking ─► pose keypoints ─► per-track motion features
      ─► activity classifier (per second) ─► temporal smoothing ─► timeline + totals ─► live overlay / dashboard
```

| Stage | Choice | Why |
|---|---|---|
| Detection + pose | Ultralytics YOLO11 pose (`yolo11m-pose`; `s` if FPS is short) | One model gives boxes + 17 keypoints; fast on an RTX 4060 in FP16 |
| Tracking | BoT-SORT with ReID (built into Ultralytics), ByteTrack as the fallback | Keeps worker IDs stable through occlusion |
| Features (per track, 1 s windows) | Wrist/elbow keypoint speed, body-centre displacement, bbox area change, pose variance, zone membership | Working vs idle is mostly about *hand/body motion*, not appearance |
| Classifier v1 | Rule-based thresholds + hysteresis | Baseline in a day, easy to explain, gives the first benchmark number |
| Classifier v2 | Small temporal model (GRU / 1D-CNN) over keypoint sequences, trained on the labels | Learns cases the rules miss (e.g. small hand movements while scanning) |
| Optional v3 | Video action model (VideoMAE / X-CLIP) fine-tuned on person crops | Only if v2 plateaus; costs more compute and needs more labels |
| Smoothing | Minimum segment duration + hysteresis (e.g. idle only after 5 s of low motion) | Removes flicker; matches how a human would label |

## Milestones

1. **Setup and look at the data**: venv, video metadata, sample frames, count workers, decide whether work zones (ROIs) are needed.
2. **Write the state definitions** in `CLAUDE.md` (the rules for edge cases).
3. **Labelling tool + ground truth**: a small OpenCV tool where keys mark W/I/A per worker while the video plays, saved to `labels/*.csv`.
   Label the whole video at 1 s resolution. Split by time: first ~60% for dev/tuning, last ~40% held out for test.
4. **Detection + tracking**: run it, check ID switches visually, save tracks to `outputs/`.
5. **Rule-based classifier + benchmark v1**: get the first numbers.
6. **Error analysis**: confusion matrix and a list of the worst segments, played back so you can watch the failures. Then fix rules and features.
7. **Learned classifier (v2)** if v1 is below target.
8. **Real-time app**: live overlay (boxes, ID, state colour, running timers) plus an optional Streamlit dashboard. Works on a file or an RTSP stream.
9. **Final report**: metrics table, per-worker totals vs GT, FPS, known limitations.

## Benchmark metrics

All metrics are computed per second against `labels/*.csv`, on the **held-out test split**.

- **Per-second accuracy and macro-F1** over {Working, Idle, Absent}, plus a confusion matrix.
- **Total-time error** (the number the task is asking for): for each worker and state, |pred − GT| seconds and % of GT.
- **Segment metrics**: segmental F1@IoU 0.5 and a count of over-segmentation (flicker).
- **Tracking sanity**: ID switches per worker (counted by hand from the overlay video if no box labels exist).
- **Throughput**: end-to-end FPS on the RTX 4060 for both resolutions.

Suggested targets: macro-F1 ≥ 0.90, total-time error ≤ 5% per state, real-time at ≥ 25 FPS.

## Layout

```
configs/default.yaml
tracker/            # package: detect_track.py, features.py, classify.py, smooth.py, benchmark.py, live.py
tools/label_tool.py
labels/             # ground truth CSVs
outputs/            # generated, git-ignored
notebooks/          # error analysis only
tests/
```

## Risks
- **Unclear definitions** cause most of the "accuracy" errors. Fix them in writing first.
- **Little data** (2 videos): split by time, don't over-tune, and report test numbers only.
- **Occlusion by racks or forklifts** causes ID switches and false Absent. Mitigate with ReID and the absent grace period.
- **Low-resolution video**: pose keypoints may be unreliable, so fall back to bbox/optical-flow motion features.
