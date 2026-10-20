# Manual: running every module by hand

Copy-paste commands for each stage of the worker activity tracker, what it reads and writes, and every option.
For the background (how it works, results), see [README.md](README.md).

## 0. Every session

```bash
cd "/home/sanch/UbuntuCode/Python/Computer vision/worker-activity-tracker"
source .venv/bin/activate

# The video names contain spaces: keep them in variables and always quote them
A="data/vidssave.com Example of Hi-Definition Video Surveillance of a Factory Floor - by CCTVDOC.COM 1080p.mp4"
B="data/vidssave.com YTB_1792115108042 360P.mp4"
W="data/warehouse_high_resolution.mp4"          # moving-camera sample
```

Any new video: put it in `data/` and use its path the same way (`V="data/my video.mp4"`).

### Order of the modules

```
               ┌──────────────────────── motion classifier ────────────────────────┐
video ─► 1. detect/track (tracker.run) ─► 2. classify + report (tracker.run) ─► outputs/<video>/
               │                     └─► 3. zones (zone_tool, optional)
               ├─► 4. labels (label_tool) ─► 5. tune (tracker.tune) ─► 6. benchmark (tracker.benchmark)
               └─► 7. VLM predictions (tracker.vlm) ─► 2. classify + report with configs/vlm*.yaml
8. live view (tracker.live): detection + classification on a stream, on its own
```

Detection (stage 1) runs once per video and is cached; everything after it re-runs in seconds.

---

## 1 + 2. Detection, classification and reports: `tracker.run`

**Reads** a video. **Writes** `outputs/<video name>/`:

| File | Content |
|---|---|
| `detections.parquet`, `camera.parquet`, `meta.json` | detection cache (stage 1) |
| `summary.json` | working / idle seconds, utilisation %, per-worker totals |
| `timeline.csv` | `track_id, start_s, end_s, state` |
| `annotated.mp4` | boxes, skeletons, states, timers |

and `outputs/all_videos_summary.csv` when several videos are given.

```bash
python -m tracker.run "$A"                                   # default config (motion, untuned thresholds)
python -m tracker.run "$A" "$B" --config configs/tuned.yaml  # tuned motion thresholds (recommended)
python -m tracker.run "$A" "$B" --config configs/vlm.yaml    # zero-shot VLM classifier (see section 7)
python -m tracker.run data/*.mp4 --config configs/tuned.yaml # every video in data/
```

| Option | Meaning |
|---|---|
| `videos` | one or more video files |
| `--config FILE` | config to use (default `configs/default.yaml`) |
| `--redetect` | ignore the cache and run detection again (after changing the detector or tracker settings) |
| `--no-video` | skip writing `annotated.mp4` (faster) |
| `--stride N` | analyse every Nth frame (`2` halves detection time on long videos); implies `--redetect` |

Examples:

```bash
python -m tracker.run "$B" --config configs/tuned.yaml --stride 2       # 17-min video, ~2x faster detection
python -m tracker.run "$A" --redetect --no-video                        # fresh detection cache only
python -m tracker.run "$W" --config configs/moving_camera.yaml          # handheld / panning camera
```

Look at the results:

```bash
cat "outputs/$(basename "$A" .mp4)/summary.json"
head "outputs/$(basename "$A" .mp4)/timeline.csv"
cat outputs/all_videos_summary.csv
```

Speed (RTX 4060): detection ~7 fps with `yolo11x-pose` at 1920 px; classification seconds; the annotated video
takes about as long as the video plays.

---

## 3. Work zones (optional): `tools/zone_tool.py`

Only people whose feet are inside a zone are counted. Needs a display (laptop).

**Reads** a video. **Writes** `zones/<video name>.json` (or `zones/<name>.json`). No zone file = whole frame.

```bash
python tools/zone_tool.py "$B"                # draw on the frame at 2 s
python tools/zone_tool.py "$B" --at 300       # draw on the frame at 300 s
python tools/zone_tool.py "$B" --name cam1    # save as zones/cam1.json (one per camera; use with live --zones cam1)
```

| Key / mouse | Action |
|---|---|
| left click | add a point to the current zone |
| `enter` / `n` | close the zone (3+ points) |
| `u` | undo last point |
| `d` | delete last zone |
| `[` / `]` | frame 10 s earlier / later |
| `s` / `q` | save / save and quit |

Quitting with no zones deletes the file (whole frame counts). After drawing, re-run stage 2 (no re-detection):

```bash
python -m tracker.run "$B" --config configs/tuned.yaml --no-video
```

---

## 4. Ground-truth labels: `tools/label_tool.py`

Needs the detection cache (stage 1) and a display.

**Reads** the video and `outputs/<video>/detections.parquet`. **Writes** `labels/<video name>.json` (every edit)
and `labels/<video name>.csv` (on quit; read by the benchmark and tuner).

```bash
python -m tracker.run "$A" --no-video          # 1. detection cache, if not there yet
python tools/label_tool.py "$A"                # 2. label
python tools/label_tool.py "$B" --suggest      #    pre-drawn predictions to accept (c) or correct
```

| Option | Meaning |
|---|---|
| `--suggest` | show the model's predictions (top half of each lane) to accept or correct; uses `configs/tuned.yaml` |
| `--min-len S` | leave tracks shorter than S seconds out of the to-do list / `n` (default 2) |
| `--width PX` | window width (default 1152); smaller if the window doesn't fit |

| Key / mouse | Action |
|---|---|
| click a box or a lane name | select a worker |
| drag on a lane | label that time range with the current brush (fix anything, any time) |
| `w` / `i` / `x` | brush = working / idle / not a worker, and label from now until the next change |
| `W` / `I` / `X` | label the selected worker's whole track |
| `c` | accept the suggestion for the selected worker (`--suggest`) |
| `n` | next worker with unlabelled time |
| click the overview bar | jump in the video |
| `space`, `a`/`d`, `j`/`l`, `,`/`.` | play/pause, ±1 s, ±10 s, ±1 frame |
| `z`, `+`/`-` | lane zoom (20 s / 1 min / 5 min / all), playback speed |
| `u`, `h`, `q` | undo, help, save and quit |

Definitions: **working** = any task (desk/computer work, packing, carrying, operating, walking with purpose);
**idle** = visibly not working for 3 s or more; **x** = not a real worker (false box, passer-by).

---

## 5. Tune the motion thresholds: `tracker.tune`

Needs labels for every video given. **Reads** the caches and `labels/*.csv`. **Writes** `configs/tuned.yaml`
(overwrites it).

```bash
python -m tracker.tune "$A" "$B"
```

| Option | Meaning |
|---|---|
| `videos` | labelled videos |
| `--config FILE` | base config whose thresholds are searched (default `configs/default.yaml`) |

Fits on the first 60 % of each video's labelled time and prints the score on the last 40 % (never used for fitting).
Takes under a minute. Keep a copy if you want to compare: `cp configs/tuned.yaml configs/tuned_backup.yaml`.

---

## 6. Benchmark against labels: `tracker.benchmark`

**Reads** the caches, `labels/*.csv` and (VLM configs) the VLM predictions. **Writes** `outputs/benchmark_<split>.json`.

```bash
python -m tracker.benchmark "$A" "$B" --config configs/tuned.yaml --split test    # tuned motion, held-out part
python -m tracker.benchmark "$A" "$B" --config configs/vlm.yaml                   # zero-shot VLM, all labels
python -m tracker.benchmark "$A" --split test                                     # default thresholds
```

| Option | Meaning |
|---|---|
| `videos` | labelled videos |
| `--config FILE` | config to evaluate (default `configs/default.yaml`) |
| `--split all\|dev\|test` | which part of the labelled time: all (default), first 60 % (dev), last 40 % (test) |

Use `--split test` for anything tuned on these labels (motion `tuned.yaml`). The VLM is not tuned, so `--split all`
is a fair test for it.

Output: per-second accuracy, precision / recall / F1 per state, macro-F1, total working / idle time error,
missed seconds, non-workers counted, confusion matrix, per-worker table.

---

## 7. Zero-shot VLM classifier: `tracker.vlm`

Asks a vision-language model (Qwen3-VL) about each worker every 2 s. Needs the detection cache (created
automatically if missing). **Writes** `outputs/<video>/vlm_<model>.parquet` (one file per model) and, with
`--save-samples`, `outputs/<video>/vlm_samples/*.jpg` (what the model saw).

```bash
python -m tracker.vlm "$A" --config configs/vlm.yaml --limit 20 --save-samples 20   # quick check
python -m tracker.vlm "$A" "$B" --config configs/vlm.yaml                           # all queries
python -m tracker.run "$A" "$B" --config configs/vlm.yaml                           # reports from the predictions
python -m tracker.benchmark "$A" "$B" --config configs/vlm.yaml                     # accuracy
```

| Option | Meaning |
|---|---|
| `videos` | one or more videos |
| `--config FILE` | VLM config (default `configs/vlm.yaml`) |
| `--limit N` | only N queries spread over the video (quick check). Note: overwrites the predictions file for that model |
| `--save-samples N` | save the first N query images to `vlm_samples/` |

`tracker.run` with a VLM config creates the predictions automatically if they don't exist; run `tracker.vlm`
yourself to redo them (e.g. after a `--limit` check, or a prompt change in `tracker/vlm.py`).

| Config | Backend | Model | Where |
|---|---|---|---|
| `configs/vlm.yaml` | in-process | Qwen3-VL-2B-Instruct | laptop (8 GB GPU); pipeline check only, says "working" for everything |
| `configs/vlm_a100.yaml` | in-process | Qwen3-VL-8B-Instruct | A100; edit `vlm.model` for 32B on 80 GB |
| `configs/vlm_server.yaml` | vLLM server | Qwen3-VL-8B-Instruct | A100 server, or laptop through an SSH tunnel |

### VLM on the A100 machine (in-process)

```bash
ssh user@a100
cd ~/worker-activity-tracker && source .venv/bin/activate
tmux new -s vlm                                                  # survives SSH disconnects (tmux attach -t vlm)
python -m tracker.vlm "$A" --config configs/vlm_a100.yaml --limit 20 --save-samples 20
python -m tracker.run "$A" "$B" --config configs/vlm_a100.yaml
python -m tracker.benchmark "$A" "$B" --config configs/vlm_a100.yaml
```

### VLM through a vLLM server

```bash
# on the A100 (tmux window 1)
source ~/vllm-env/bin/activate
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}'

# on the laptop: tunnel (keep open), then check
ssh -N -L 8000:localhost:8000 user@a100
curl http://localhost:8000/v1/models

# then, on the laptop (or in A100 tmux window 2)
python -m tracker.vlm "$A" "$B" --config configs/vlm_server.yaml
python -m tracker.run "$A" "$B" --config configs/vlm_server.yaml
python -m tracker.benchmark "$A" "$B" --config configs/vlm_server.yaml
```

Setup of the A100 machine, copying code / videos / results, and troubleshooting: README.md, section 6.

---

## 8. Live view: `tracker.live`

Detection + classification on a file, IP camera or webcam, with an on-screen overlay. Independent of the cache.

```bash
python -m tracker.live "$A"                                        # fast model (configs/live.yaml)
python -m tracker.live "$B" --config configs/tuned.yaml            # accurate model + tuned thresholds (slower)
python -m tracker.live rtsp://user:pass@camera-ip/stream           # IP camera
python -m tracker.live 0                                           # webcam
python -m tracker.live "$B" --config configs/vlm_live.yaml         # VLM decisions in the background (local 2B)
python -m tracker.live "$B" --config configs/vlm_live_server.yaml  # VLM decisions from the A100 (SSH tunnel)
python -m tracker.live "$A" --save outputs/live.mp4 --no-show      # record without a window (also on a server)
```

| Option | Meaning |
|---|---|
| `source` | video file, `rtsp://` / `http://` stream, or webcam number |
| `--config FILE` | default `configs/live.yaml` (yolo11s-pose) |
| `--save FILE.mp4` | also write the annotated stream |
| `--no-show` | no window |
| `--display-width PX` | window width (default 1280) |
| `--zones NAME` | use `zones/NAME.json` (default: the video file's name; streams have none unless given) |

Keys: `q` quit, `space` pause, `d` show motion / VLM signals. Prints the totals and average FPS at the end.

---

## 9. Tests

```bash
pytest -q                     # 18 tests: engine, motion, zones, labels, VLM parsing
pytest -q tests/test_vlm.py   # one file
```

---

## Recipes

### A. New video, tuned motion classifier (no labelling)

```bash
V="data/my video.mp4"
python -m tracker.run "$V" --config configs/tuned.yaml
cat "outputs/$(basename "$V" .mp4)/summary.json"
```

### B. New video, zero-shot VLM (A100)

```bash
python -m tracker.vlm "$V" --config configs/vlm_a100.yaml --limit 10 --save-samples 10   # check it looks right
python -m tracker.run "$V" --config configs/vlm_a100.yaml
```

### C. Measure accuracy on a new video

```bash
python -m tracker.run "$V" --no-video                                   # detection cache
python tools/label_tool.py "$V" --suggest                               # label (laptop)
python -m tracker.benchmark "$V" --config configs/tuned.yaml --split test
python -m tracker.benchmark "$V" --config configs/vlm_a100.yaml         # on the A100
```

### D. Re-fit the motion thresholds after labelling more videos

```bash
python -m tracker.tune "$A" "$B" "$V"
python -m tracker.benchmark "$A" "$B" "$V" --config configs/tuned.yaml --split test
python -m tracker.run "$A" "$B" "$V" --config configs/tuned.yaml
```

### E. Start over for one video

```bash
rm -rf "outputs/$(basename "$V" .mp4)"          # removes cache, predictions, reports (labels and zones stay)
python -m tracker.run "$V" --config configs/tuned.yaml
```

## Configs at a glance

| File | Purpose |
|---|---|
| `configs/default.yaml` | base settings: detector (`yolo11x-pose`, 1920 px), motion, thresholds, zones, outputs |
| `configs/tuned.yaml` | thresholds fitted to the labels by `tracker.tune` |
| `configs/live.yaml` | live view: fast detector |
| `configs/moving_camera.yaml` | handheld / panning cameras |
| `configs/vlm.yaml`, `vlm_a100.yaml`, `vlm_server.yaml` | zero-shot VLM classifier: local 2B / A100 8B / vLLM server |
| `configs/vlm_live.yaml`, `vlm_live_server.yaml` | live view with VLM decisions: local / A100 server |
| `configs/botsort.yaml` | tracker settings |

Configs inherit with `extends: <file>.yaml` and only list what they change, so the easiest way to try a setting is
a small new file, e.g. `configs/my_test.yaml`:

```yaml
extends: tuned.yaml
classifier:
  idle_after_s: 5.0
```
