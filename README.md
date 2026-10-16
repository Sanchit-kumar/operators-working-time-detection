# Worker Activity Tracker

Measures how long each factory worker spends **Working** vs **Idle** in CCTV video, benchmarks the result
against human labels, and shows it live.

- **Working:** any task (desk/computer work, packing, sorting, carrying, operating equipment, walking with purpose).
- **Idle:** visibly not doing a task for 3 s or more.

## How it works

| Stage | What | Where |
|---|---|---|
| 1. Detection + pose | **YOLO11x-pose** (batch) / YOLO11s-pose (live): person boxes + 17 body keypoints | `tracker/detect.py` |
| 2. Tracking | **BoT-SORT** with ReID: a stable ID per worker | `configs/botsort.yaml` |
| 3. Motion features | per worker, measured over 0.2 s (fps-independent): pixel **motion energy** in the box, **wrist speed** relative to the body, **body speed** | `tracker/motion.py`, `tracker/engine.py` |
| 4a. Classifier (motion) | **rule-based threshold classifier**: *working* if any feature (2 s median) exceeds its threshold, else *idle*; a hysteresis state machine removes flicker | `tracker/engine.py` |
| 4b. Classifier (VLM, zero-shot) | **Qwen3-VL** looks at each worker every 2 s (context frame + 4 crops) and answers working / idle / not-a-person from the written definitions; no training or tuning | `tracker/vlm.py` |
| 5. Tuning | grid search of the thresholds against human labels (first 60 % of each video's labelled time), reported on the last 40 % | `tracker/tune.py` |

Two classifiers, chosen by config: the **motion** classifier (thresholds fitted to your labels, `configs/tuned.yaml`)
and the **zero-shot VLM** classifier (`configs/vlm*.yaml`, section 6), which needs no labels or tuning.

Current tuned thresholds (`configs/tuned.yaml`, fitted on videos A + B): energy 0.02, wrist 0.05, body 0.05
body-heights/s, idle after 5 s, minimum track 3 s, minimum mean detection confidence 0.2.

## Setup (once)

```bash
cd "/home/sanch/UbuntuCode/Python/Computer vision/worker-activity-tracker"
python3 -m venv .venv && source .venv/bin/activate
# PyTorch must match the NVIDIA driver. This machine's driver supports CUDA 12.4:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Every session afterwards:

```bash
cd "/home/sanch/UbuntuCode/Python/Computer vision/worker-activity-tracker"
source .venv/bin/activate
```

Video names used below (they contain spaces, so keep the quotes):

```bash
A="data/vidssave.com Example of Hi-Definition Video Surveillance of a Factory Floor - by CCTVDOC.COM 1080p.mp4"
B="data/vidssave.com YTB_1792115108042 360P.mp4"
```

## 1. Run inference (batch): reports + annotated videos

```bash
python -m tracker.run "$A" "$B" --config configs/tuned.yaml              # tuned thresholds (recommended)
python -m tracker.run "$A" "$B"                                          # default thresholds
python -m tracker.run "$B" --config configs/tuned.yaml --stride 2        # every 2nd frame: ~2x faster on long videos
python -m tracker.run "$A" --redetect                                    # force re-running detection (GPU stage)
python -m tracker.run "$A" --no-video                                    # skip writing the annotated video
python -m tracker.run data/x.mp4 --config configs/moving_camera.yaml     # handheld / panning cameras
```

Detection is cached per video, so re-running with other thresholds takes seconds. Speed on the RTX 4060:
~7 fps with YOLO11x-pose at 1920 px (a 17-min 360p video with `--stride 2` takes ~30 min the first time).

Outputs in `outputs/<video name>/`:

| File | Content |
|---|---|
| `summary.json` | total working / idle seconds, utilisation %, per-worker totals |
| `timeline.csv` | `track_id, start_s, end_s, state` segments |
| `annotated.mp4` | boxes, skeletons, states, running timers, motion signals |
| `detections.parquet`, `camera.parquet`, `meta.json` | detection cache |

`outputs/all_videos_summary.csv` has one row per video when several are run together.

## 2. Real-time view

```bash
python -m tracker.live "$A"                                      # video file (configs/live.yaml: fast model)
python -m tracker.live "$A" --config configs/tuned.yaml          # accurate model + tuned thresholds (slower)
python -m tracker.live rtsp://user:pass@camera-ip/stream         # IP camera
python -m tracker.live 0                                         # webcam
python -m tracker.live "$A" --save outputs/live.mp4 --no-show    # record without a window
python -m tracker.live rtsp://camera/stream --zones line3        # use zones/line3.json
```

Keys: `q` quit, `space` pause, `d` show motion signals. To make `live.yaml` use the tuned thresholds, change
its first line to `extends: tuned.yaml`.

## 3. Work zones (optional, fixed cameras)

Only people whose feet are inside a zone are counted. Use it to exclude walkways, offices, or objects that get
mis-detected as people.

```bash
python tools/zone_tool.py "$B"               # draw on the video -> zones/<video name>.json
python tools/zone_tool.py "$B" --at 300      # draw on the frame at 300 s
python tools/zone_tool.py "$B" --name cam1   # one zone file per camera: zones/cam1.json
```

Keys: click = add point, `enter` = close zone, `u` undo point, `d` delete zone, `[` `]` move 10 s, `s` save,
`q` save and quit. Then re-run step 1 (no re-detection needed).

## 4. Label ground truth

Needs the detection cache from step 1.

```bash
python tools/label_tool.py "$A"
python tools/label_tool.py "$B" --suggest               # pre-draws model predictions to accept / correct
python tools/label_tool.py "$B" --suggest --min-len 5   # to-do list only for tracks >= 5 s
python tools/label_tool.py "$A" --width 1024            # smaller window
```

| Key / mouse | Action |
|---|---|
| click a box / lane name | select a worker |
| drag on a lane | label that time range with the current brush (go back and fix anything) |
| `w` / `i` / `x` | brush = working / idle / not-a-worker, and label from now until the next change |
| `W` / `I` / `X` | label the selected worker's whole track |
| `c` | accept the model's suggestion for the selected worker (`--suggest`) |
| `n` | jump to the next worker with unlabelled time |
| click the overview bar | jump anywhere in the video |
| `space`, `a`/`d`, `j`/`l`, `,`/`.` | play/pause, ±1 s, ±10 s, ±1 frame |
| `z`, `+`/`-` | zoom lanes (20 s / 1 min / 5 min / all), playback speed |
| `u`, `h`, `q` | undo, help, save and quit |

Edits save immediately to `labels/<video>.json`; `labels/<video>.csv` (used by the benchmark) is written on quit.
Suggestions never count until accepted. The last 40 % of each video's labelled time is the
test split, so label continuous stretches.

## 5. Tune and benchmark

```bash
python -m tracker.tune "$A" "$B"                                                 # fit thresholds -> configs/tuned.yaml
python -m tracker.benchmark "$A" "$B" --config configs/tuned.yaml --split test   # held-out accuracy
python -m tracker.benchmark "$A" "$B" --config configs/tuned.yaml                # whole videos
python -m tracker.benchmark "$A" --split test                                    # default thresholds, one video
```

Only labelled videos can be passed to `tune` / `benchmark`. Reported: per-second accuracy, precision / recall / F1
per state, macro-F1, confusion matrix, and the headline **total working / idle time error** (overall and per
worker). Results are also saved to `outputs/benchmark_<split>.json`.

## 6. Zero-shot VLM classifier (Qwen3-VL)

Every 2 s, for every tracked person, the VLM gets one full frame with the person in a red box plus 4 close-up
crops from the last 2 s, and answers `{"state": working | idle | not_a_person, "confidence", "reason"}`. The
definitions live in the prompt (`tracker/vlm.py`, `USER_PROMPT`). `not_a_person` removes false detections (e.g.
the sack pile in video B) without zones. Detection and tracking are the same as before (the cache is reused).

| Config | Where | Model |
|---|---|---|
| `configs/vlm.yaml` | this laptop (8 GB GPU), in-process | Qwen3-VL-2B-Instruct (4.3 GB) |
| `configs/vlm_a100.yaml` | A100, in-process (simplest) | Qwen3-VL-8B-Instruct (17.5 GB); 32B on an 80 GB card |
| `configs/vlm_server.yaml` | A100 + vLLM server (fastest) | Qwen3-VL-8B-Instruct |
| `configs/vlm_live.yaml` | live view + background VLM, in-process | Qwen3-VL-2B-Instruct |
| `configs/vlm_live_server.yaml` | live view on the laptop, VLM on the A100 (SSH tunnel) | Qwen3-VL-8B-Instruct via vLLM |

```bash
# quick check: 20 queries spread over the video, and save what the model sees to outputs/<video>/vlm_samples/
python -m tracker.vlm "$A" --config configs/vlm.yaml --limit 20 --save-samples 20

# full run: VLM predictions (if missing) + report + annotated video (overlay shows "vlm: <state>")
python -m tracker.run "$A" "$B" --config configs/vlm.yaml

# re-run only the VLM (e.g. after changing the prompt); predictions are stored per model name
python -m tracker.vlm "$A" "$B" --config configs/vlm.yaml

# live view with VLM decisions computed in the background (video never waits; motion covers gaps)
python -m tracker.live "$B" --config configs/vlm_live.yaml

# benchmark against your labels: no tuning involved, so all labelled data is a fair test
python -m tracker.benchmark "$A" "$B" --config configs/vlm.yaml
python -m tracker.benchmark "$A" "$B" --config configs/tuned.yaml        # motion classifier, for comparison
```

### Using the A100 over SSH

Replace `user@a100` with your login (or add a `Host a100` entry to `~/.ssh/config`). Two ways to use it:

| | Option A: everything on the A100 | Option B: A100 as a model server (SSH tunnel) |
|---|---|---|
| Runs on the A100 | detection + VLM + reports | only `vllm serve` |
| Runs on the laptop | nothing (copy results back) | detection, labelling, reports, **live view** |
| Videos leave the laptop | yes (copied to the A100) | only the crops sent per query |
| Best for | batch benchmark of many / long videos | live demo, quick experiments |

#### Option A: run everything on the A100

On the laptop, copy the code (without the venv, caches or weights) and the videos:

```bash
cd "/home/sanch/UbuntuCode/Python/Computer vision/worker-activity-tracker"
rsync -av --exclude .venv --exclude outputs --exclude data --exclude '*.pt' ./ user@a100:~/worker-activity-tracker/
rsync -av data/ user@a100:~/worker-activity-tracker/data/
# optional: reuse the detections already computed here, so the A100 skips the detection stage
rsync -avm --include '*/' --include 'detections.parquet' --include 'camera.parquet' --include 'meta.json' \
      --exclude '*' outputs/ user@a100:~/worker-activity-tracker/outputs/
```

On the A100 (`ssh user@a100`), one-time setup:

```bash
cd ~/worker-activity-tracker
python3 -m venv .venv && source .venv/bin/activate
nvidia-smi                                         # note the CUDA version the driver supports
pip install torch torchvision                      # or the matching build, e.g. --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
export HF_HOME=/path/with/space/hf                 # optional: models are 17.5 GB (8B) / 67 GB (32B)
```

Run (inside `tmux` or `screen`, so a dropped SSH connection doesn't stop it):

```bash
tmux new -s vlm                                    # re-attach later with: tmux attach -t vlm
source .venv/bin/activate
A="data/vidssave.com Example of Hi-Definition Video Surveillance of a Factory Floor - by CCTVDOC.COM 1080p.mp4"
B="data/vidssave.com YTB_1792115108042 360P.mp4"

python -m tracker.vlm "$A" --config configs/vlm_a100.yaml --limit 20 --save-samples 20   # quick check
python -m tracker.run "$A" "$B" --config configs/vlm_a100.yaml                          # reports + annotated videos
python -m tracker.benchmark "$A" "$B" --config configs/vlm_a100.yaml                    # accuracy vs labels
python -m tracker.benchmark "$A" "$B" --config configs/tuned.yaml                       # motion classifier, to compare
```

For the 32B model on an 80 GB A100, set `vlm.model: Qwen/Qwen3-VL-32B-Instruct` in `configs/vlm_a100.yaml`.

For more speed, use vLLM on the A100 instead of the in-process model (separate venv recommended):

```bash
python3 -m venv ~/vllm-env && source ~/vllm-env/bin/activate && pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}'   # tmux window 1
# tmux window 2, in the project venv:
python -m tracker.run "$A" "$B" --config configs/vlm_server.yaml
python -m tracker.benchmark "$A" "$B" --config configs/vlm_server.yaml
```

Copy results back to the laptop:

```bash
rsync -avm --include '*/' --include 'summary.json' --include 'timeline.csv' --include 'vlm_*.parquet' \
      --include 'annotated.mp4' --include '*.json' --include '*.csv' --exclude '*' \
      user@a100:~/worker-activity-tracker/outputs/ outputs/
```

#### Option B: A100 as a model server through an SSH tunnel

On the A100 (in `tmux`), start the server. It listens only on the A100's localhost; the tunnel carries the traffic:

```bash
source ~/vllm-env/bin/activate
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}'
```

On the laptop, open the tunnel (keep this terminal open) and check it:

```bash
ssh -N -L 8000:localhost:8000 user@a100
curl http://localhost:8000/v1/models               # in another terminal: should list Qwen/Qwen3-VL-8B-Instruct
```

Then run everything on the laptop as usual; the VLM queries go to the A100:

```bash
python -m tracker.run "$A" "$B" --config configs/vlm_server.yaml          # batch reports
python -m tracker.benchmark "$A" "$B" --config configs/vlm_server.yaml    # accuracy vs labels
python -m tracker.live "$B" --config configs/vlm_live_server.yaml         # live view with A100 decisions
```

If port 8000 is taken on the laptop, use `-L 8001:localhost:8000` and set `vlm.base_url: http://localhost:8001/v1`.
If the server runs a different model, set `vlm.model` in the config to the same name (`curl .../v1/models` shows it).

## 7. Tests

```bash
pytest -q
```

## Results so far

Held-out test: the last 40 % of each video's labelled time (never used for tuning), 129 labelled worker-seconds
(video A: 17 s clip, 9 workers; video B: 360p, labelled 0-188 s).

| | Default thresholds | Tuned on A + B |
|---|---|---|
| Per-second accuracy | 0.17 | **0.73** |
| Working F1 | 0.14 | **0.83** |
| Working-time error | 87 % | **9.9 %** |
| Idle-time error | 39 % | 33 % (only 18 idle seconds in the test set) |
| Missed worker-seconds | 90 | 17 |

Final inference with `configs/tuned.yaml`:

| Video | Workers (tracks) | Working | Idle | Utilisation |
|---|---|---|---|---|
| A: 1080p, 6 fps, 17 s | 10 | 80.0 s | 28.8 s | 73.5 % (labels: 62.5 %) |
| B: 360p, 25 fps, 17.5 min | 149 | 1533.8 s | 674.3 s | 69.5 % |

These numbers come from a small test set (129 worker-seconds, 18 of them idle), so treat them as preliminary.

Zero-shot VLM, Qwen3-VL-2B (laptop, video A, 120 labelled worker-seconds): answered "working" for all 54 queries
(accuracy 0.59, idle recall 0). The pipeline works end to end, but 2B is too small to separate working from idle.
The 8B / 32B models on the A100 are the real test (section 6).
More labelled video, especially idle periods, is the most effective way to raise and verify accuracy.

Known limitations: low-resolution cameras (360p) miss seated workers behind monitors, and tracks fragment into
many IDs (totals are still valid, per-worker splits are not). In video B a pile of blue sacks on the left is
sometimes detected as a person; draw a work zone that excludes it (section 3). Workers doing fine hand work with their back to the
camera move very little and can be classified idle.

## Configs

| File | Use |
|---|---|
| `configs/default.yaml` | fixed factory cameras, accurate model (`yolo11x-pose`, 1920 px), batch |
| `configs/tuned.yaml` | written by `tracker.tune` from your labels |
| `configs/live.yaml` | real-time view: fast model (`yolo11s-pose`) |
| `configs/moving_camera.yaml` | handheld / panning cameras (camera-motion compensation on) |
| `configs/botsort.yaml` | tracker settings |

Configs can inherit with `extends: default.yaml` and override only what differs.
