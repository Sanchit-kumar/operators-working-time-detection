"""Batch processing of one or more videos.

    python -m tracker.run data/*.mp4                  # detect (cached) + classify + report
    python -m tracker.run data/x.mp4 --redetect       # force re-running the GPU stage
    python -m tracker.run data/x.mp4 --no-video       # skip writing the annotated video

Outputs per video in outputs/<video>/: detections.parquet, camera.parquet (cache), timeline.csv, summary.json,
annotated.mp4.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from .config import load_config, output_dir
from .detect import Detector, cam_row, iter_video, load_cache, save_cache, to_rows
from .engine import ActivityEngine
from .render import draw


def video_info(path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    info = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    cap.release()
    return info


def detect_stage(cfg, video: Path, out: Path) -> None:
    fps, n, w, h = video_info(video)
    stride = cfg["detector"]["vid_stride"]
    det = Detector(cfg)
    rows, cams = [], []
    start = time.time()
    for idx, t, frame in tqdm(iter_video(video, stride), total=n // stride, desc=f"detect {video.name}"):
        d = det(frame, idx, t)
        rows += to_rows(d)
        cams.append(cam_row(d))
    elapsed = time.time() - start
    save_cache(out, rows, cams, dict(video=str(video), fps=fps, frames=n, width=w, height=h, stride=stride,
                                     detect_fps=round(len(cams) / elapsed, 1)))


def classify_stage(cfg, video: Path, out: Path, save_video: bool) -> dict:
    meta = json.loads((out / "meta.json").read_text())
    engine = ActivityEngine(cfg, frame_dt=meta["stride"] / meta["fps"])
    writer = frames = None
    if save_video:
        scale = min(1.0, 1280 / meta["width"])
        size = (int(meta["width"] * scale), int(meta["height"] * scale))
        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 meta["fps"] / meta["stride"], size)
        frames = iter_video(video, meta["stride"])
    for d in load_cache(out):
        visible = engine.update(d)
        if writer:
            _, _, frame = next(frames)
            view = cv2.resize(frame, size)
            writer.write(draw(view, visible, engine, debug=True, scale=scale))
    if writer:
        writer.release()

    pd.DataFrame(engine.timeline(), columns=["track_id", "start_s", "end_s", "state"]).to_csv(
        out / "timeline.csv", index=False)
    summary = dict(video=video.name, duration_s=round(meta["frames"] / meta["fps"], 1),
                   detect_fps=meta.get("detect_fps"), **engine.summary())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--config", default=None)
    p.add_argument("--redetect", action="store_true")
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config) if a.config else load_config()

    results = []
    for video in a.videos:
        out = output_dir(cfg, video)
        if a.redetect or not (out / "detections.parquet").exists():
            detect_stage(cfg, video, out)
        s = classify_stage(cfg, video, out, cfg["output"]["save_video"] and not a.no_video)
        results.append(s)
        print(f"\n{video.name}: {s['workers']} workers | working {s['working_s']}s | idle {s['idle_s']}s | "
              f"utilisation {s['working_pct']}%")
        for tid, tot in s["per_worker"].items():
            print(f"  worker #{tid}: working {tot['working']}s  idle {tot['idle']}s")

    if len(results) > 1:
        df = pd.DataFrame([{k: r[k] for k in ("video", "duration_s", "workers", "working_s", "idle_s", "working_pct")}
                           for r in results])
        df.to_csv(Path(cfg["output"]["dir"]) / "all_videos_summary.csv", index=False)
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
