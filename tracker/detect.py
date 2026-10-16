"""Stage 1: person detection + tracking + pose + motion energy (GPU, expensive, cached)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .motion import MotionEstimator

N_KPTS = 17


@dataclass
class FrameDetections:
    frame_idx: int
    t: float                  # seconds from video start
    ids: np.ndarray           # (N,) int track ids
    boxes: np.ndarray         # (N, 4) x1, y1, x2, y2
    confs: np.ndarray         # (N,)
    kpts: np.ndarray          # (N, 17, 3) x, y, conf (COCO order)
    energy: np.ndarray        # (N,) camera-compensated motion energy inside each box
    affine: np.ndarray        # (2, 3) previous -> current frame camera transform


class Detector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # imported lazily: stage 2 and tests don't need the GPU stack

        d = cfg["detector"]
        self.model = YOLO(d["model"])
        self.track_kwargs = dict(persist=True, tracker=d["tracker"], imgsz=d["imgsz"], conf=d["conf"],
                                 device=d["device"], quantize=d["quantize"], classes=[0], verbose=False)
        self.imgsz = d["imgsz"]
        m = cfg["motion"]
        self.motion = MotionEstimator(m["work_width"], m["diff_threshold"])

    def __call__(self, frame: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        # The model letterboxes to imgsz anyway; resizing first also keeps the tracker's camera-motion
        # compensation and ReID off the full-resolution (e.g. 4K) frame, which is much faster.
        s = min(1.0, self.imgsz / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) if s < 1 else frame
        r = self.model.track(small, **self.track_kwargs)[0]
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy() / s
            confs = r.boxes.conf.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(ids), N_KPTS, 3))
            kpts[..., :2] /= s
        else:
            ids, boxes, confs = np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)
            kpts = np.zeros((0, N_KPTS, 3))
        affine, energy = self.motion.update(small, boxes * s)
        affine[:, 2] /= s  # translation back to full-resolution pixels
        return FrameDetections(frame_idx, t, ids, boxes, confs, kpts, energy, affine)


def iter_video(path: str | Path, vid_stride: int = 1):
    """Yield (frame_idx, t, frame) for every vid_stride-th frame."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % vid_stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield idx, idx / fps, frame
        idx += 1
    cap.release()


# ---- cache (so stage 2 can be re-tuned without re-running the GPU) ----

KPT_COLS = [f"k{i}_{c}" for i in range(N_KPTS) for c in "xyc"]
AFF_COLS = [f"a{i}" for i in range(6)]


def to_rows(d: FrameDetections) -> list[dict]:
    rows = []
    for i in range(len(d.ids)):
        row = dict(frame=d.frame_idx, t=d.t, track_id=int(d.ids[i]), conf=float(d.confs[i]),
                   x1=d.boxes[i, 0], y1=d.boxes[i, 1], x2=d.boxes[i, 2], y2=d.boxes[i, 3],
                   energy=float(d.energy[i]))
        row.update(zip(KPT_COLS, d.kpts[i].ravel().tolist()))
        rows.append(row)
    return rows


def save_cache(out_dir: Path, rows: list[dict], cams: list[dict], meta: dict) -> None:
    pd.DataFrame(rows).to_parquet(out_dir / "detections.parquet")
    pd.DataFrame(cams).to_parquet(out_dir / "camera.parquet")
    pd.Series(meta).to_json(out_dir / "meta.json")


def load_cache(out_dir: Path):
    """Yield FrameDetections in frame order from a saved cache."""
    det = pd.read_parquet(out_dir / "detections.parquet")
    cams = pd.read_parquet(out_dir / "camera.parquet")
    by_frame = dict(tuple(det.groupby("frame"))) if len(det) else {}
    for cam in cams.itertuples(index=False):
        g = by_frame.get(cam.frame)
        affine = np.array([getattr(cam, c) for c in AFF_COLS], np.float32).reshape(2, 3)
        if g is None:
            yield FrameDetections(cam.frame, cam.t, np.zeros(0, int), np.zeros((0, 4)), np.zeros(0),
                                  np.zeros((0, N_KPTS, 3)), np.zeros(0), affine)
            continue
        yield FrameDetections(cam.frame, cam.t, g.track_id.to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy(),
                              g.conf.to_numpy(), g[KPT_COLS].to_numpy().reshape(-1, N_KPTS, 3),
                              g.energy.to_numpy(), affine)


def cam_row(d: FrameDetections) -> dict:
    return dict(frame=d.frame_idx, t=d.t, **dict(zip(AFF_COLS, d.affine.ravel().tolist())))
