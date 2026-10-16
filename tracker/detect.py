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
        self.track_kwargs = dict(persist=True, tracker=d["tracker"], conf=d["conf"],
                                 device=d["device"], quantize=d["quantize"], classes=[0], verbose=False)
        self.imgsz_cfg = d["imgsz"]
        self.imgsz = None
        m = cfg["motion"]
        self.motion = MotionEstimator(m["work_width"], m["diff_threshold"], m["fixed_camera"], m["interval_s"])

    @staticmethod
    def auto_imgsz(frame_shape) -> int:
        """Inference size from the video resolution: native size (small far-away workers stay detectable),
        at least 1280 (low-res video is upscaled, which helps tiny people) and at most 1920 (speed)."""
        return int(np.clip(round(max(frame_shape[:2]) / 32) * 32, 1280, 1920))

    def __call__(self, frame: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        if self.imgsz is None:
            self.imgsz = self.auto_imgsz(frame.shape) if self.imgsz_cfg == "auto" else int(self.imgsz_cfg)
            self.track_kwargs["imgsz"] = self.imgsz
        # The model letterboxes to imgsz anyway; resizing first also keeps the tracker's ReID and our motion
        # estimation off the full-resolution (e.g. 4K) frame, which is much faster.
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
        affine, energy = self.motion.update(small, boxes * s, t)
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


DET_COLS = ["frame", "t", "track_id", "conf", "x1", "y1", "x2", "y2", "energy", *KPT_COLS]


class CacheWriter:
    """Accumulates detections as compact numpy blocks (a 15-min video has ~100k+ person-frames)."""

    def __init__(self):
        self.det, self.cam = [], []

    def add(self, d: FrameDetections) -> None:
        n = len(d.ids)
        if n:
            self.det.append(np.column_stack([np.full(n, d.frame_idx), np.full(n, d.t), d.ids, d.confs, d.boxes,
                                             d.energy, d.kpts.reshape(n, -1)]).astype(np.float64))
        self.cam.append(np.r_[d.frame_idx, d.t, d.affine.ravel()])

    def save(self, out_dir: Path, meta: dict) -> None:
        det = pd.DataFrame(np.vstack(self.det) if self.det else np.zeros((0, len(DET_COLS))), columns=DET_COLS)
        det = det.astype({"frame": int, "track_id": int})
        cols = {c: np.float32 for c in DET_COLS[3:]}
        det.astype(cols).to_parquet(out_dir / "detections.parquet")
        cam = pd.DataFrame(np.vstack(self.cam), columns=["frame", "t", *AFF_COLS]).astype({"frame": int})
        cam.to_parquet(out_dir / "camera.parquet")
        pd.Series(meta).to_json(out_dir / "meta.json")


def load_cache(out_dir: Path):
    """Yield FrameDetections in frame order from a saved cache."""
    det = pd.read_parquet(out_dir / "detections.parquet").sort_values(["frame", "track_id"], kind="stable")
    cams = pd.read_parquet(out_dir / "camera.parquet")
    frames = det.frame.to_numpy()
    ids, confs, energy = det.track_id.to_numpy(), det.conf.to_numpy(np.float64), det.energy.to_numpy(np.float64)
    boxes = det[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
    kpts = det[KPT_COLS].to_numpy(np.float64).reshape(-1, N_KPTS, 3)
    affines = cams[AFF_COLS].to_numpy(np.float32).reshape(-1, 2, 3)
    starts = np.searchsorted(frames, cams.frame.to_numpy(), "left")
    ends = np.searchsorted(frames, cams.frame.to_numpy(), "right")
    for k, (f, t) in enumerate(zip(cams.frame.to_numpy(), cams.t.to_numpy())):
        a, b = starts[k], ends[k]
        yield FrameDetections(int(f), float(t), ids[a:b], boxes[a:b], confs[a:b], kpts[a:b], energy[a:b],
                              affines[k].copy())
