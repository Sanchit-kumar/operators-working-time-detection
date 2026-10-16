"""Overlay drawing for the live view and annotated output videos."""

import cv2
import numpy as np

from . import IDLE, WORKING
from .engine import ActivityEngine, Track

COLORS = {WORKING: (60, 200, 60), IDLE: (40, 140, 255), None: (180, 180, 180)}  # BGR
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12)]


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def draw(img: np.ndarray, visible: list[Track], engine: ActivityEngine, fps: float | None = None,
         debug: bool = False, scale: float = 1.0) -> np.ndarray:
    """Draw onto img, which may be a resized copy of the frame the tracks were detected in (scale).

    Tracks not yet confirmed as workers (too short so far, see features.min_track_s) are drawn grey and
    are not in the totals yet.
    """
    th = max(1, img.shape[1] // 640)  # line thickness scales with resolution
    fs = 0.5 * th
    H, W = img.shape[:2]

    # Summary panel first, so worker labels are drawn on top of it rather than hidden under it.
    s = engine.summary()
    lines = [f"Time {fmt(engine.t)}   Workers {s['workers']}",
             f"Working {fmt(s['working_s'])}   Idle {fmt(s['idle_s'])}   Util {s['working_pct']}%"]
    if fps:
        lines.append(f"{fps:.1f} FPS")
    pad, lh = 8 * th, int(24 * th)
    cv2.rectangle(img, (0, 0), (int(370 * th), pad + lh * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + lh * (i + 1) - 6 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * th,
                    (255, 255, 255), th, cv2.LINE_AA)

    confirmed = {t.track_id for t in engine.workers()}
    for tr in visible:
        ok = tr.track_id in confirmed
        color = COLORS[tr.state] if ok else COLORS[None]
        x1, y1, x2, y2 = (tr.box * scale).astype(int)
        pts = (tr.kpts[:, :2] * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, (2 if ok else 1) * th)
        for a, b in SKELETON:
            if tr.kpts[a, 2] > 0.4 and tr.kpts[b, 2] > 0.4:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, th)
        if ok:
            tot = tr.totals()
            label = f"#{tr.track_id} {tr.state}  W {fmt(tot[WORKING])} I {fmt(tot[IDLE])}"
        else:
            label = f"#{tr.track_id} new"
        if debug and tr.signals:
            sg = tr.signals
            label += f"  e{sg['energy']:.2f} w{sg['wrist']:.2f} b{sg['body']:.2f}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        bh = tht + 8 * th
        lx = int(np.clip(x1, 0, max(W - tw - 6, 0)))  # keep the label inside the frame
        ly = y1 if y1 - bh >= 0 else y1 + bh          # above the box, or inside it at the top edge
        cv2.rectangle(img, (lx, ly - bh), (lx + tw + 6, ly), color, -1)
        cv2.putText(img, label, (lx + 3, ly - 4 * th), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
    return img
