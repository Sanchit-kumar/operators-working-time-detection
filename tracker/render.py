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
    H, W = img.shape[:2]
    th = max(1, W // 1280)  # line thickness scales with resolution
    fs = 0.42 * W / 1280    # worker labels stay small: factory workers are often far away and close together

    # Summary panel first, so worker labels are drawn on top of it rather than hidden under it.
    s = engine.summary()
    lines = [f"Time {fmt(engine.t)}   Workers {s['workers']}",
             f"Working {fmt(s['working_s'])}   Idle {fmt(s['idle_s'])}   Util {s['working_pct']}%"]
    if fps:
        lines.append(f"{fps:.1f} FPS")
    k = W / 1280
    pad, lh = int(8 * k), int(24 * k)
    cv2.rectangle(img, (0, 0), (int(370 * k), pad + lh * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + lh * (i + 1) - int(6 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k,
                    (255, 255, 255), th, cv2.LINE_AA)

    if engine.zones is not None:  # work zones and the people currently outside them
        for z in engine.zones:
            cv2.polylines(img, [(z * scale).astype(np.int32)], True, (255, 200, 0), th)
        for box in engine.outside:
            x1, y1, x2, y2 = (box * scale).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (110, 110, 110), 1)

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
            lines = [f"#{tr.track_id} {tr.state}", f"W {fmt(tot[WORKING])} I {fmt(tot[IDLE])}"]
        else:
            lines = [f"#{tr.track_id} new"]
        if debug and tr.signals:
            sg = tr.signals
            lines.append(f"vlm: {sg['ext']}" if sg.get("ext") else f"e{sg['energy']:.2f} w{sg['wrist']:.2f} b{sg['body']:.2f}")
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0] for t in lines]
        tw, lh = max(w for w, _ in sizes), max(h for _, h in sizes) + int(5 * fs / 0.42)
        bh = lh * len(lines) + 4
        lx = int(np.clip(x1, 0, max(W - tw - 6, 0)))  # keep the label inside the frame
        ly = y1 - bh if y1 - bh >= 0 else y1          # above the box, or inside it at the top edge
        cv2.rectangle(img, (lx, ly), (lx + tw + 6, ly + bh), color, -1)
        for j, t in enumerate(lines):
            cv2.putText(img, t, (lx + 3, ly + lh * (j + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1,
                        cv2.LINE_AA)
    return img
