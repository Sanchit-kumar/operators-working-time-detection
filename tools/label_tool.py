"""Ground-truth labelling tool for Working / Idle.

Run the detection stage first (it caches tracks):
    python -m tracker.run data/video.mp4 --no-video
then:
    python tools/label_tool.py data/video.mp4

You label by marking state *changes*: select a worker, press w or i, and that state applies from now until the
next change. Each box shows its current label.

Keys
  space        play / pause
  a / d        back / forward 1 s          j / l   back / forward 10 s
  click        select the worker under the mouse
  tab          select next visible worker
  w            selected worker is WORKING from now on
  i            selected worker is IDLE from now on
  x            selected track is NOT a worker (passer-by, false detection) from now on
  u            undo the last change for the selected worker
  + / -        playback speed
  s            save        q   save and quit

Saved files:
  labels/<video>.json   change-points (re-open to continue labelling)
  labels/<video>.csv    one row per worker per second: second, track_id, x1, y1, x2, y2, state
"""

import json
import sys
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.config import ROOT, load_config, output_dir  # noqa: E402

COLORS = {"working": (60, 200, 60), "idle": (40, 140, 255), "ignore": (120, 120, 120), None: (255, 255, 255)}
KEY_STATE = {ord("w"): "working", ord("i"): "idle", ord("x"): "ignore"}


class Labels:
    def __init__(self, path: Path):
        self.path = path
        self.points: dict[int, list[tuple[float, str]]] = {}
        if path.exists():
            self.points = {int(k): [tuple(p) for p in v] for k, v in json.loads(path.read_text()).items()}

    def state(self, tid: int, t: float) -> str | None:
        pts = self.points.get(tid, [])
        i = bisect_right([p[0] for p in pts], t + 1e-6)
        return pts[i - 1][1] if i else None

    def set(self, tid: int, t: float, state: str) -> None:
        pts = [p for p in self.points.get(tid, []) if abs(p[0] - t) > 1e-6]
        pts.append((round(t, 3), state))
        self.points[tid] = sorted(pts)

    def undo(self, tid: int) -> None:
        if self.points.get(tid):
            self.points[tid].pop()

    def save(self) -> None:
        self.path.write_text(json.dumps({str(k): v for k, v in self.points.items()}, indent=1))


def export_csv(labels: Labels, det: pd.DataFrame, fps: float, out: Path) -> int:
    """One row per labelled track per second, with the track's box at the middle of that second."""
    rows = []
    det = det.assign(second=(det.t).astype(int))
    for (sec, tid), g in det.groupby(["second", "track_id"]):
        state = labels.state(int(tid), sec + 0.5)
        if state is None:
            continue
        r = g.iloc[(g.t - (sec + 0.5)).abs().argmin()]
        rows.append(dict(second=int(sec), track_id=int(tid), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, state=state))
    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)


def main():
    video = Path(sys.argv[1])
    cfg = load_config()
    cache = output_dir(cfg, video)
    if not (cache / "detections.parquet").exists():
        sys.exit(f"No detection cache in {cache}. Run: python -m tracker.run {video} --no-video")
    det = pd.read_parquet(cache / "detections.parquet")
    by_frame = {f: g for f, g in det.groupby("frame")}
    det_frames = sorted(by_frame)

    (ROOT / "labels").mkdir(exist_ok=True)
    labels = Labels(ROOT / "labels" / f"{video.stem}.json")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = 1280
    scale = W / cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    state = dict(frame=0, playing=False, selected=None, speed=1.0, boxes=[])

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            hits = [(tid, (bx2 - bx1) * (by2 - by1)) for tid, (bx1, by1, bx2, by2) in state["boxes"]
                    if bx1 <= x <= bx2 and by1 <= y <= by2]
            if hits:
                state["selected"] = min(hits, key=lambda h: h[1])[0]  # smallest box under the cursor

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    cached_idx, cached_img = -1, None

    while True:
        f = state["frame"]
        if f != cached_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f) if f != cached_idx + 1 else None
            ok, img = cap.read()
            if not ok:
                state["frame"], state["playing"] = max(f - 1, 0), False
                continue
            cached_idx, cached_img = f, cv2.resize(img, None, fx=scale, fy=scale)
        view = cached_img.copy()
        t = f / fps

        # draw tracks (use the nearest cached frame when the detector ran with vid_stride > 1)
        g = by_frame.get(f)
        if g is None and det_frames:
            i = min(bisect_right(det_frames, f), len(det_frames) - 1)
            near = min(det_frames[max(i - 1, 0)], det_frames[i], key=lambda k: abs(k - f))
            g = by_frame[near] if abs(near - f) <= 2 else None
        state["boxes"] = []
        for r in (g.itertuples() if g is not None else []):
            tid = int(r.track_id)
            b = (np.array([r.x1, r.y1, r.x2, r.y2]) * scale).astype(int)
            state["boxes"].append((tid, b))
            lab = labels.state(tid, t)
            thick = 4 if tid == state["selected"] else 2
            cv2.rectangle(view, tuple(b[:2]), tuple(b[2:]), COLORS[lab], thick)
            cv2.putText(view, f"#{tid} {lab or '?'}", (b[0], b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS[lab], 2, cv2.LINE_AA)

        # status bar + timeline of the selected worker
        H = view.shape[0]
        bar = np.zeros((70, W, 3), np.uint8)
        sel = state["selected"]
        if sel is not None:
            for x in range(W):
                lab = labels.state(sel, x / W * n / fps)
                bar[40:62, x] = COLORS[lab] if lab else (40, 40, 40)
        cx = int(f / max(n - 1, 1) * (W - 1))
        bar[36:66, cx] = (255, 255, 255)
        txt = (f"{int(t // 60):02d}:{t % 60:05.2f}  frame {f}/{n}  sel #{sel}  speed x{state['speed']}  "
               f"{'PLAY' if state['playing'] else 'PAUSE'}   [w]ork [i]dle [x]ignore [u]ndo [s]ave [q]uit")
        cv2.putText(bar, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("label", np.vstack([view, bar]))

        delay = max(1, int(1000 / fps / state["speed"])) if state["playing"] else 30
        k = cv2.waitKey(delay) & 0xFF
        visible = [b[0] for b in state["boxes"]]
        if k == ord("q"):
            break
        elif k == ord(" "):
            state["playing"] = not state["playing"]
        elif k in (ord("a"), ord("d"), ord("j"), ord("l")):
            step = {ord("a"): -1, ord("d"): 1, ord("j"): -10, ord("l"): 10}[k]
            state["frame"] = int(np.clip(f + step * fps, 0, n - 1))
        elif k == 9 and visible:  # tab
            i = visible.index(sel) + 1 if sel in visible else 0
            state["selected"] = visible[i % len(visible)]
        elif k in KEY_STATE and sel is not None:
            labels.set(sel, t, KEY_STATE[k])
        elif k == ord("u") and sel is not None:
            labels.undo(sel)
        elif k in (ord("+"), ord("=")):
            state["speed"] = min(state["speed"] * 2, 8)
        elif k == ord("-"):
            state["speed"] = max(state["speed"] / 2, 0.25)
        elif k == ord("s"):
            labels.save()
            print("saved")
        if state["playing"]:
            state["frame"] = min(f + 1, n - 1)

    labels.save()
    rows = export_csv(labels, det, fps, ROOT / "labels" / f"{video.stem}.csv")
    print(f"saved {labels.path} and {rows} labelled worker-seconds to labels/{video.stem}.csv")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
