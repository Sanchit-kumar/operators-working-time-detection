"""Ground-truth labelling tool for Working / Idle.

Run the detection stage first (it caches the tracked workers):
    python -m tracker.run data/video.mp4 --no-video
then:
    python tools/label_tool.py data/video.mp4

Screen layout
  video        boxes coloured by label (white = not labelled yet). Click a box to select that worker.
  overview     the whole video; click or drag to jump anywhere.
  lanes        one timeline row per worker on screen. DRAG across a lane to paint that range with the current
               brush (w / i / x). Click a lane to select the worker and jump to that time.

Keys
  w / i / x    set the brush to working / idle / not-a-worker, and label the selected worker from now
               until their next label change
  W / I / X    label the selected worker's WHOLE track (fastest when someone does one thing throughout)
  space        play / pause                  + / -   playback speed
  a / d        back / forward 1 s            j / l   back / forward 10 s      , / .   one frame
  tab          select the next worker on screen
  z            zoom the lanes: 20 s / 1 min / 5 min / whole video
  u            undo (any number of steps)
  n            jump to the next worker that still has unlabelled time
  h            show / hide help              q       save and quit

Every edit is saved immediately to labels/<video>.json; labels/<video>.csv (read by the benchmark) is written on
quit. Re-open the tool any time to continue or correct.
"""

import argparse
import copy
import json
import sys
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.config import ROOT, load_config, output_dir  # noqa: E402
from tracker.zones import load_zones  # noqa: E402

COLORS = {"working": (60, 200, 60), "idle": (40, 140, 255), "ignore": (130, 130, 130), None: (255, 255, 255)}
NAMES = {"working": "WORKING", "idle": "IDLE", "ignore": "NOT A WORKER", None: "-"}
BRUSH_KEYS = {ord("w"): "working", ord("i"): "idle", ord("x"): "ignore"}
WHOLE_KEYS = {ord("W"): "working", ord("I"): "idle", ord("X"): "ignore"}
ZOOMS = [20, 60, 300, None]  # lane window in seconds; None = whole video
LANE_H, LANE_LABEL_W, MAX_LANES = 24, 120, 8
STATUS_H, OVERVIEW_H = 28, 22
EMPTY_LANE = (45, 45, 45)
UNLABELLED = (90, 90, 160)

HELP = [
    "click a box = select worker      drag on a lane = paint range with brush",
    "w / i / x = brush + label from now     W / I / X = label whole track",
    "space play   a/d 1s   j/l 10s   ,/. frame   click overview = jump",
    "u undo   n next unlabelled   z zoom lanes   +/- speed   h help   q quit",
]


class Labels:
    """Per-track change-points [(t, state)], state in working / idle / ignore / None (= unlabelled)."""

    def __init__(self, path: Path):
        self.path = path
        self.points: dict[int, list[tuple[float, str | None]]] = {}
        if path.exists():
            self.points = {int(k): [tuple(p) for p in v] for k, v in json.loads(path.read_text()).items()}
        self.history: list[dict] = []
        self.version = 0  # bumped on every change, so callers can cache derived values

    def state(self, tid: int, t: float) -> str | None:
        pts = self.points.get(tid, [])
        i = bisect_right([p[0] for p in pts], t + 1e-6)
        return pts[i - 1][1] if i else None

    def _checkpoint(self):
        self.history.append(copy.deepcopy(self.points))
        self.history = self.history[-200:]

    def undo(self) -> bool:
        if not self.history:
            return False
        self.points = self.history.pop()
        self.version += 1
        self.save()
        return True

    def paint(self, tid: int, t0: float, t1: float | None, state: str) -> None:
        """Label [t0, t1) with state; t1=None means until the worker's next existing change."""
        self._checkpoint()
        pts = self.points.get(tid, [])
        if t1 is None:  # until the next change after t0
            later = [p[0] for p in pts if p[0] > t0 + 1e-6]
            t1 = later[0] if later else float("inf")
        t0, t1 = min(t0, t1), max(t0, t1)
        after = self.state(tid, t1)
        pts = [p for p in pts if not (t0 - 1e-6 <= p[0] <= t1 + 1e-6)]
        pts.append((round(t0, 3), state))
        if t1 != float("inf"):
            pts.append((round(t1, 3), after))
        pts.sort()
        merged = []  # drop change-points that don't change anything
        for p in pts:
            if not merged or merged[-1][1] != p[1]:
                merged.append(p)
        if merged and merged[0][1] is None:
            merged = merged[1:]
        self.points[tid] = merged
        self.version += 1
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps({str(k): v for k, v in self.points.items()}, indent=1))


class FrameReader:
    """Random access to video frames with a small cache; forward steps decode sequentially (fast)."""

    def __init__(self, path: Path, width: int):
        self.cap = cv2.VideoCapture(str(path))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.full_w, self.full_h = int(self.cap.get(3)), int(self.cap.get(4))
        self.scale = width / self.full_w
        self.size = (width, int(round(self.full_h * self.scale)))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.pos = 0  # index of the frame the next cap.read() returns

    def get(self, f: int) -> np.ndarray:
        f = int(np.clip(f, 0, self.n - 1))
        if f in self.cache:
            self.cache.move_to_end(f)
            return self.cache[f]
        if 0 <= f - self.pos < 50:
            for _ in range(f - self.pos):
                self.cap.grab()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = self.cap.read()
        self.pos = f + 1
        img = cv2.resize(img, self.size) if ok else np.zeros((self.size[1], self.size[0], 3), np.uint8)
        self.cache[f] = img
        if len(self.cache) > 300:
            self.cache.popitem(last=False)
        return img


def export_csv(labels: Labels, det: pd.DataFrame, out: Path) -> int:
    """One row per labelled track per second, with the track's box at the middle of that second."""
    rows = []
    det = det.assign(second=det.t.astype(int))
    for (sec, tid), g in det.groupby(["second", "track_id"]):
        state = labels.state(int(tid), sec + 0.5)
        if state is None:
            continue
        r = g.iloc[(g.t - (sec + 0.5)).abs().argmin()]
        rows.append(dict(second=int(sec), track_id=int(tid), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, state=state))
    pd.DataFrame(rows, columns=["second", "track_id", "x1", "y1", "x2", "y2", "state"]).to_csv(out, index=False)
    return len(rows)


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--width", type=int, default=1152, help="display width of the video (smaller if the window is too tall)")
    a = p.parse_args()

    cfg = load_config()
    cache = output_dir(cfg, a.video)
    if not (cache / "detections.parquet").exists():
        sys.exit(f'No detection cache in {cache}. Run first: python -m tracker.run "{a.video}" --no-video')
    det = pd.read_parquet(cache / "detections.parquet")
    by_frame = {f: g for f, g in det.groupby("frame")}
    det_frames = sorted(by_frame)
    life = det.groupby("track_id").t.agg(["min", "max"])
    life["max"] += 0.5  # cover the last frame's second

    (ROOT / "labels").mkdir(exist_ok=True)
    labels = Labels(ROOT / "labels" / f"{a.video.stem}.json")
    rd = FrameReader(a.video, a.width)
    W, Hv = rd.size
    dur = rd.n / rd.fps
    zones = load_zones(cfg, a.video.stem, rd.full_w, rd.full_h)

    st = dict(frame=0, playing=False, selected=None, brush="working", speed=1.0, zoom=1, help=True,
              drag=None, lanes=[], boxes=[], msg="")

    def t_of(f):
        return f / rd.fps

    def seek(t):
        st["frame"] = int(np.clip(round(t * rd.fps), 0, rd.n - 1))

    def window():
        z = ZOOMS[st["zoom"]]
        if z is None or z >= dur:
            return 0.0, dur
        t = t_of(st["frame"])
        w0 = float(np.clip(t - z / 2, 0, dur - z))
        return w0, w0 + z

    def lane_x_to_t(x):
        w0, w1 = window()
        return w0 + (x - LANE_LABEL_W) / (W - LANE_LABEL_W - 4) * (w1 - w0)

    def lane_t_to_x(t):
        w0, w1 = window()
        return int(LANE_LABEL_W + (t - w0) / (w1 - w0) * (W - LANE_LABEL_W - 4))

    todo_cache = {}

    def unlabelled_tracks():
        if labels.version in todo_cache:
            return todo_cache[labels.version]
        out = []
        for tid, (t0, t1) in life.iterrows():
            ts = np.arange(t0, t1, 0.5)
            if any(labels.state(int(tid), x) is None for x in ts):
                out.append((t0, int(tid)))
        todo_cache.clear()
        todo_cache[labels.version] = sorted(out)
        return todo_cache[labels.version]

    def on_mouse(event, x, y, flags, _):
        y_over = Hv + STATUS_H
        y_lanes = y_over + OVERVIEW_H + 4
        if y < Hv:  # video: select the smallest box under the cursor
            if event == cv2.EVENT_LBUTTONDOWN:
                hits = [(tid, (b[2] - b[0]) * (b[3] - b[1])) for tid, b in st["boxes"]
                        if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
                if hits:
                    st["selected"] = min(hits, key=lambda h: h[1])[0]
            return
        if y_over <= y < y_over + OVERVIEW_H:  # overview: click / drag to jump
            if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
                seek(x / W * dur)
            return
        k = (y - y_lanes) // LANE_H
        if 0 <= k < len(st["lanes"]):
            tid = st["lanes"][k]
            if event == cv2.EVENT_LBUTTONDOWN:
                st["selected"] = tid
                if x >= LANE_LABEL_W:
                    t = lane_x_to_t(x)
                    st["drag"] = [tid, t, t]
                    seek(t)
            elif event == cv2.EVENT_MOUSEMOVE and st["drag"] and flags & cv2.EVENT_FLAG_LBUTTON:
                st["drag"][2] = lane_x_to_t(max(x, LANE_LABEL_W))
        if event == cv2.EVENT_LBUTTONUP and st["drag"]:
            tid, t0, t1 = st["drag"]
            st["drag"] = None
            if abs(t1 - t0) >= 0.2:
                lo, hi = max(min(t0, t1), life.loc[tid, "min"]), min(max(t0, t1), life.loc[tid, "max"])
                if hi > lo:
                    labels.paint(tid, lo, hi, st["brush"])
                    st["msg"] = f"#{tid} {fmt(lo)}-{fmt(hi)} = {NAMES[st['brush']]}"

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)

    while True:
        f = st["frame"]
        t = t_of(f)
        view = rd.get(f).copy()
        for z in zones or []:
            cv2.polylines(view, [(z * rd.scale).astype(np.int32)], True, (255, 200, 0), 1)

        # boxes on the video (nearest cached frame when detection ran with vid_stride > 1)
        g = by_frame.get(f)
        if g is None and det_frames:
            i = min(bisect_right(det_frames, f), len(det_frames) - 1)
            near = min(det_frames[max(i - 1, 0)], det_frames[i], key=lambda k: abs(k - f))
            g = by_frame[near] if abs(near - f) <= 2 else None
        st["boxes"] = []
        for r in (g.itertuples() if g is not None else []):
            tid = int(r.track_id)
            b = (np.array([r.x1, r.y1, r.x2, r.y2]) * rd.scale).astype(int)
            st["boxes"].append((tid, b))
            lab = labels.state(tid, t)
            sel = tid == st["selected"]
            cv2.rectangle(view, tuple(b[:2]), tuple(b[2:]), COLORS[lab], 3 if sel else 1)
            txt = f"#{tid}" + ("" if lab else " ?")
            org = (int(b[0]), int(max(b[1] - 4, 12)))
            cv2.putText(view, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[lab], 1, cv2.LINE_AA)
        if st["help"]:
            cv2.rectangle(view, (0, 0), (560, 20 * len(HELP) + 10), (0, 0, 0), -1)
            for j, line in enumerate(HELP):
                cv2.putText(view, line, (8, 20 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)

        # lanes: workers on screen now, plus the selected one
        on_screen = [tid for tid, _ in st["boxes"]]
        lanes = ([st["selected"]] if st["selected"] is not None and st["selected"] not in on_screen else []) + on_screen
        st["lanes"] = lanes[:MAX_LANES]

        panel = np.full((STATUS_H + OVERVIEW_H + 22 + LANE_H * MAX_LANES, W, 3), 25, np.uint8)
        todo = unlabelled_tracks()
        status = (f"{fmt(t)} / {fmt(dur)}   sel #{st['selected']}   brush: {NAMES[st['brush']]}   "
                  f"x{st['speed']:g} {'PLAY' if st['playing'] else 'PAUSE'}   to label: {len(todo)}   {st['msg']}")
        cv2.putText(panel, status, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(panel, (W - 14, 14), 8, COLORS[st["brush"]], -1)

        oy = STATUS_H  # overview of the whole video, with the lane window and the playhead
        panel[oy:oy + OVERVIEW_H] = (60, 60, 60)
        w0, w1 = window()
        cv2.rectangle(panel, (int(w0 / dur * W), oy), (int(w1 / dur * W), oy + OVERVIEW_H - 1), (110, 110, 110), -1)
        cv2.line(panel, (int(t / dur * W), oy), (int(t / dur * W), oy + OVERVIEW_H), (0, 255, 255), 2)
        cv2.putText(panel, "overview (click to jump)", (6, oy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        ly0 = STATUS_H + OVERVIEW_H + 4
        for k, tid in enumerate(st["lanes"]):
            y = ly0 + k * LANE_H
            t0, t1 = life.loc[tid, "min"], life.loc[tid, "max"]
            panel[y:y + LANE_H - 3, LANE_LABEL_W:W - 4] = EMPTY_LANE
            for x in range(LANE_LABEL_W, W - 4, 2):
                tx = lane_x_to_t(x)
                if t0 <= tx <= t1:
                    lab = labels.state(tid, tx)
                    panel[y:y + LANE_H - 3, x:x + 2] = COLORS[lab] if lab else UNLABELLED
            if st["drag"] and st["drag"][0] == tid:
                xa, xb = sorted((lane_t_to_x(st["drag"][1]), lane_t_to_x(st["drag"][2])))
                cv2.rectangle(panel, (xa, y), (xb, y + LANE_H - 4), COLORS[st["brush"]], 2)
            is_sel = tid == st["selected"]
            cv2.rectangle(panel, (0, y), (LANE_LABEL_W - 4, y + LANE_H - 3), (90, 90, 90) if is_sel else (50, 50, 50), -1)
            lab = labels.state(tid, t)
            cv2.putText(panel, f"#{tid} {lab or '?'}", (6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[lab], 1,
                        cv2.LINE_AA)
        if st["lanes"]:
            px = lane_t_to_x(t)
            y_end = ly0 + LANE_H * len(st["lanes"])
            cv2.line(panel, (px, ly0), (px, y_end), (0, 255, 255), 1)
            cv2.putText(panel, f"lanes {fmt(w0)} - {fmt(w1)}  (z = zoom)", (W - 250, y_end + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(panel, "no workers on screen here", (LANE_LABEL_W, ly0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (160, 160, 160), 1)

        cv2.imshow("label", np.vstack([view, panel]))

        delay = max(1, int(1000 / rd.fps / min(st["speed"], 1))) if st["playing"] else 20
        k = cv2.waitKey(delay) & 0xFF
        sel = st["selected"]
        if k == ord("q"):
            break
        elif k == ord(" "):
            st["playing"] = not st["playing"]
        elif k in (ord("a"), ord("d"), ord("j"), ord("l")):
            seek(t + {ord("a"): -1, ord("d"): 1, ord("j"): -10, ord("l"): 10}[k])
        elif k in (ord(","), ord(".")):
            st["frame"] = int(np.clip(f + (1 if k == ord(".") else -1), 0, rd.n - 1))
        elif k == 9 and on_screen:  # tab
            i = on_screen.index(sel) + 1 if sel in on_screen else 0
            st["selected"] = on_screen[i % len(on_screen)]
        elif k in BRUSH_KEYS:
            st["brush"] = BRUSH_KEYS[k]
            if sel is not None:
                labels.paint(sel, max(t, life.loc[sel, "min"]), None, st["brush"])
                st["msg"] = f"#{sel} from {fmt(t)} = {NAMES[st['brush']]}"
        elif k in WHOLE_KEYS and sel is not None:
            st["brush"] = WHOLE_KEYS[k]
            labels.paint(sel, life.loc[sel, "min"], life.loc[sel, "max"], st["brush"])
            st["msg"] = f"#{sel} whole track = {NAMES[st['brush']]}"
        elif k == ord("u"):
            st["msg"] = "undone" if labels.undo() else "nothing to undo"
        elif k == ord("n"):
            todo = unlabelled_tracks()
            if todo:
                t_first, tid = next(((t0, tid) for t0, tid in todo if tid != sel), todo[0])
                ts = np.arange(life.loc[tid, "min"], life.loc[tid, "max"], 0.5)
                seek(next((x for x in ts if labels.state(tid, x) is None), t_first))
                st["selected"], st["playing"] = tid, False
            else:
                st["msg"] = "all workers labelled"
        elif k == ord("z"):
            st["zoom"] = (st["zoom"] + 1) % len(ZOOMS)
        elif k == ord("h"):
            st["help"] = not st["help"]
        elif k in (ord("+"), ord("=")):
            st["speed"] = min(st["speed"] * 2, 16)
        elif k == ord("-"):
            st["speed"] = max(st["speed"] / 2, 0.25)

        if st["playing"]:
            step = max(1, int(st["speed"]))
            if f + step >= rd.n:
                st["playing"] = False
            st["frame"] = min(f + step, rd.n - 1)

    labels.save()
    rows = export_csv(labels, det, ROOT / "labels" / f"{a.video.stem}.csv")
    print(f"saved {labels.path} and {rows} labelled worker-seconds to labels/{a.video.stem}.csv")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
