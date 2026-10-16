"""Real-time view: detection, tracking and Working/Idle classification with a live overlay.

    python -m tracker.live data/video.mp4            # a file, played at processing speed
    python -m tracker.live rtsp://user:pass@cam/stream
    python -m tracker.live 0                          # webcam
    python -m tracker.live data/video.mp4 --save outputs/live.mp4 --no-show
    python -m tracker.live rtsp://cam/stream --zones line3    # work zones from zones/line3.json

Work zones default to zones/<video name>.json for files (see tools/zone_tool.py).

Keys: q quit, space pause, d toggle debug signals.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2

from .config import ROOT, load_config
from .detect import Detector
from .engine import ActivityEngine
from .zones import load_zones
from .render import draw


class LatestFrameReader:
    """Reads a live stream on a background thread and keeps only the newest frame.

    Without this, a stream that produces frames faster than we process them builds up a buffer and the
    display falls further and further behind real time.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap, self.frame, self.ok = cap, None, True
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok, self.frame = ok, frame
                self.cond.notify()

    def read(self):
        with self.cond:
            self.cond.wait_for(lambda: self.frame is not None or not self.ok)
            frame, self.frame = self.frame, None
            return frame is not None, frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--config", default=ROOT / "configs/live.yaml")
    p.add_argument("--save", default=None, help="also write the annotated stream to this .mp4")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--display-width", type=int, default=1280)
    p.add_argument("--zones", default=None, help="zone file name in zones/ (default: the video's name)")
    a = p.parse_args()
    cfg = load_config(a.config)

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = cfg["detector"]["vid_stride"]
    det = Detector(cfg)
    is_file = isinstance(src, str) and not src.startswith(("rtsp", "http"))
    zone_name = a.zones or (Path(src).stem if is_file else None)
    zones = load_zones(cfg, zone_name, int(cap.get(3)), int(cap.get(4))) if zone_name else None
    engine = ActivityEngine(cfg, frame_dt=stride / fps_src, zones=zones)
    reader = cap if is_file else LatestFrameReader(cap)  # files: every frame; streams: newest frame

    writer, debug, idx, fps, n_done = None, False, 0, 0.0, 0
    tic = last = time.time()
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        if is_file and idx % stride:
            idx += 1
            continue
        # Files use the video clock; streams use the wall clock.
        t = idx / fps_src if is_file else time.time() - tic
        visible = engine.update(det(frame, idx, t))
        idx += 1
        n_done += 1

        now = time.time()
        inst = 1.0 / max(now - last, 1e-6)
        fps = 0.9 * fps + 0.1 * inst if fps else inst
        last = now

        scale = a.display_width / frame.shape[1]
        view = draw(cv2.resize(frame, None, fx=scale, fy=scale), visible, engine, fps=fps, debug=debug, scale=scale)

        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_src / stride,
                                         (view.shape[1], view.shape[0]))
            writer.write(view)
        if not a.no_show:
            cv2.imshow("Worker activity", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord(" "):
                cv2.waitKey(0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    s = engine.summary()
    print(f"workers {s['workers']} | working {s['working_s']}s | idle {s['idle_s']}s | util {s['working_pct']}% | "
          f"avg {n_done / (time.time() - tic):.1f} FPS (incl. warm-up)")


if __name__ == "__main__":
    main()
