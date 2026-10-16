"""Draw work zones for a fixed camera. Only people whose feet are inside a zone are counted as workers.

    python tools/zone_tool.py data/video.mp4               # saves zones/<video name>.json
    python tools/zone_tool.py data/video.mp4 --name line3  # saves zones/line3.json (e.g. one file per camera)
    python tools/zone_tool.py data/video.mp4 --at 120      # draw on the frame at 120 s

Keys
  left click   add a point to the current zone
  enter / n    close the current zone (needs 3+ points)
  u            undo the last point          d   delete the last zone
  s            save                         q   save and quit
  [ / ]        previous / next frame (10 s) to check where people stand
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.config import load_config  # noqa: E402
from tracker.zones import zones_path  # noqa: E402

DISPLAY_W = 1280


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--name", default=None)
    p.add_argument("--at", type=float, default=2.0, help="time (s) of the frame to draw on")
    a = p.parse_args()

    cfg = load_config()
    path = zones_path(cfg, a.name or a.video.stem)
    path.parent.mkdir(exist_ok=True)
    zones = json.loads(path.read_text())["zones"] if path.exists() else []

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    W, H = cap.get(3), cap.get(4)
    scale = DISPLAY_W / W
    size = (DISPLAY_W, int(H * scale))
    t = a.at
    current: list[list[float]] = []  # normalised points of the zone being drawn

    def read(t_s):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
        ok, img = cap.read()
        return cv2.resize(img, size) if ok else np.zeros((size[1], size[0], 3), np.uint8)

    frame = read(t)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([round(x / size[0], 4), round(y / size[1], 4)])

    cv2.namedWindow("zones")
    cv2.setMouseCallback("zones", on_mouse)

    def save():
        path.write_text(json.dumps({"zones": zones}, indent=1))
        print(f"saved {len(zones)} zone(s) to {path}")

    while True:
        view = frame.copy()
        overlay = view.copy()
        for z in zones:
            pts = (np.array(z["points"]) * size).astype(np.int32)
            cv2.fillPoly(overlay, [pts], (255, 200, 0))
            cv2.putText(view, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        view = cv2.addWeighted(overlay, 0.3, view, 0.7, 0)
        for z in zones:
            cv2.polylines(view, [(np.array(z["points"]) * size).astype(np.int32)], True, (255, 200, 0), 2)
        if current:
            pts = (np.array(current) * size).astype(np.int32)
            cv2.polylines(view, [pts], False, (0, 255, 255), 2)
            for q in pts:
                cv2.circle(view, tuple(q), 4, (0, 255, 255), -1)
        cv2.putText(view, f"{t:.0f}s  zones: {len(zones)}  click=point  enter=close  u=undo  d=delete  s=save  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("zones", view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10, ord("n")) and len(current) >= 3:
            zones.append({"name": f"zone {len(zones) + 1}", "points": current})
            current = []
        elif k == ord("u") and current:
            current.pop()
        elif k == ord("d") and zones:
            zones.pop()
        elif k in (ord("["), ord("]")):
            t = float(np.clip(t + (10 if k == ord("]") else -10), 0, max(duration - 1, 0)))
            frame = read(t)
        elif k == ord("s"):
            save()
        elif k == ord("q"):
            save()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
