"""Rename videos in data/ to video1.mp4, video2.mp4, ... and move everything that belongs to them.

For each renamed video it also renames labels/<name>.{json,csv}, zones/<name>.json and outputs/<name>/ (fixing
the path stored in meta.json), and records the original name in videos.csv. Labels and zones are moved with
`git mv` when tracked, so their history is kept.

    python tools/rename_videos.py                      # show the plan for every video not named videoN yet
    python tools/rename_videos.py --yes                # apply it (alphabetical order)
    python tools/rename_videos.py "data/b.mp4" "data/a.mp4" --yes   # apply in this order
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA, LABELS, ZONES, OUTPUTS = ROOT / "data", ROOT / "labels", ROOT / "zones", ROOT / "outputs"
INDEX = ROOT / "videos.csv"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
NAME = re.compile(r"^video(\d+)$")


def is_tracked(path: Path) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True)
    return r.returncode == 0


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst}")
    if src.is_file() and is_tracked(src):
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)
    else:
        src.rename(dst)
    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    info = dict(width=int(cap.get(3)), height=int(cap.get(4)), fps=round(fps, 2),
                duration_s=round(n / fps, 1) if fps else "")
    cap.release()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", type=Path, help="videos to rename, in this order (default: all, sorted)")
    p.add_argument("--yes", action="store_true", help="apply (otherwise only show the plan)")
    a = p.parse_args()

    existing = [f for f in DATA.iterdir() if f.suffix.lower() in VIDEO_EXT]
    taken = [int(m.group(1)) for f in existing if (m := NAME.match(f.stem))]
    todo = [v.resolve() for v in a.videos] if a.videos else sorted(f for f in existing if not NAME.match(f.stem))
    todo = [v for v in todo if not NAME.match(v.stem)]
    if not todo:
        print("all videos already follow videoN naming")
        return

    n = max(taken, default=0)
    plan = []
    for v in todo:
        if not v.exists():
            sys.exit(f"not found: {v}")
        n += 1
        plan.append((v, f"video{n}"))
    for v, new in plan:
        print(f"{v.name}  ->  {new}{v.suffix.lower()}")
    if not a.yes:
        print("\n(dry run: add --yes to apply)")
        return

    rows = list(csv.DictReader(INDEX.open())) if INDEX.exists() else []
    for v, new in plan:
        old = v.stem
        print(f"\n{old} -> {new}")
        info = video_info(v)
        move(v, DATA / f"{new}{v.suffix.lower()}")
        for ext in (".json", ".csv"):
            move(LABELS / f"{old}{ext}", LABELS / f"{new}{ext}")
        move(ZONES / f"{old}.json", ZONES / f"{new}.json")
        out = OUTPUTS / old
        if out.exists():
            move(out, OUTPUTS / new)
            meta = OUTPUTS / new / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                m["video"] = f"data/{new}{v.suffix.lower()}"
                meta.write_text(json.dumps(m))
            summary = OUTPUTS / new / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                s["video"] = f"{new}{v.suffix.lower()}"
                summary.write_text(json.dumps(s, indent=2))
        rows.append(dict(name=f"{new}{v.suffix.lower()}", original_name=v.name, **info))

    with INDEX.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "original_name", "width", "height", "fps", "duration_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nrecorded in {INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
