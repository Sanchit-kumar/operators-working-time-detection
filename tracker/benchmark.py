"""Benchmark predictions against ground-truth labels (from tools/label_tool.py).

    python -m tracker.benchmark data/a.mp4 data/b.mp4              # whole videos
    python -m tracker.benchmark data/a.mp4 --split test            # last 40 % of each video only
    python -m tracker.benchmark data/a.mp4 --config configs/tuned.yaml

Ground truth rows are matched to predicted tracks by box IoU at the same second, so labels stay valid if the
tracker or detector changes. Requires the detection cache (python -m tracker.run <video> --no-video).

Metrics
  per-second accuracy, per-class precision/recall/F1, macro-F1, confusion matrix
  total-time error: |predicted - true| seconds of working and idle, overall and per worker
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import IDLE, STATES, WORKING
from .config import ROOT, load_config, output_dir
from .detect import load_cache
from .engine import ActivityEngine

MISSED = "missed"
DEV_FRACTION = 0.6  # first 60 % of each video is the dev (tuning) split, the rest is test


def replay(cfg: dict, cache: Path, record: bool = False) -> ActivityEngine:
    meta = json.loads((cache / "meta.json").read_text())
    engine = ActivityEngine(cfg, frame_dt=meta["stride"] / meta["fps"], record=record)
    for d in load_cache(cache):
        engine.update(d)
    return engine


def split_window(cache: Path, split: str) -> tuple[float, float]:
    meta = json.loads((cache / "meta.json").read_text())
    dur = meta["frames"] / meta["fps"]
    return {"all": (0, dur), "dev": (0, dur * DEV_FRACTION), "test": (dur * DEV_FRACTION, dur)}[split]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of box a (4,) against boxes b (N, 4)."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def state_at(track, t: float) -> str | None:
    for s in track.segments:
        if s.start <= t < s.end:
            return s.state
    return None


def match_predictions(engine: ActivityEngine, gt: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """Add a `pred` column to the ground truth: the state of the predicted worker covering each GT box."""
    workers = {tr.track_id: tr for tr in engine.workers()}
    det = det.assign(second=det.t.astype(int))
    preds = []
    for sec, g in gt.groupby("second"):
        d = det[det.second == sec]
        if len(d):  # each predicted track's box closest to the middle of the second (tracks may drop frames)
            d = d.loc[(d.t - (sec + 0.5)).abs().groupby(d.track_id).idxmin()]
        boxes, ids = d[["x1", "y1", "x2", "y2"]].to_numpy(), d.track_id.to_numpy()
        used = set()
        for r in g.itertuples():
            pred = MISSED
            if len(boxes):
                ious = iou(np.array([r.x1, r.y1, r.x2, r.y2]), boxes)
                for j in np.argsort(-ious):
                    if ious[j] < 0.3:
                        break
                    if ids[j] in used:
                        continue
                    used.add(ids[j])
                    tr = workers.get(int(ids[j]))
                    pred = (state_at(tr, sec + 0.5) if tr else None) or MISSED
                    break
            preds.append((r.Index, pred))
    out = gt.copy()
    out["pred"] = pd.Series(dict(preds))
    return out


def metrics(df: pd.DataFrame) -> dict:
    """df: rows with columns state (GT, working/idle) and pred (working/idle/missed)."""
    m = {"worker_seconds": len(df), "accuracy": round(float((df.state == df.pred).mean()), 4) if len(df) else None}
    f1s = []
    for c in STATES:
        tp = int(((df.state == c) & (df.pred == c)).sum())
        fp = int(((df.state != c) & (df.pred == c)).sum())
        fn = int(((df.state == c) & (df.pred != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if tp + fn:
            f1s.append(f1)
        m[c] = dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4), support=tp + fn)
    m["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else None
    m["confusion"] = pd.crosstab(df.state, df.pred).to_dict()
    for c in STATES:
        true_s, pred_s = int((df.state == c).sum()), int((df.pred == c).sum())
        m[f"{c}_time"] = dict(true_s=true_s, pred_s=pred_s, abs_err_s=abs(pred_s - true_s),
                              err_pct=round(100 * abs(pred_s - true_s) / true_s, 2) if true_s else None)
    m["missed_s"] = int((df.pred == MISSED).sum())
    return m


@lru_cache(maxsize=None)
def _load(cache: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ground truth and detections, loaded once per video (the tuner evaluates hundreds of times)."""
    return pd.read_csv(ROOT / "labels" / f"{stem}.csv"), pd.read_parquet(cache / "detections.parquet")


def evaluate(cfg: dict, videos: list[Path], split: str = "all", engines: dict | None = None) -> tuple[dict, pd.DataFrame]:
    frames = []
    for v in videos:
        cache = output_dir(cfg, v)
        gt, det = _load(cache, v.stem)
        t0, t1 = split_window(cache, split)
        gt = gt[(gt.second >= t0) & (gt.second + 1 <= t1 + 1e-6)]
        engine = engines[v] if engines else replay(cfg, cache)
        m = match_predictions(engine, gt, det)
        m["video"] = v.name
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    ignored = df[df.state == "ignore"]
    scored = df[df.state.isin(STATES)]
    res = metrics(scored)
    # Non-workers (marked x) that the system counted as working/idle: inflates totals.
    res["false_worker_s"] = int(ignored.pred.isin(STATES).sum())
    return res, scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--config", default=None)
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    a = p.parse_args()
    cfg = load_config(a.config) if a.config else load_config()

    res, df = evaluate(cfg, a.videos, a.split)
    print(f"\n=== Benchmark ({a.split} split, {res['worker_seconds']} labelled worker-seconds) ===")
    print(f"accuracy {res['accuracy']}   macro-F1 {res['macro_f1']}")
    for c in STATES:
        r = res[c]
        print(f"  {c:8s} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  (n={r['support']})")
    print("\nTotal time (the headline number):")
    for c in STATES:
        t = res[f"{c}_time"]
        print(f"  {c:8s} true {t['true_s']:6d}s  predicted {t['pred_s']:6d}s  error {t['abs_err_s']}s ({t['err_pct']}%)")
    print(f"  missed worker-seconds: {res['missed_s']}   non-workers counted: {res['false_worker_s']}s")
    print("\nConfusion (rows = truth, cols = predicted):")
    print(pd.crosstab(df.state, df.pred).to_string())

    print("\nPer worker (GT track id):")
    per = df.groupby(["video", "track_id"]).apply(
        lambda g: pd.Series({"true_work": (g.state == WORKING).sum(), "pred_work": (g.pred == WORKING).sum(),
                             "true_idle": (g.state == IDLE).sum(), "pred_idle": (g.pred == IDLE).sum(),
                             "acc": round((g.state == g.pred).mean(), 3)}), include_groups=False)
    print(per.to_string())

    out = ROOT / cfg["output"]["dir"] / f"benchmark_{a.split}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
