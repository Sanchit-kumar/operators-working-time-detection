"""Grid-search classifier thresholds on the DEV split, then report the TEST split once.

    python -m tracker.tune data/*.mp4

Writes configs/tuned.yaml. Only the cheap stage 2 is re-run, from the detection cache.
"""

import argparse
import copy
import itertools
from pathlib import Path

import yaml

from .benchmark import evaluate, replay
from .config import ROOT, load_config, output_dir

GRID = {
    "energy_thr": [0.02, 0.04, 0.06, 0.09, 0.13],
    "wrist_thr": [0.1, 0.18, 0.25, 0.4, 0.6],
    "body_thr": [0.1, 0.2, 0.35],
    "idle_after_s": [1.5, 3.0, 5.0],
}


def objective(res: dict) -> tuple:
    # Primary: macro-F1. Tie-break: smaller total working-time error.
    err = res["working_time"]["err_pct"] or 0.0
    return (res["macro_f1"] or 0.0, -err)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    base = load_config(a.config) if a.config else load_config()

    # Features don't depend on the thresholds: extract them once, then re-run only the state machine.
    engines = {v: replay(base, output_dir(base, v), record=True) for v in a.videos}
    results = []
    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    for i, values in enumerate(combos, 1):
        cfg = copy.deepcopy(base)
        cfg["classifier"].update(dict(zip(keys, values)))
        for e in engines.values():
            e.reclassify(cfg["classifier"])
        res, _ = evaluate(cfg, a.videos, "dev", engines)
        results.append((objective(res), dict(zip(keys, values)), res))
        print(f"\r{i}/{len(combos)}", end="", flush=True)
    results.sort(key=lambda r: r[0], reverse=True)

    print("\n\nTop 5 on DEV:")
    for obj, params, res in results[:5]:
        print(f"  macro-F1 {res['macro_f1']}  working err {res['working_time']['err_pct']}%  {params}")

    best = copy.deepcopy(base)
    best["classifier"].update(results[0][1])
    test, _ = evaluate(best, a.videos, "test")
    print(f"\nBest params on TEST: macro-F1 {test['macro_f1']}  accuracy {test['accuracy']}  "
          f"working err {test['working_time']['err_pct']}%  idle err {test['idle_time']['err_pct']}%")

    best["detector"]["tracker"] = str(Path(best["detector"]["tracker"]).relative_to(ROOT))
    out = ROOT / "configs/tuned.yaml"
    out.write_text(yaml.safe_dump(best, sort_keys=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
