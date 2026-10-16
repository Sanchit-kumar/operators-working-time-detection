"""Stage-2 tests on synthetic detections (no GPU, no video)."""

import numpy as np
import pandas as pd

from tracker import IDLE, WORKING
from tracker.benchmark import metrics
from tracker.config import load_config
from tracker.detect import FrameDetections
from tracker.engine import ActivityEngine

FPS = 25.0


def make_frame(i: int, energy: float, tid: int = 1) -> FrameDetections:
    kpts = np.zeros((1, 17, 3))
    return FrameDetections(i, i / FPS, np.array([tid]), np.array([[100.0, 100.0, 200.0, 300.0]]), np.array([0.9]),
                           kpts, np.array([energy]), np.eye(2, 3, dtype=np.float32))


def run(energies: list[float]) -> ActivityEngine:
    engine = ActivityEngine(load_config(), frame_dt=1 / FPS)
    for i, e in enumerate(energies):
        engine.update(make_frame(i, e))
    return engine


def seconds(pattern: list[tuple[float, float]]) -> list[float]:
    """[(energy, seconds), ...] -> per-frame energies."""
    return [e for e, s in pattern for _ in range(int(s * FPS))]


def test_work_idle_work_totals():
    eng = run(seconds([(0.3, 10), (0.0, 20), (0.3, 10)]))
    tr = eng.tracks[1]
    states = [s.state for s in tr.segments]
    assert states == [WORKING, IDLE, WORKING]
    tot = tr.totals()
    # Window median delays the change by ~window/2 (1 s); retroactive switching removes the hold delay.
    assert abs(tot[WORKING] - 20) < 2.5, tot
    assert abs(tot[IDLE] - 20) < 2.5, tot
    assert abs(sum(tot.values()) - 40) < 0.2


def test_short_pause_stays_working():
    # A 1.5 s pause is shorter than idle_after_s, so it must not create an idle segment.
    eng = run(seconds([(0.3, 10), (0.0, 1.5), (0.3, 10)]))
    assert [s.state for s in eng.tracks[1].segments] == [WORKING]


def test_short_track_is_not_a_worker():
    eng = run(seconds([(0.3, 1.0)]))
    assert eng.workers() == []


def test_metrics_perfect_and_errors():
    df = pd.DataFrame({"state": [WORKING] * 8 + [IDLE] * 2, "pred": [WORKING] * 7 + [IDLE] * 3})
    m = metrics(df)
    assert m["accuracy"] == 0.9
    assert m["working_time"] == dict(true_s=8, pred_s=7, abs_err_s=1, err_pct=12.5)
    assert m[IDLE]["recall"] == 1.0


def test_reclassify_matches_fresh_run():
    energies = seconds([(0.3, 10), (0.04, 8), (0.0, 6), (0.3, 5)])
    new = dict(load_config()["classifier"], energy_thr=0.03, idle_after_s=1.5)

    recorded = ActivityEngine(load_config(), frame_dt=1 / FPS, record=True)
    for i, e in enumerate(energies):
        recorded.update(make_frame(i, e))
    recorded.reclassify(new)

    cfg = load_config()
    cfg["classifier"] = new
    fresh = ActivityEngine(cfg, frame_dt=1 / FPS)
    for i, e in enumerate(energies):
        fresh.update(make_frame(i, e))

    assert recorded.timeline() == fresh.timeline()
    assert len(fresh.timeline()) >= 2  # thresholds actually changed the outcome


def test_gap_is_not_counted():
    # Worker visible 10 s, gone 20 s (occluded / left), back 10 s: only 20 s should be counted.
    engine = ActivityEngine(load_config(), frame_dt=1 / FPS)
    frames = list(range(int(10 * FPS))) + list(range(int(30 * FPS), int(40 * FPS)))
    for i in frames:
        engine.update(make_frame(i, 0.3))
    tot = engine.tracks[1].totals()
    assert abs(sum(tot.values()) - 20) < 0.2, tot


def test_people_outside_zones_are_not_counted():
    inside = [np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], np.float32)]   # feet at (150, 300)
    outside = [np.array([[500, 500], [900, 500], [900, 900], [500, 900]], np.float32)]
    for zones, expected in [(inside, 1), (outside, 0)]:
        engine = ActivityEngine(load_config(), frame_dt=1 / FPS, zones=zones)
        for i in range(int(5 * FPS)):
            engine.update(make_frame(i, 0.3))
        assert len(engine.workers()) == expected
