"""VLM classifier: answer parsing and how the engine uses external decisions."""

import numpy as np

from tracker import IDLE, WORKING
from tracker.config import load_config
from tracker.engine import ActivityEngine
from tracker.vlm import parse_answer
from test_engine import FPS, make_frame


def test_parse_answer_variants():
    assert parse_answer('{"state": "idle", "confidence": 0.8, "reason": "standing"}') == ("idle", 0.8, "standing")
    assert parse_answer('Here you go:\n```json\n{"state": "working", "confidence": 1.4}\n```')[:2] == ("working", 1.0)
    assert parse_answer('{"state": "not_a_person", "confidence": 0.9}')[0] == "not_a_person"
    assert parse_answer("The worker is idle.")[0] == "idle"
    assert parse_answer("")[0] is None


def run_with_vlm(decisions, energy=0.3, seconds=20):
    """decisions: list of (t, state) for track 1; motion energy is constant (would say working)."""
    t = np.array([d[0] for d in decisions], float)
    ext = {1: (t, [d[1] for d in decisions], np.full(len(t), 0.9))}
    cfg = load_config("configs/vlm.yaml")
    engine = ActivityEngine(cfg, frame_dt=1 / FPS, external=ext)
    for i in range(int(seconds * FPS)):
        engine.update(make_frame(i, energy))
    return engine


def test_vlm_overrides_motion():
    # Motion says working the whole time; the VLM says idle from 10 s on -> idle is used.
    dec = [(t, WORKING) for t in (2, 4, 6, 8, 10)] + [(t, IDLE) for t in (12, 14, 16, 18, 20)]
    tot = run_with_vlm(dec).tracks[1].totals()
    assert abs(tot[WORKING] - 10) < 1.5 and abs(tot[IDLE] - 10) < 1.5, tot


def test_not_a_person_is_not_counted():
    eng = run_with_vlm([(t, "not_a_person") for t in range(2, 21, 2)])
    assert eng.workers() == []


def test_motion_fallback_without_decisions():
    eng = run_with_vlm([(100.0, IDLE)])  # no decision near the clip: motion (working) is used
    assert eng.tracks[1].totals()[WORKING] > 15
