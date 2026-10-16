"""Stage 2: per-track motion features -> Working/Idle state machine -> timeline.

Runs online (one frame at a time), so the same code drives batch processing, the benchmark and the live view.
"""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import IDLE, WORKING
from .detect import FrameDetections
from .motion import nearest_in_time, to3x3
from .zones import in_zones, load_zones

L_WRIST, R_WRIST = 9, 10
BODY_KPTS = [5, 6, 11, 12]  # shoulders and hips


def apply_affine(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ affine[:, :2].T + affine[:, 2]


@dataclass
class Segment:
    start: float
    end: float
    state: str


@dataclass
class Track:
    track_id: int
    first_t: float
    last_t: float = 0.0
    conf_sum: float = 0.0
    n_obs: int = 0
    box: np.ndarray = None
    kpts: np.ndarray = None
    history: deque = field(default_factory=deque)  # (t, energy, wrist_speed, body_speed) over window_s
    past: deque = field(default_factory=deque)     # (t, kpts, box, camera_cum) over ~2 x motion interval
    state: str | None = None
    pending: str | None = None
    pending_since: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    signals: dict = field(default_factory=dict)  # latest windowed features, for display/debug
    log: list = field(default_factory=list)      # (t, energy, wrist, body, ready, gap) per frame, when recording

    def totals(self) -> dict[str, float]:
        out = {WORKING: 0.0, IDLE: 0.0}
        for s in self.segments:
            out[s.state] += s.end - s.start
        return out

    @property
    def duration(self) -> float:
        return self.last_t - self.first_t


class ActivityEngine:
    def __init__(self, cfg: dict, frame_dt: float, record: bool = False, zones: list[np.ndarray] | None = None,
                 external: dict | None = None):
        """record=True keeps every track's feature history so reclassify() can try new thresholds cheaply.
        zones: pixel polygons (see tracker/zones.py); people outside them are not counted.
        external: per-track decisions from another classifier (the VLM, see tracker/vlm.py):
            {track_id: (decision times, states, confidences)}. Used instead of the motion thresholds where
            available; "not_a_person" removes the box from the counts."""
        self.record = record
        self.zones = zones
        self.external = external
        vlm = cfg.get("vlm", {})
        self.ext_hold = vlm.get("interval_s", 2.0)       # a decision covers the clip before it and holds this long
        self.ext_clip = vlm.get("clip_s", 2.0)
        self.ext_min_conf = vlm.get("min_confidence", 0.0)
        self.outside: list[np.ndarray] = []  # boxes of people outside the work zones in the last frame
        self.f = cfg["features"]
        self.c = cfg["classifier"]
        self.kpt_conf = cfg["motion"]["kpt_conf"]
        self.interval = cfg["motion"]["interval_s"]
        self.cum = np.eye(3)  # cumulative camera transform (first frame -> current), full-res pixels
        self.dt = frame_dt
        self.tracks: dict[int, Track] = {}
        self.t = 0.0

    # ---------- public ----------

    def update(self, d: FrameDetections) -> list[Track]:
        self.t = d.t
        self.cum = to3x3(d.affine) @ self.cum
        visible, self.outside = [], []
        for i, tid in enumerate(d.ids):
            tid = int(tid)
            box, kpts = d.boxes[i], d.kpts[i]
            ext = self.external_state(tid, d.t)
            if not in_zones(box, self.zones) or ext == "not_a_person":
                self.outside.append(box)
                continue
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, first_t=d.t, last_t=d.t)
            gap = d.t - tr.last_t
            wrist, body = self._speeds(tr, box, kpts, d.t)
            tr.past.append((d.t, kpts, box, self.cum.copy()))
            while len(tr.past) > 2 and tr.past[1][0] <= d.t - 2 * self.interval:
                tr.past.popleft()
            tr.history.append((d.t, float(d.energy[i]), wrist, body))
            while tr.history and tr.history[0][0] < d.t - self.f["window_s"]:
                tr.history.popleft()
            tr.box, tr.kpts, tr.last_t = box, kpts, d.t
            tr.conf_sum += float(d.confs[i])
            tr.n_obs += 1
            energy, wrist, body = np.median(np.array([x[1:] for x in tr.history]), axis=0)
            ready = len(tr.history) >= 3  # wait for a few samples before the first decision
            if self.record:
                tr.log.append((d.t, energy, wrist, body, ready, gap, ext))
            self._step(tr, d.t, energy, wrist, body, ready, gap, ext)
            visible.append(tr)
        return visible

    def reclassify(self, classifier_cfg: dict) -> None:
        """Re-run only the state machine with new thresholds (requires record=True)."""
        self.c = classifier_cfg
        for tr in self.tracks.values():
            tr.state, tr.pending, tr.segments = None, None, []
            for t, energy, wrist, body, ready, gap, ext in tr.log:
                self._step(tr, t, energy, wrist, body, ready, gap, ext)

    def external_state(self, tid: int, t: float) -> str | None:
        """The external (VLM) decision covering time t for this track, or None (fall back to motion)."""
        if not self.external or tid not in self.external:
            return None
        times, states, confs = self.external[tid]
        j = int(np.searchsorted(times, t))  # first decision at or after t: its clip contains t
        if j < len(times) and times[j] - t <= self.ext_clip:
            k = j
        elif j > 0 and t - times[j - 1] <= self.ext_hold:  # otherwise hold the previous decision
            k = j - 1
        else:
            return None
        return states[k] if confs[k] >= self.ext_min_conf else None

    def workers(self) -> list[Track]:
        """Tracks that count as workers: long and confident enough (filters brief false detections)."""
        return [t for t in self.tracks.values()
                if t.segments and t.duration >= self.f["min_track_s"]
                and t.conf_sum / t.n_obs >= self.f["min_track_conf"]]

    def summary(self) -> dict:
        per_worker = {}
        total = {WORKING: 0.0, IDLE: 0.0}
        for tr in self.workers():
            tot = tr.totals()
            per_worker[tr.track_id] = {k: round(v, 2) for k, v in tot.items()}
            for k in total:
                total[k] += tot[k]
        observed = sum(total.values())
        return dict(
            workers=len(per_worker),
            working_s=round(total[WORKING], 2),
            idle_s=round(total[IDLE], 2),
            working_pct=round(100 * total[WORKING] / observed, 1) if observed else 0.0,
            per_worker=per_worker,
        )

    def timeline(self) -> list[dict]:
        rows = []
        for tr in self.workers():
            for s in tr.segments:
                rows.append(dict(track_id=tr.track_id, start_s=round(s.start, 3), end_s=round(s.end, 3),
                                 state=s.state))
        return rows

    # ---------- internals ----------

    def _speeds(self, tr: Track, box, kpts, t) -> tuple[float, float]:
        """Wrist and body speed in body-sizes per second, measured over ~interval_s and camera-compensated."""
        if not tr.past:
            return 0.0, 0.0
        t_ref, kpts_ref, box_ref, cum_ref = nearest_in_time(tr.past, t - self.interval)
        dt = t - t_ref
        if dt <= 0 or dt > 3 * self.interval:  # track was not seen recently: no reliable reference
            return 0.0, 0.0
        warp = (self.cum @ np.linalg.inv(cum_ref))[:2]  # reference frame -> current frame (camera motion only)
        scale = max(box[2] - box[0], box[3] - box[1], 1.0)
        prev = apply_affine(warp, kpts_ref[:, :2])
        ok = (kpts[:, 2] > self.kpt_conf) & (kpts_ref[:, 2] > self.kpt_conf)
        disp = np.linalg.norm(kpts[:, :2] - prev, axis=1) / scale / dt

        wrists = [disp[k] for k in (L_WRIST, R_WRIST) if ok[k]]
        wrist = float(np.mean(wrists)) if wrists else 0.0

        body_ok = [k for k in BODY_KPTS if ok[k]]
        if body_ok:
            body = float(np.mean(disp[body_ok]))
        else:  # fall back to box centre
            c_now = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            c_prev = apply_affine(warp, np.array([[(box_ref[0] + box_ref[2]) / 2, (box_ref[1] + box_ref[3]) / 2]]))[0]
            body = float(np.linalg.norm(c_now - c_prev) / scale / dt)
        # Wrist motion should mean hands moving relative to the body, not the whole person walking.
        wrist = max(wrist - body, 0.0) if wrists else 0.0
        return wrist, body

    def _step(self, tr: Track, t: float, energy: float, wrist: float, body: float, ready: bool,
              gap: float, ext: str | None = None) -> None:
        """Threshold the windowed signals and advance the track's Working/Idle state machine.

        gap: seconds since this track was last seen. Time while unseen (occluded, outside the zone) is not
        counted: the open segment stays closed where the track was last seen and a new one starts now.
        """
        score = max(energy / self.c["energy_thr"], wrist / self.c["wrist_thr"], body / self.c["body_thr"])
        tr.signals = dict(energy=energy, wrist=wrist, body=body, score=score, ext=ext)
        if ext in (WORKING, IDLE):  # external (VLM) decision takes precedence over the motion thresholds
            raw, ready = ext, True
        else:
            raw = WORKING if score >= 1.0 else IDLE

        if tr.state is None:
            if ready:
                tr.state = raw
                tr.segments.append(Segment(tr.first_t, t + self.dt, raw))
            return

        if gap > self.f["max_gap_s"]:
            tr.segments.append(Segment(t, t + self.dt, tr.state))
            tr.pending = None
        seg = tr.segments[-1]
        seg.end = t + self.dt
        if raw == tr.state:
            tr.pending = None
            return
        if tr.pending != raw:
            tr.pending, tr.pending_since = raw, t
        hold = self.c["idle_after_s"] if raw == IDLE else self.c["work_after_s"]
        if t - tr.pending_since >= hold:
            # Switch retroactively: the new state actually began when the change was first seen.
            seg.end = tr.pending_since
            tr.segments.append(Segment(tr.pending_since, t + self.dt, raw))
            tr.state, tr.pending = raw, None


def engine_for_cache(cfg: dict, cache: Path, record: bool = False) -> ActivityEngine:
    """An engine set up for a video's detection cache (frame timing from meta.json, zones by video name)."""
    meta = json.loads((cache / "meta.json").read_text())
    zones = load_zones(cfg, cache.name, meta["width"], meta["height"])
    external = None
    if cfg["classifier"].get("mode", "motion") == "vlm":
        from .vlm import load_predictions, predictions_path

        external = load_predictions(cfg, cache)
        if external is None:
            raise FileNotFoundError(f"No VLM predictions {predictions_path(cfg, cache)}: run python -m tracker.vlm first")
    return ActivityEngine(cfg, frame_dt=meta["stride"] / meta["fps"], record=record, zones=zones, external=external)
