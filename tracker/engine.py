"""Stage 2: per-track motion features -> Working/Idle state machine -> timeline.

Runs online (one frame at a time), so the same code drives batch processing, the benchmark and the live view.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import IDLE, WORKING
from .detect import FrameDetections

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
    history: deque = field(default_factory=deque)  # (t, energy, wrist_speed, body_speed)
    state: str | None = None
    pending: str | None = None
    pending_since: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    signals: dict = field(default_factory=dict)  # latest windowed features, for display/debug
    log: list = field(default_factory=list)      # (t, energy, wrist, body, ready) per frame, when recording

    def totals(self) -> dict[str, float]:
        out = {WORKING: 0.0, IDLE: 0.0}
        for s in self.segments:
            out[s.state] += s.end - s.start
        return out

    @property
    def duration(self) -> float:
        return self.last_t - self.first_t


class ActivityEngine:
    def __init__(self, cfg: dict, frame_dt: float, record: bool = False):
        """record=True keeps every track's feature history so reclassify() can try new thresholds cheaply."""
        self.record = record
        self.f = cfg["features"]
        self.c = cfg["classifier"]
        self.kpt_conf = cfg["motion"]["kpt_conf"]
        self.dt = frame_dt
        self.tracks: dict[int, Track] = {}
        self.t = 0.0

    # ---------- public ----------

    def update(self, d: FrameDetections) -> list[Track]:
        self.t = d.t
        visible = []
        for i, tid in enumerate(d.ids):
            tid = int(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, first_t=d.t)
            box, kpts = d.boxes[i], d.kpts[i]
            wrist, body = self._speeds(tr, box, kpts, d.affine, d.t)
            tr.history.append((d.t, float(d.energy[i]), wrist, body))
            while tr.history and tr.history[0][0] < d.t - self.f["window_s"]:
                tr.history.popleft()
            tr.box, tr.kpts, tr.last_t = box, kpts, d.t
            tr.conf_sum += float(d.confs[i])
            tr.n_obs += 1
            energy, wrist, body = np.median(np.array([x[1:] for x in tr.history]), axis=0)
            ready = len(tr.history) >= 3  # wait for a few samples before the first decision
            if self.record:
                tr.log.append((d.t, energy, wrist, body, ready))
            self._step(tr, d.t, energy, wrist, body, ready)
            visible.append(tr)
        return visible

    def reclassify(self, classifier_cfg: dict) -> None:
        """Re-run only the state machine with new thresholds (requires record=True)."""
        self.c = classifier_cfg
        for tr in self.tracks.values():
            tr.state, tr.pending, tr.segments = None, None, []
            for t, energy, wrist, body, ready in tr.log:
                self._step(tr, t, energy, wrist, body, ready)

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

    def _speeds(self, tr: Track, box, kpts, affine, t) -> tuple[float, float]:
        """Wrist and body speed in body-sizes per second, camera-compensated."""
        if tr.kpts is None or t - tr.last_t > 3 * self.dt:
            return 0.0, 0.0
        dt = t - tr.last_t
        scale = max(box[2] - box[0], box[3] - box[1], 1.0)
        prev = apply_affine(affine, tr.kpts[:, :2])  # where the previous keypoints would be if only the camera moved
        ok = (kpts[:, 2] > self.kpt_conf) & (tr.kpts[:, 2] > self.kpt_conf)
        disp = np.linalg.norm(kpts[:, :2] - prev, axis=1) / scale / dt

        wrists = [disp[k] for k in (L_WRIST, R_WRIST) if ok[k]]
        wrist = float(np.mean(wrists)) if wrists else 0.0

        body_ok = [k for k in BODY_KPTS if ok[k]]
        if body_ok:
            body = float(np.mean(disp[body_ok]))
        else:  # fall back to box centre
            c_now = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            c_prev = apply_affine(affine, np.array([[(tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2]]))[0]
            body = float(np.linalg.norm(c_now - c_prev) / scale / dt)
        # Wrist motion should mean hands moving relative to the body, not the whole person walking.
        wrist = max(wrist - body, 0.0) if wrists else 0.0
        return wrist, body

    def _step(self, tr: Track, t: float, energy: float, wrist: float, body: float, ready: bool) -> None:
        """Threshold the windowed signals and advance the track's Working/Idle state machine."""
        score = max(energy / self.c["energy_thr"], wrist / self.c["wrist_thr"], body / self.c["body_thr"])
        tr.signals = dict(energy=energy, wrist=wrist, body=body, score=score)
        raw = WORKING if score >= 1.0 else IDLE

        if tr.state is None:
            if ready:
                tr.state = raw
                tr.segments.append(Segment(tr.first_t, t + self.dt, raw))
            return

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
