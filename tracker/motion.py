"""Camera-motion compensation and per-person motion energy.

Motion is always measured against the frame ~`interval_s` earlier (not simply the previous frame), so the
signals mean the same thing at 6 fps and at 25 fps and one set of thresholds works for every video.

For moving cameras we estimate the global (background) motion between frames, chain it into a cumulative
transform, warp the reference frame onto the current one, and measure the residual change inside each box.
"""

from collections import deque

import cv2
import numpy as np


def to3x3(a: np.ndarray) -> np.ndarray:
    return np.vstack([a, [0.0, 0.0, 1.0]])


def nearest_in_time(buffer, target_t: float):
    """The buffer entry (tuples starting with t) whose time is closest to target_t."""
    return min(buffer, key=lambda e: abs(e[0] - target_t))


class MotionEstimator:
    def __init__(self, work_width: int = 960, diff_threshold: int = 18, fixed_camera: bool = False,
                 interval_s: float = 0.2):
        self.fixed_camera = fixed_camera  # skip camera-motion estimation (fixed CCTV): faster, no estimation noise
        self.work_width = work_width
        self.diff_threshold = diff_threshold
        self.interval_s = interval_s
        self.scale = None
        self.cum = np.eye(3)               # cumulative camera transform, first frame -> current (work coords)
        self.buffer = deque()              # (t, gray, cum) of recent frames

    def update(self, frame: np.ndarray, boxes: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Process one frame.

        Args:
            frame: BGR frame.
            boxes: (N, 4) person boxes x1, y1, x2, y2 in `frame` pixels.
            t: frame time in seconds.

        Returns:
            affine: 2x3 matrix mapping previous-frame coordinates to current-frame coordinates (`frame` pixels).
            energy: (N,) fraction of pixels in each box that changed over ~interval_s, camera-compensated.
        """
        if self.scale is None:
            self.scale = self.work_width / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small_boxes = (boxes * self.scale).astype(int) if len(boxes) else np.zeros((0, 4), int)

        step = np.eye(2, 3, dtype=np.float32)
        if self.buffer and not self.fixed_camera:
            step = self._estimate_affine(self.buffer[-1][1], gray, small_boxes)
        self.cum = to3x3(step) @ self.cum
        self.buffer.append((t, gray, self.cum.copy()))
        while len(self.buffer) > 2 and self.buffer[1][0] <= t - 2 * self.interval_s:
            self.buffer.popleft()

        energy = np.zeros(len(boxes), np.float32)
        if len(self.buffer) > 1:
            t_ref, ref, cum_ref = nearest_in_time(list(self.buffer)[:-1], t - self.interval_s)
            warp = (self.cum @ np.linalg.inv(cum_ref))[:2].astype(np.float32)  # reference -> current frame
            warped = cv2.warpAffine(ref, warp, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            changed = cv2.absdiff(gray, warped) > self.diff_threshold
            h, w = gray.shape
            for i, (x1, y1, x2, y2) in enumerate(small_boxes):
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                if x2 > x1 and y2 > y1:
                    energy[i] = changed[y1:y2, x1:x2].mean()

        # Same step transform expressed in `frame` coordinates (translation scales, rotation does not).
        affine = step.copy()
        affine[:, 2] /= self.scale
        return affine, energy

    @staticmethod
    def _estimate_affine(prev: np.ndarray, cur: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        mask = np.full(prev.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:  # people are not background
            mask[max(y1, 0):y2, max(x1, 0):x2] = 0
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)
        identity = np.eye(2, 3, dtype=np.float32)
        if pts is None or len(pts) < 10:
            return identity
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
        good = status.ravel() == 1
        if good.sum() < 10:
            return identity
        affine, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                ransacReprojThreshold=2.0)
        return identity if affine is None else affine.astype(np.float32)
