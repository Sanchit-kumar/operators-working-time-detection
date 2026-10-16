"""Camera-motion compensation and per-person motion energy.

The camera may pan or shake, so raw pixel change is not evidence of a worker moving. We estimate the
global (background) motion between frames, warp the previous frame onto the current one, and measure
the residual pixel change inside each person's box.
"""

import cv2
import numpy as np


class MotionEstimator:
    def __init__(self, work_width: int = 640, diff_threshold: int = 18):
        self.work_width = work_width
        self.diff_threshold = diff_threshold
        self.prev_gray = None
        self.scale = None

    def update(self, frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process one frame.

        Args:
            frame: BGR frame at full resolution.
            boxes: (N, 4) person boxes x1, y1, x2, y2 in full-resolution pixels.

        Returns:
            affine: 2x3 matrix mapping previous-frame full-res coordinates to current-frame coordinates.
            energy: (N,) fraction of pixels in each box that changed after camera compensation.
        """
        if self.scale is None:
            self.scale = self.work_width / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small_boxes = (boxes * self.scale).astype(int) if len(boxes) else np.zeros((0, 4), int)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.eye(2, 3, dtype=np.float32), np.zeros(len(boxes), np.float32)

        affine_small = self._estimate_affine(self.prev_gray, gray, small_boxes)
        warped = cv2.warpAffine(self.prev_gray, affine_small, (gray.shape[1], gray.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)
        changed = cv2.absdiff(gray, warped) > self.diff_threshold

        energy = np.zeros(len(boxes), np.float32)
        h, w = gray.shape
        for i, (x1, y1, x2, y2) in enumerate(small_boxes):
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if x2 > x1 and y2 > y1:
                energy[i] = changed[y1:y2, x1:x2].mean()
        self.prev_gray = gray

        # Same transform expressed in full-resolution coordinates (translation scales, rotation does not).
        affine = affine_small.copy()
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
