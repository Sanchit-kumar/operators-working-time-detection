"""Motion energy must not depend on the video frame rate."""

import numpy as np

from tracker.motion import MotionEstimator

BOX = np.array([[100.0, 100.0, 300.0, 300.0]])


def square_frames(fps: float, seconds: float = 2.0, speed_px_s: float = 100.0):
    """A white square moving right at constant speed on a black background."""
    for i in range(int(fps * seconds)):
        t = i / fps
        img = np.zeros((480, 640, 3), np.uint8)
        x = int(120 + speed_px_s * t)
        img[150:250, x:x + 60] = 255
        yield t, img


def mean_energy(fps: float) -> float:
    m = MotionEstimator(work_width=640, fixed_camera=True, interval_s=0.2)
    vals = [m.update(img, BOX, t)[1][0] for t, img in square_frames(fps)]
    return float(np.mean(vals[int(fps * 0.5):]))  # skip warm-up


def test_energy_is_fps_independent():
    e6, e25 = mean_energy(6.0), mean_energy(25.0)
    assert e6 > 0 and e25 > 0
    # 6 fps can only pick 0.167 s for a 0.2 s interval, so allow some difference; frame-to-frame differencing
    # would differ ~4x (0.167 s vs 0.04 s between frames).
    assert abs(e6 - e25) / e25 < 0.3, (e6, e25)
