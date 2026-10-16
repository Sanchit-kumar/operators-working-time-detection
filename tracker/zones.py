"""Work zones: polygons where a person counts as a worker (fixed cameras only).

Stored per video/camera in zones/<name>.json as normalised coordinates (0..1), so they work at any resolution:
    {"zones": [{"name": "packing table", "points": [[0.31, 0.42], [0.55, 0.40], ...]}]}
A person is inside when the bottom-centre of their box (their feet) is inside any zone.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT


def zones_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["zones"]["dir"] / f"{name}.json"


def load_zones(cfg: dict, name: str, width: int, height: int) -> list[np.ndarray] | None:
    """Pixel polygons for this video/camera, or None when no zone file exists (whole frame counts)."""
    path = zones_path(cfg, name)
    if not path.exists():
        return None
    zones = json.loads(path.read_text())["zones"]
    return [(np.array(z["points"], np.float32) * [width, height]).astype(np.float32) for z in zones]


def foot_point(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2), float(box[3])


def in_zones(box: np.ndarray, zones: list[np.ndarray] | None) -> bool:
    if zones is None:
        return True
    p = foot_point(box)
    return any(cv2.pointPolygonTest(z, p, False) >= 0 for z in zones)
