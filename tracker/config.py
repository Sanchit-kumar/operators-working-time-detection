from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    """Read a YAML config; `extends: other.yaml` (relative to this file) inherits and overrides another one."""
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    return _merge(_read(path.parent / parent), cfg) if parent else cfg


def load_config(path: str | Path = ROOT / "configs/default.yaml") -> dict:
    cfg = _read(Path(path))
    tracker_cfg = Path(cfg["detector"]["tracker"])
    if not tracker_cfg.is_absolute():
        cfg["detector"]["tracker"] = str(ROOT / tracker_cfg)
    return cfg


def output_dir(cfg: dict, video: str | Path) -> Path:
    d = ROOT / cfg["output"]["dir"] / Path(video).stem
    d.mkdir(parents=True, exist_ok=True)
    return d
