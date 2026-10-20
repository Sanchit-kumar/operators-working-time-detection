"""Zero-shot Working / Idle classification with a vision-language model (VLM).

For every tracked person, every `vlm.interval_s` seconds, the VLM sees:
  1. one full CCTV frame with the person in a red box (scene context), and
  2. `vlm.n_frames` close-up crops of that person over the last `vlm.clip_s` seconds,
and answers {"state": working | idle | not_a_person, "confidence", "reason"}. No training or tuning: the
definitions are in the prompt. The engine then uses these decisions instead of the motion thresholds
(classifier.mode: vlm), with the same smoothing.

Backends (vlm.backend):
  hf      Hugging Face transformers in this process (any Qwen-VL size; optional 4-bit quantisation)
  openai  any OpenAI-compatible server, e.g. vLLM on a GPU machine:
              vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}'

    python -m tracker.vlm "data/x.mp4" --config configs/vlm.yaml                 # predictions only
    python -m tracker.vlm "data/x.mp4" --config configs/vlm.yaml --limit 20 --save-samples 20   # quick check
    python -m tracker.run "data/x.mp4" --config configs/vlm.yaml                 # predictions (if missing) + report

Predictions are saved per model in outputs/<video>/vlm_<model>.parquet, so models can be compared.
"""

import argparse
import base64
import io
import json
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import load_config, output_dir
from .detect import iter_video

VLM_STATES = ("working", "idle", "not_a_person")

SYSTEM_PROMPT = "You are an expert analyst of factory CCTV footage. You classify what one specific person is doing."

USER_PROMPT = """Image 1 is a CCTV frame of a factory floor; the person of interest is inside the RED box.
Images 2 to {last} are close-up crops of that same person over the last {clip:.0f} seconds, in time order.

Classify the person's activity during these seconds:
- "working": doing any work task, e.g. using a computer or doing desk work, packing, sorting, scanning, handling, \
carrying or moving items, pushing a cart or pallet jack, operating equipment or tools, writing, inspecting items, \
walking with purpose to fetch or deliver something. Standing or sitting still while doing hand work or computer \
work counts as working.
- "idle": not doing a work task, e.g. standing or sitting with empty hands, waiting, looking around, chatting \
without working, using a personal phone, resting.
- "not_a_person": the red box does not contain a real person (e.g. a pile of sacks or boxes, clothing, a reflection).

Reply with JSON only, no other text:
{{"state": "working" | "idle" | "not_a_person", "confidence": <number 0.0 to 1.0>, "reason": "<at most 12 words>"}}"""


# ---------------------------------------------------------------- queries

@dataclass
class Query:
    track_id: int
    t: float                                   # decision time: the end of the clip
    samples: list[tuple[int, np.ndarray]]      # (frame index, box) per crop, oldest first
    images: list[bytes] = field(default_factory=list)  # JPEG: context frame, then crops


def plan_queries(det: pd.DataFrame, cfg: dict) -> list[Query]:
    """One query every interval_s per track, using the track's own detections for the crops."""
    v = cfg["vlm"]
    queries = []
    for tid, g in det.sort_values("t").groupby("track_id"):
        ts = g.t.to_numpy()
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        frames = g.frame.to_numpy()
        if ts[-1] - ts[0] < v["min_track_s"]:
            continue
        for tq in np.arange(ts[0] + min(v["clip_s"], ts[-1] - ts[0]), ts[-1] + 1e-6, v["interval_s"]):
            samples = []
            for st in np.linspace(tq - v["clip_s"], tq, v["n_frames"]):
                j = int(np.abs(ts - st).argmin())
                if abs(ts[j] - st) <= 0.5 and (not samples or samples[-1][0] != frames[j]):
                    samples.append((int(frames[j]), boxes[j]))
            if samples:
                queries.append(Query(int(tid), float(tq), samples))
    return queries


def _jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _crop(frame: np.ndarray, box: np.ndarray, margin: float, size: int) -> np.ndarray:
    """Person crop with context margin (hands, table, items), resized so the longer side is `size`."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - margin * bw)), int(min(w, x2 + margin * bw))
    y1, y2 = int(max(0, y1 - margin * bh)), int(min(h, y2 + margin * bh))
    crop = frame[y1:y2, x1:x2]
    s = size / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def _context(frame: np.ndarray, box: np.ndarray, width: int) -> np.ndarray:
    s = width / frame.shape[1]
    img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = (box * s).astype(int)
    pad = 4
    cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (0, 0, 255), 3)
    return img


def build_images(video: Path, queries: list[Query], cfg: dict, stride: int) -> None:
    """One sequential pass over the video, cutting every crop the queries need (stored as JPEG)."""
    v = cfg["vlm"]
    need: dict[int, list[tuple[Query, int]]] = {}
    for q in queries:
        for k, (f, _) in enumerate(q.samples):
            need.setdefault(f, []).append((q, k))
    crops: dict[tuple[int, int], bytes] = {}
    contexts: dict[int, bytes] = {}
    last_frame = max(need) if need else -1
    for idx, _, frame in tqdm(iter_video(video, stride), desc="crops", total=last_frame // stride + 1):
        if idx > last_frame:
            break
        for q, k in need.get(idx, []):
            box = q.samples[k][1]
            crops[(id(q), k)] = _jpeg(_crop(frame, box, v["crop_margin"], v["crop_size"]))
            if k == len(q.samples) - 1:
                contexts[id(q)] = _jpeg(_context(frame, box, v["context_width"]), 85)
    for q in queries:
        q.images = [contexts[id(q)]] + [crops[(id(q), k)] for k in range(len(q.samples))]


# ---------------------------------------------------------------- backends

def user_text(cfg: dict, n_images: int) -> str:
    return USER_PROMPT.format(last=n_images, clip=cfg["vlm"]["clip_s"])


class HFBackend:
    """Hugging Face transformers, in process. quantize: none | 4bit | 8bit (bitsandbytes, CUDA only)."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        v = cfg["vlm"]
        self.cfg, self.batch_size, self.max_new_tokens = cfg, v["batch_size"], v["max_new_tokens"]
        self.verbose = True
        kwargs = dict(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
        if v.get("quantize") in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=v["quantize"] == "4bit", load_in_8bit=v["quantize"] == "8bit",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.processor = AutoProcessor.from_pretrained(v["model"])
        self.processor.tokenizer.padding_side = "left"  # batched generation
        self.model = AutoModelForImageTextToText.from_pretrained(v["model"], **kwargs).eval()

    def _messages(self, q: Query) -> list[dict]:
        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in q.images]
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [*({"type": "image", "image": im} for im in images),
                                         {"type": "text", "text": user_text(self.cfg, len(images))}]},
        ]

    def __call__(self, queries: list[Query]) -> list[str]:
        import torch

        out = []
        for i in tqdm(range(0, len(queries), self.batch_size), desc="vlm", disable=not self.verbose):
            batch = [self._messages(q) for q in queries[i:i + self.batch_size]]
            inputs = self.processor.apply_chat_template(batch, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            out += self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


class OpenAIBackend:
    """OpenAI-compatible chat completions server (vLLM, SGLang, TGI, LM Studio, ...)."""

    def __init__(self, cfg: dict):
        import requests

        v = cfg["vlm"]
        self.cfg, self.session = cfg, requests.Session()
        self.url = v["base_url"].rstrip("/") + "/chat/completions"
        self.model, self.concurrency, self.max_new_tokens = v["model"], v["concurrency"], v["max_new_tokens"]
        self.headers = {"Authorization": f"Bearer {v.get('api_key') or 'EMPTY'}"}
        self.verbose = True
        self._check_server(v["base_url"])

    def _check_server(self, base_url: str) -> None:
        """Fail early with a clear message if the server is down or serves a different model."""
        try:
            r = self.session.get(base_url.rstrip("/") + "/models", headers=self.headers, timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise SystemExit(f"Cannot reach the VLM server at {base_url} ({type(e).__name__}). Is `vllm serve` "
                             f"running, and the SSH tunnel open (ssh -N -L 8000:localhost:8000 user@a100)?")
        served = [m["id"] for m in r.json().get("data", [])]
        if served and self.model not in served:
            raise SystemExit(f"The server serves {served}, but vlm.model is {self.model!r}. Set vlm.model to one of them.")

    def _one(self, q: Query) -> str:
        images = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
                  for b in q.images]
        body = dict(model=self.model, temperature=0.0, max_tokens=self.max_new_tokens, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [*images, {"type": "text", "text": user_text(self.cfg, len(images))}]},
        ])
        for attempt in range(3):
            try:
                r = self.session.post(self.url, json=body, headers=self.headers, timeout=300)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # transient server errors: retry, then record the failure
                if attempt == 2:
                    return f"ERROR: {e}"
                time.sleep(2 ** attempt)

    def __call__(self, queries: list[Query]) -> list[str]:
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(tqdm(pool.map(self._one, queries), total=len(queries), desc="vlm", disable=not self.verbose))


BACKENDS = {"hf": HFBackend, "openai": OpenAIBackend}


# ---------------------------------------------------------------- parsing and storage

def parse_answer(text: str) -> tuple[str | None, float, str]:
    """(state, confidence, reason) from the model's reply; tolerant of extra text around the JSON."""
    m = re.search(r"\{.*?\}", text, re.S)
    state, conf, reason = None, 0.5, ""
    if m:
        try:
            d = json.loads(m.group(0))
            state = str(d.get("state", "")).lower()
            conf = float(d.get("confidence", 0.5))
            reason = str(d.get("reason", ""))
        except (ValueError, TypeError):
            pass
    low = (state or text).lower()
    if "not_a_person" in low or "not a person" in low:
        state = "not_a_person"
    elif "idle" in low:
        state = "idle"
    elif "work" in low:
        state = "working"
    else:
        state = None
    return state, float(np.clip(conf, 0, 1)), reason


def predictions_path(cfg: dict, cache: Path) -> Path:
    return cache / f"vlm_{Path(cfg['vlm']['model']).name}.parquet"


def load_predictions(cfg: dict, cache: Path) -> dict[int, tuple[np.ndarray, list[str], np.ndarray]] | None:
    """Per track: (decision times, states, confidences), or None if this model hasn't been run."""
    path = predictions_path(cfg, cache)
    if not path.exists():
        return None
    df = pd.read_parquet(path).dropna(subset=["state"]).sort_values("t")
    return {int(tid): (g.t.to_numpy(), g.state.tolist(), g.confidence.to_numpy()) for tid, g in df.groupby("track_id")}


def run_vlm(cfg: dict, video: Path, limit: int | None = None, save_samples: int = 0) -> Path:
    cache = output_dir(cfg, video)
    meta = json.loads((cache / "meta.json").read_text())
    det = pd.read_parquet(cache / "detections.parquet")
    queries = plan_queries(det, cfg)
    if limit:  # spread a quick check over the whole video
        queries = [queries[i] for i in np.linspace(0, len(queries) - 1, min(limit, len(queries))).astype(int)]
    print(f"{video.name}: {len(queries)} VLM queries ({cfg['vlm']['backend']}: {cfg['vlm']['model']})")
    backend = BACKENDS[cfg["vlm"]["backend"]](cfg)  # first, so a missing server / model fails before the crop pass
    build_images(video, queries, cfg, meta["stride"])

    if save_samples:  # what the model sees, for inspection
        sample_dir = cache / "vlm_samples"
        sample_dir.mkdir(exist_ok=True)
        for q in queries[:save_samples]:
            imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in q.images]
            h = max(i.shape[0] for i in imgs)
            strip = np.hstack([cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 4, cv2.BORDER_CONSTANT) for i in imgs])
            cv2.imwrite(str(sample_dir / f"track{q.track_id}_t{q.t:07.1f}.jpg"), strip)

    start = time.time()
    answers = backend(queries)
    elapsed = time.time() - start
    failed = sum(a.startswith("ERROR") for a in answers)
    if failed > len(answers) / 2:  # don't save a file of errors: reports would silently fall back to motion
        raise SystemExit(f"{failed}/{len(answers)} VLM requests failed, e.g. {next(a for a in answers if a.startswith('ERROR'))}")
    if failed:
        print(f"warning: {failed}/{len(answers)} VLM requests failed; those times fall back to the motion classifier")
    rows = []
    for q, a in zip(queries, answers):
        state, conf, reason = parse_answer(a)
        rows.append(dict(track_id=q.track_id, t=q.t, state=state, confidence=conf, reason=reason, raw=a))
    df = pd.DataFrame(rows, columns=["track_id", "t", "state", "confidence", "reason", "raw"])
    path = predictions_path(cfg, cache)
    df.to_parquet(path)
    print(f"saved {path.name}: {len(df)} answers in {elapsed:.0f}s ({len(df) / max(elapsed, 1e-6):.2f}/s), "
          f"unparsed {df.state.isna().sum()}, states {df.state.value_counts().to_dict()}")
    return path


class LiveVLM:
    """VLM decisions for the live view, computed on a background thread so the video never waits.

    Call observe() every frame. It keeps a few recent crops per visible worker and, every interval_s, queues a
    query. Answers are appended to `self.decisions` ({track_id: (times, states, confidences)}), which the
    engine reads as its external classifier. Decisions arrive with the model's latency; the engine holds the
    latest one for vlm.live_hold_s and falls back to motion when there is none.
    """

    def __init__(self, cfg: dict):
        self.cfg, v = cfg, cfg["vlm"]
        self.backend = BACKENDS[v["backend"]](cfg)
        self.backend.verbose = False
        self.decisions: dict[int, tuple[list, list, list]] = {}
        self.crops: dict[int, list[tuple[float, bytes]]] = {}   # recent crops per track
        self.next_query: dict[int, float] = {}
        self.pending: set[int] = set()
        self.queue: queue.Queue = queue.Queue()
        self.crop_every = v["clip_s"] / max(v["n_frames"] - 1, 1)
        threading.Thread(target=self._worker, daemon=True).start()

    def observe(self, frame: np.ndarray, tracks, t: float) -> None:
        v = self.cfg["vlm"]
        for tr in tracks:
            tid = tr.track_id
            buf = self.crops.setdefault(tid, [])
            if not buf or t - buf[-1][0] >= self.crop_every - 1e-3:
                buf.append((t, _jpeg(_crop(frame, tr.box, v["crop_margin"], v["crop_size"]))))
                del buf[:-v["n_frames"]]
            if t >= self.next_query.get(tid, tr.first_t + min(v["clip_s"], 1.0)) and tid not in self.pending:
                q = Query(tid, t, [], [_jpeg(_context(frame, tr.box, v["context_width"]), 85)] + [b for _, b in buf])
                self.pending.add(tid)
                self.next_query[tid] = t + v["interval_s"]
                self.queue.put(q)

    def _worker(self):
        while True:
            batch = [self.queue.get()]
            while not self.queue.empty() and len(batch) < self.cfg["vlm"]["batch_size"]:
                batch.append(self.queue.get())
            try:
                answers = self.backend(batch)
            except Exception as e:  # keep the live view running; motion covers the gap
                print(f"VLM error: {e}")
                answers = [""] * len(batch)
            for q, a in zip(batch, answers):
                state, conf, _ = parse_answer(a)
                if state:
                    times, states, confs = self.decisions.get(q.track_id, ([], [], []))
                    self.decisions[q.track_id] = (times + [q.t], states + [state], confs + [conf])  # atomic swap
                self.pending.discard(q.track_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--config", default="configs/vlm.yaml")
    p.add_argument("--limit", type=int, help="only N queries spread over the video (quick check)")
    p.add_argument("--save-samples", type=int, default=0, help="save the first N query images for inspection")
    a = p.parse_args()
    cfg = load_config(a.config)
    for video in a.videos:
        cache = output_dir(cfg, video)
        if not (cache / "detections.parquet").exists():
            from .run import detect_stage

            detect_stage(cfg, video, cache)
        run_vlm(cfg, video, a.limit, a.save_samples)


if __name__ == "__main__":
    main()
