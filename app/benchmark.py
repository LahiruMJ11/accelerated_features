"""MegaDepth-1500 benchmark, following the paper / modules/eval/megadepth1500.py protocol:
images resized to the sizes stored in megadepth_1500.json (max dim 1200), XFeat top-4096,
XFeat* top-10000, poselib relative pose with a 2.5 px RANSAC threshold, AUC@5/10/20 over all pairs.
FPS is measured separately at VGA (extract one frame + match against a cached reference)."""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from . import engine, pairs

CACHE = Path(__file__).resolve().parent / "cache"
PARTIAL = CACHE / "megadepth1500_partial.json"
RESULT = CACHE / "megadepth1500_results.json"
METHODS = ("xfeat", "xfeat_star")

_state = {"status": "idle", "done": 0, "total": 0, "eta_s": None, "error": None, "live": None}
_cancel = threading.Event()
_thread = None


def _summ(recs):
    """recs: list of [R_err, t_err, n_matches, n_inliers] -> aggregate metrics."""
    err = [max(r[0], r[1]) for r in recs]
    auc = engine.error_auc(err)
    arr = np.array(err)
    mir = [r[3] / r[2] if r[2] else 0.0 for r in recs]
    return {"auc5": auc[5] * 100, "auc10": auc[10] * 100, "auc20": auc[20] * 100,
            "acc10": float((arr <= 10).mean() * 100),
            "mir": float(np.mean(mir)), "inliers": float(np.mean([r[3] for r in recs])),
            "matches": float(np.mean([r[2] for r in recs]))}


@torch.inference_mode()
def measure_fps(n_frames=30, warmup=3):
    """VGA FPS: per frame = extract one image + match against a cached reference (real-time demo setting)."""
    m = engine.get_model()
    ref = pairs.read(0, 0); cur = pairs.read(0, 1)
    ref = cv2.resize(ref, (640, 480)); cur = cv2.resize(cur, (640, 480))
    res = {}
    with engine.LOCK:
        x0, x1 = m.parse_input(ref), m.parse_input(cur)
        r_sp = m.detectAndCompute(x0, top_k=engine.TOP_K_SPARSE)[0]
        r_st = m.detectAndComputeDense(x0, top_k=engine.TOP_K_STAR)

        def sparse():
            o = m.detectAndCompute(x1, top_k=engine.TOP_K_SPARSE)[0]
            m.match(r_sp["descriptors"], o["descriptors"], min_cossim=-1)

        def star():
            o = m.detectAndComputeDense(x1, top_k=engine.TOP_K_STAR)
            idx = m.batch_match(r_st["descriptors"], o["descriptors"])
            m.refine_matches(r_st, o, matches=idx, batch_idx=0)

        for key, fn in (("xfeat", sparse), ("xfeat_star", star)):
            for _ in range(warmup):
                fn()
            ts = []
            for _ in range(n_frames):
                t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
            fps = 1.0 / np.array(ts)
            res[key] = {"fps": float(fps.mean()), "fps_std": float(fps.std())}
    return res


def _run(limit, thr):
    try:
        total = min(limit or pairs.count(), pairs.count())
        recs = {k: [] for k in METHODS}
        if not limit and PARTIAL.exists():
            part = json.loads(PARTIAL.read_text())
            if part.get("thr") == thr:
                recs = part["recs"]
        start = len(recs["xfeat"])
        _state.update(status="running", done=start, total=total, error=None, eta_s=None)
        t0, n0 = time.time(), start
        for i in range(start, total):
            if _cancel.is_set():
                _state["status"] = "cancelled"
                return
            m = pairs.meta(i)
            im0, im1 = pairs.read(i, 0), pairs.read(i, 1)
            for key, fn, k in (("xfeat", engine.run_sparse, engine.TOP_K_SPARSE),
                               ("xfeat_star", engine.run_star, engine.TOP_K_STAR)):
                with engine.LOCK:
                    p0, p1, _ = fn(im0, im1, k)
                e = engine.estimate(p0 * np.array(m["scale0"]), p1 * np.array(m["scale1"]), m["gt"], thr)
                recs[key].append([e["R_err"], e["t_err"], len(p0), int(e["inl"].sum())])
            _state["done"] = i + 1
            el = time.time() - t0
            _state["eta_s"] = el / (i + 1 - n0) * (total - i - 1)
            if (i + 1) % 10 == 0 or i + 1 == total:
                _state["live"] = {k: _summ(recs[k]) for k in METHODS}
                if not limit:      # test runs (limit=N) are never persisted
                    PARTIAL.write_text(json.dumps({"thr": thr, "recs": recs}))
        fps = measure_fps()
        result = {"date": datetime.now().isoformat(timespec="seconds"), "pairs": total, "ransac_thr": thr,
                  "device": engine.device_name(),
                  "methods": {k: {**_summ(recs[k]), **fps[k]} for k in METHODS}}
        if not limit:
            CACHE.mkdir(exist_ok=True)
            RESULT.write_text(json.dumps(result))
            if PARTIAL.exists():
                PARTIAL.unlink()
        _state.update(status="done", live=None, last=result)
    except Exception as ex:  # surfaced in the UI
        _state.update(status="error", error=f"{type(ex).__name__}: {ex}")


def start(limit=None, thr=engine.RANSAC_THR):
    global _thread
    if _thread and _thread.is_alive():
        return False
    CACHE.mkdir(exist_ok=True)
    _cancel.clear()
    _thread = threading.Thread(target=_run, args=(limit, thr), daemon=True)
    _thread.start()
    return True


def cancel():
    _cancel.set()


def status():
    out = dict(_state)
    out["result"] = json.loads(RESULT.read_text()) if RESULT.exists() else None
    out["resumable"] = PARTIAL.exists()
    return out
