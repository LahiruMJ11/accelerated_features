"""ScanNet-1500 relative pose benchmark (paper Table 2): XFeat and XFeat*, VGA images, poselib.
Matches are extracted once and cached on disk, then 12 RANSAC thresholds (0.5 ... 6.0 px) are swept as in
modules/eval/scannet1500.py; the paper searches the best threshold per method, here the one with the best
mean of AUC@5/10/20."""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import poselib
import torch

from . import engine, scannet

CACHE = Path(__file__).resolve().parent / "cache"
MATCHES = CACHE / "scannet_matches"
RESULT = CACHE / "scannet_results.json"
METHODS = ("xfeat", "xfeat_star")
THRS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

_state = {"status": "idle", "phase": None, "done": 0, "total": 0, "eta_s": None, "error": None, "live": None}
_cancel = threading.Event()
_thread = None


def _extract(i):
    """Match pair i with both methods (VGA); cache keypoints in the original 1296x968 frame."""
    files = {k: MATCHES / f"{k}_{i}.npy" for k in METHODS}
    if all(f.exists() for f in files.values()):
        return
    a, sa = scannet.read_vga(i, 0)
    b, sb = scannet.read_vga(i, 1)
    for key, fn, k in (("xfeat", engine.run_sparse, engine.TOP_K_SPARSE),
                       ("xfeat_star", engine.run_star, engine.TOP_K_STAR)):
        with engine.LOCK:
            p0, p1, _ = fn(a, b, k)
        arr = np.concatenate([p0 * np.array(sa), p1 * np.array(sb)], 1).astype(np.float32)
        np.save(files[key], arr)


def _pose_err(arr, gt, thr):
    """Same options as modules/eval/scannet1500.py (poselib, 10000 iterations, p=0.99999)."""
    if len(arr) < 5:
        return np.inf
    try:
        M, _ = poselib.estimate_relative_pose(
            arr[:, :2].astype(np.float64), arr[:, 2:].astype(np.float64),
            engine._cam(gt["K0"]), engine._cam(gt["K1"]),
            {"max_epipolar_error": thr, "success_prob": 0.99999, "max_iterations": 10000})
        t_err, R_err = engine.relative_pose_error(gt["T_0to1"], M.R, M.t)
        return max(t_err, R_err)
    except Exception:
        return np.inf


def _auc(errs):
    a = engine.error_auc(errs)
    return {"auc5": a[5] * 100, "auc10": a[10] * 100, "auc20": a[20] * 100}


def _run(limit):
    try:
        total = min(limit or scannet.count(), scannet.count())
        MATCHES.mkdir(parents=True, exist_ok=True)
        _state.update(status="running", phase="matching", done=0, total=total, error=None, eta_s=None, live=None)
        t0 = time.time()
        first = None
        for i in range(total):
            if _cancel.is_set():
                _state["status"] = "cancelled"
                return
            if first is None and not all((MATCHES / f"{k}_{i}.npy").exists() for k in METHODS):
                first = i
                t0 = time.time()
            _extract(i)
            _state["done"] = i + 1
            if first is not None and i + 1 > first:
                _state["eta_s"] = (time.time() - t0) / (i + 1 - first) * (total - i - 1)
        gts = [scannet.meta(i)["gt"] for i in range(total)]
        _state.update(phase="sweeping", done=0, total=len(THRS) * len(METHODS), eta_s=None)
        sweep = {k: {} for k in METHODS}
        unit, t1 = 0, time.time()
        for key in METHODS:
            arrs = [np.load(MATCHES / f"{key}_{i}.npy") for i in range(total)]
            for thr in THRS:
                if _cancel.is_set():
                    _state["status"] = "cancelled"
                    return
                errs = [_pose_err(arrs[i], gts[i], thr) for i in range(total)]
                sweep[key][thr] = _auc(errs)
                unit += 1
                _state["done"] = unit
                _state["eta_s"] = (time.time() - t1) / unit * (_state["total"] - unit)
        methods = {}
        for key in METHODS:
            best = max(THRS, key=lambda t: sum(sweep[key][t].values()))
            methods[key] = {**sweep[key][best], "best_thr": best,
                            "by_thr": {str(t): sweep[key][t] for t in THRS}}
        result = {"date": datetime.now().isoformat(timespec="seconds"), "pairs": total,
                  "resolution": "640x480", "device": engine.device_name(), "methods": methods}
        if not limit:
            RESULT.write_text(json.dumps(result))
        _state.update(status="done", phase=None, last=result)
    except Exception as ex:
        _state.update(status="error", error=f"{type(ex).__name__}: {ex}")


def start(limit=None):
    global _thread
    if _thread and _thread.is_alive():
        return False
    CACHE.mkdir(exist_ok=True)
    _cancel.clear()
    _thread = threading.Thread(target=_run, args=(limit,), daemon=True)
    _thread.start()
    return True


def cancel():
    _cancel.set()


def best_thresholds():
    """Per-method best threshold from the last full run, for per-pair matching."""
    if RESULT.exists():
        m = json.loads(RESULT.read_text())["methods"]
        return {k: v["best_thr"] for k, v in m.items()}
    return None


def status():
    out = dict(_state)
    out["result"] = json.loads(RESULT.read_text()) if RESULT.exists() else None
    out["resumable"] = MATCHES.exists() and any(MATCHES.iterdir())
    return out
