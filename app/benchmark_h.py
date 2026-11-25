"""HPatches homography benchmark (paper Sec. 4.2 / Table 3, XFeat sparse only).
All 116 sequences x 5 pairs = 580 pairs. Homography by MAGSAC++ (OpenCV USAC_MAGSAC); error = mean
corner distance (ALIKE protocol); MHA@t = % of pairs with error <= t px, split Illumination / Viewpoint.
Paper protocol details not stated (resize, keypoints, threshold) -> defaults: 480 px short side, 4096, 3 px."""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import engine, hpatches

CACHE = Path(__file__).resolve().parent / "cache"
PARTIAL = CACHE / "hpatches_partial.json"
RESULT = CACHE / "hpatches_results.json"
THRS = (3, 5, 7)

_state = {"status": "idle", "done": 0, "total": 0, "eta_s": None, "error": None, "live": None}
_cancel = threading.Event()
_thread = None


def summarize(recs):
    """recs: list of [group, corner_err]"""
    out = {}
    for g in ("illumination", "viewpoint"):
        e = np.array([r[1] if r[1] is not None else np.inf for r in recs if r[0] == g])
        out[g] = {"n": int(len(e)), **{f"mha{t}": float((e <= t).mean() * 100) if len(e) else None for t in THRS}}
    return out


def _run(cfg, limit):
    try:
        total = min(limit or hpatches.count(), hpatches.count())
        recs = []
        if not limit and PARTIAL.exists():
            part = json.loads(PARTIAL.read_text())
            if part.get("cfg") == cfg:
                recs = part["recs"]
        _state.update(status="running", done=len(recs), total=total, error=None, eta_s=None)
        t0, n0 = time.time(), len(recs)
        for i in range(len(recs), total):
            if _cancel.is_set():
                _state["status"] = "cancelled"
                return
            im0, im1, Hgt, meta = hpatches.load_pair(i, cfg["short_side"])
            with engine.LOCK:
                p0, p1, _ = engine.run_sparse(im0, im1, cfg["top_k"])
            H, _ = engine.estimate_homography(p0, p1, cfg["thr"])
            err = engine.corner_error(H, Hgt, im0.shape[1], im0.shape[0])
            recs.append([meta["group"], err if np.isfinite(err) else None])
            _state["done"] = i + 1
            _state["eta_s"] = (time.time() - t0) / (i + 1 - n0) * (total - i - 1)
            if (i + 1) % 20 == 0 or i + 1 == total:
                _state["live"] = summarize(recs)
                if not limit:
                    PARTIAL.write_text(json.dumps({"cfg": cfg, "recs": recs}))
        result = {"date": datetime.now().isoformat(timespec="seconds"), "pairs": total, "cfg": cfg,
                  "device": engine.device_name(), "groups": summarize(recs)}
        if not limit:
            CACHE.mkdir(exist_ok=True)
            RESULT.write_text(json.dumps(result))
            if PARTIAL.exists():
                PARTIAL.unlink()
        _state.update(status="done", live=None, last=result)
    except Exception as ex:
        _state.update(status="error", error=f"{type(ex).__name__}: {ex}")


def start(short_side=480, thr=3.0, top_k=4096, limit=None):
    global _thread
    if _thread and _thread.is_alive():
        return False
    CACHE.mkdir(exist_ok=True)
    _cancel.clear()
    cfg = {"short_side": short_side, "thr": thr, "top_k": top_k}
    _thread = threading.Thread(target=_run, args=(cfg, limit), daemon=True)
    _thread.start()
    return True


def cancel():
    _cancel.set()


def status():
    out = dict(_state)
    out["result"] = json.loads(RESULT.read_text()) if RESULT.exists() else None
    out["resumable"] = PARTIAL.exists()
    return out
