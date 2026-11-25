"""MegaDepth-1500 pair index (ground-truth poses + intrinsics) and image loading."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "assets" / "megadepth_1500.json"
IMG_ROOT = ROOT / "datasets" / "Mega1500" / "megadepth_test_1500"

_pairs = None
_by_names = None


def _load():
    global _pairs, _by_names
    if _pairs is None:
        _pairs = json.loads(JSON.read_text())
        _by_names = {}
        for i, p in enumerate(_pairs):
            a, b = (Path(n).name for n in p["pair_names"])
            _by_names[(a, b)] = (i, False)
            _by_names[(b, a)] = (i, True)
    return _pairs


def available():
    return IMG_ROOT.exists()


def count():
    return len(_load())


def meta(pid, swapped=False):
    """Ground-truth meta of a pair, optionally with the two images swapped."""
    p = _load()[pid]
    K0, K1 = np.array(p["K0"]), np.array(p["K1"])
    s0, s1 = p["scale0"], p["scale1"]
    h0, h1 = p["size0_hw"], p["size1_hw"]
    names = p["pair_names"]
    T = np.array(p["T_0to1"])
    if swapped:
        K0, K1, s0, s1, h0, h1 = K1, K0, s1, s0, h1, h0
        names = names[::-1]
        T = np.array(p["T_1to0"])
    return {"id": pid, "scene": p["scene_id"], "names": names, "size0_hw": h0, "size1_hw": h1,
            "scale0": s0, "scale1": s1, "gt": {"K0": K0, "K1": K1, "T_0to1": T}}


def find(name0, name1):
    """Look up an uploaded pair by file basenames -> (pid, swapped) or None."""
    _load()
    return _by_names.get((Path(name0).name, Path(name1).name))


def read(pid, k, swapped=False):
    """Image k (0/1) of a pair, resized to the protocol resolution (size_hw)."""
    m = meta(pid, swapped)
    img = cv2.imread(str(IMG_ROOT / m["names"][k]))
    if img is None:
        raise FileNotFoundError(m["names"][k])
    h, w = m["size0_hw" if k == 0 else "size1_hw"]
    return cv2.resize(img, (w, h))


def samples(n=24, seed=None):
    total = count()
    if seed is None:
        ids = np.linspace(0, total - 1, n).astype(int)
    else:
        ids = np.random.RandomState(seed).choice(total, n, replace=False)
    return [{"id": int(i), "scene": _pairs[i]["scene_id"]} for i in ids]
