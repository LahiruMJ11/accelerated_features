"""Live single-query localization: XFeat mutual-NN matching against the retrieved database images (torch, here),
then PnP + RANSAC against the triangulated map (pycolmap, in the helper process of app.loc.service)."""
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from multiprocessing.connection import Listener

import h5py
import numpy as np
import torch

from . import matching, paths

_lock = threading.Lock()
_proc = None
_conn = None
_retrieval = None
_queries = None
_map = None


def _load_retrieval():
    global _retrieval
    if _retrieval is None:
        d = defaultdict(list)
        for q, r in matching.read_pairs(paths.PAIRS_QUERY):
            d[q].append(r)
        _retrieval = dict(d)
    return _retrieval


def _load_queries():
    """name -> (split, camera dict)"""
    global _queries
    if _queries is None:
        _queries = {}
        for split, f in paths.QUERY_FILES.items():
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                p = line.split()
                _queries[p[0]] = (split, {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                                           "params": [float(x) for x in p[4:]]})
    return _queries


def query_names(split=None):
    return [n for n, (s, _) in _load_queries().items() if split in (None, s)]


def ready():
    return paths.SFM.joinpath("points3D.bin").exists() and paths.MATCHES_QUERY.exists()


def _start():
    global _proc, _conn
    if _conn is not None and _proc.poll() is None:
        return
    key = os.urandom(16)
    lst = Listener(("127.0.0.1", 0), authkey=key)
    _proc = subprocess.Popen([sys.executable, "-m", "app.loc.service", str(lst.address[1]), key.hex()],
                             cwd=paths.ROOT, stdout=open(paths.OUT / "service.log", "w"), stderr=subprocess.STDOUT)
    _conn = lst.accept()
    msg = _conn.recv()                      # blocks until the map is loaded
    assert msg.get("ready"), msg
    lst.close()


def load_map():
    """Map data for the top-down view (built by the helper on first start)."""
    global _map
    if _map is None:
        with _lock:
            _start()
        z = np.load(paths.OUT / "map.npz")
        proj = lambda x: (np.asarray(x) - z["mu"]) @ z["basis"].T
        _map = {"mu": z["mu"], "basis": z["basis"], "names": [str(n) for n in z["names"]],
                "centers2": proj(z["centers"]), "pts2": proj(z["pts"])}
    return _map


def map_json():
    m = load_map()
    return {"centers": np.round(m["centers2"], 1).tolist(), "points": np.round(m["pts2"], 1).tolist(),
            "names": m["names"]}


def _desc(fd, name):
    return torch.from_numpy(fd[name]["descriptors"].__array__().T.copy()).float()     # (N,64)


def localize(qname, top_k=50):
    qs = _load_queries()
    if qname not in qs:
        raise KeyError(qname)
    split, cam = qs[qname]
    retr = _load_retrieval().get(qname, [])[:top_k]
    t0 = time.time()
    with h5py.File(paths.FEATURES, "r", libver="latest") as fd:
        kpq = fd[qname]["keypoints"].__array__().astype(np.float64)
        qsize = fd[qname]["image_size"].__array__().tolist()
        dq = _desc(fd, qname)
        matches, dbkp, dbsize = {}, {}, {}
        for dbn in retr:
            m0, _ = matching.mnn(dq, _desc(fd, dbn))
            m0 = m0.numpy()
            idx = np.nonzero(m0 != -1)[0]
            matches[dbn] = np.stack([idx, m0[idx]], 1)
        match_ms = (time.time() - t0) * 1e3
        with _lock:
            _start()
            _conn.send({"op": "localize", "kpq": kpq, "camera": cam, "matches": matches, "thr": 12})
            res = _conn.recv()
        if not res.get("ok"):
            return {"name": qname, "split": split, "ok": False, "error": res.get("error"),
                    "num_matches": res.get("num_matches", 0), "retrieved": len(retr),
                    "timings": {"match_ms": match_ms, "pnp_ms": res.get("pnp_ms", 0)}}
        mp = load_map()
        c2 = ((np.array(res["center"]) - mp["mu"]) @ mp["basis"].T)
        f2 = np.array(res["forward"]) @ mp["basis"].T
        f2 = f2 / (np.linalg.norm(f2) + 1e-9)
        best = res["best"]
        pairs = np.array(res["best_pairs"], int).reshape(-1, 3)
        kd = fd[best]["keypoints"].__array__().astype(np.float64)
        line = np.concatenate([kpq[pairs[:, 0]], kd[pairs[:, 1]]], 1) if len(pairs) else np.zeros((0, 4))
        bsize = fd[best]["image_size"].__array__().tolist()
    per_db = sorted(res["per_db"], key=lambda d: -d["inliers"])[:8]
    names = {n: i for i, n in enumerate(mp["names"])}
    for d in per_db:
        d["center"] = np.round(mp["centers2"][names[d["name"]]], 1).tolist()
    return {"name": qname, "split": split, "ok": True, "retrieved": len(retr), "image_size": qsize,
            "num_matches": res["num_matches"], "num_corr": res["num_corr"], "num_inliers": res["num_inliers"],
            "inlier_ratio": res["num_inliers"] / max(1, res["num_corr"]),
            "timings": {"match_ms": match_ms, "pnp_ms": res["pnp_ms"], "total_ms": (time.time() - t0) * 1e3},
            "pose": {"center": np.round(c2, 2).tolist(), "forward": np.round(f2, 3).tolist()},
            "per_db": per_db,
            "best": {"name": best, "image_size": bsize, "pts": np.round(line, 1).ravel().tolist(),
                     "inl": pairs[:, 2].tolist()}}
