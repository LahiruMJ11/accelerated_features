"""Mutual nearest-neighbour matching of XFeat descriptors for image pairs, written in HLoc's matches h5 layout."""
import time

import h5py
import numpy as np
import torch

from . import paths


def read_pairs(path):
    return [tuple(l.split()) for l in open(path).read().splitlines() if l.strip()]


def pair_key(a, b):
    return a.replace("/", "-") + "/" + b.replace("/", "-")


def mnn(d0, d1):
    """d0 (N,64), d1 (M,64) float32, L2-normalised -> matches0 (N,) int, scores0 (N,)."""
    sim = d0 @ d1.T
    s0, m0 = sim.max(1)
    m1 = sim.argmax(0)
    ok = m1[m0] == torch.arange(len(m0))
    return torch.where(ok, m0, torch.full_like(m0, -1)), s0


class DescCache:
    """Descriptors (float16) for every image, loaded lazily and kept in memory."""

    def __init__(self, path=paths.FEATURES):
        self.path, self.d = path, {}
        self.fd = h5py.File(path, "r", libver="latest")

    def get(self, name):
        if name not in self.d:
            self.d[name] = torch.from_numpy(self.fd[name]["descriptors"].__array__().T.copy())     # (N,64) half
        return self.d[name]


def run(pairs_file, out_h5, state, cancel, only_names=None):
    pairs = read_pairs(pairs_file)
    if only_names is not None:
        pairs = [p for p in pairs if p[0] in only_names and p[1] in only_names]
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "a", libver="latest") as fd:
        have = set()
        for a in fd.keys():
            for b in fd[a].keys():
                have.add(a + "/" + b)
    todo = [p for p in pairs if pair_key(*p) not in have]
    state.update(done=len(pairs) - len(todo), total=len(pairs))
    if not todo:
        return True
    cache = DescCache()
    t0, n0 = time.time(), state["done"]
    fd = h5py.File(out_h5, "a", libver="latest")
    try:
        for i, (a, b) in enumerate(todo):
            if cancel.is_set():
                return False
            d0, d1 = cache.get(a).float(), cache.get(b).float()
            m0, s0 = mnn(d0, d1)
            g = fd.create_group(pair_key(a, b))
            g.create_dataset("matches0", data=m0.numpy().astype(np.int16))
            g.create_dataset("matching_scores0", data=s0.numpy().astype(np.float16))
            state["done"] = n0 + i + 1
            if i % 200 == 0:
                state["eta_s"] = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
                fd.flush()
    finally:
        fd.close()
        cache.fd.close()
    return True
