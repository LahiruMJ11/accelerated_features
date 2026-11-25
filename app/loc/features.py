"""XFeat feature extraction for Aachen, stored in HLoc's h5 layout (keypoints in the original image frame)."""
import time

import cv2
import h5py
import numpy as np
import torch

from app import engine
from . import paths


def h5_leaf_groups(path):
    """Names of the groups that directly hold datasets (HLoc layout: db/1045.jpg, query/day/nexus4/x.jpg)."""
    names = set()
    with h5py.File(str(path), "r", libver="latest") as fd:
        fd.visititems(lambda n, o: names.add(n.strip("/")) if isinstance(o, h5py.Group) and "keypoints" in o else None)
    return names


def extract_image(model, path, max_dim=paths.MAX_DIM, top_k=paths.TOP_K):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    with engine.LOCK:
        out = model.detectAndCompute(model.parse_input(small), top_k=top_k)[0]
    kpts = out["keypoints"].cpu().numpy().astype(np.float32)
    scales = np.array([w / nw, h / nh], np.float32)
    kpts = (kpts + 0.5) * scales[None] - 0.5                   # HLoc convention (pixel centres)
    return {"keypoints": kpts, "scores": out["scores"].cpu().numpy().astype(np.float16),
            "descriptors": out["descriptors"].cpu().numpy().T.astype(np.float16),      # (64, N)
            "image_size": np.array([w, h])}, (w, h)


def run(names, state, cancel):
    """Extract the images in `names` that are not yet in the h5 file."""
    paths.OUT.mkdir(parents=True, exist_ok=True)
    model = engine.get_model()
    done = h5_leaf_groups(paths.FEATURES) if paths.FEATURES.exists() else set()
    todo = [n for n in names if n not in done]
    state.update(done=len(names) - len(todo), total=len(names))
    t0, n0 = time.time(), state["done"]
    for i, name in enumerate(todo):
        if cancel.is_set():
            return False
        pred, _ = extract_image(model, paths.IMAGES / name)
        with h5py.File(paths.FEATURES, "a", libver="latest") as fd:
            g = fd.create_group(name)
            for k, v in pred.items():
                g.create_dataset(k, data=v)
            g["keypoints"].attrs["uncertainty"] = 1.0 * (pred["image_size"].max() / paths.MAX_DIM)
        state["done"] = n0 + i + 1
        state["eta_s"] = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
    return True
