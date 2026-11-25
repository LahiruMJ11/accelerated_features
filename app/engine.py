"""
Inference + geometry for the XFeat UI.

Matching code paths mirror XFeat.match_xfeat / XFeat.match_xfeat_star (single pair),
but expose intermediate counts (keypoints, coarse matches) and timings.
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from modules.xfeat import XFeat

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "xfeat.pt"      # one checkpoint serves XFeat and XFeat*

TOP_K_SPARSE = 4096     # paper: XFeat
TOP_K_STAR = 10_000     # paper: XFeat*
RANSAC_THR = 2.5        # px, default of modules/eval/megadepth1500.py

LOCK = threading.RLock()
_model = None


def get_model():
    global _model
    if _model is None:
        _model = XFeat(weights=str(WEIGHTS), top_k=TOP_K_SPARSE)
    return _model


def device_name():
    return str(get_model().dev)


# ----------------------------------------------------------------- images
def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def fit_max_dim(img, max_dim):
    """Resize so the longer side ~= max_dim, both sides multiples of 32.
    Returns resized image and (sx, sy) so that original_xy = resized_xy * (sx, sy)."""
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    nw = max(32, int(round(w * s / 32)) * 32)
    nh = max(32, int(round(h * s / 32)) * 32)
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, (w / nw, h / nh)


# ---------------------------------------------------------------- matching
@torch.inference_mode()
def run_sparse(bgr0, bgr1, top_k=TOP_K_SPARSE):
    """XFeat: top-k keypoints from the detector branch + mutual-NN matching."""
    m = get_model()
    t = time.perf_counter()
    x0, x1 = m.parse_input(bgr0), m.parse_input(bgr1)
    o0 = m.detectAndCompute(x0, top_k=top_k)[0]
    o1 = m.detectAndCompute(x1, top_k=top_k)[0]
    i0, i1 = m.match(o0["descriptors"], o1["descriptors"], min_cossim=-1)
    p0 = o0["keypoints"][i0].cpu().numpy()
    p1 = o1["keypoints"][i1].cpu().numpy()
    dt = time.perf_counter() - t
    return p0, p1, {"kpts0": len(o0["keypoints"]), "kpts1": len(o1["keypoints"]), "time_ms": dt * 1e3}


@torch.inference_mode()
def run_star(bgr0, bgr1, top_k=TOP_K_STAR):
    """XFeat*: dual-scale coarse features, mutual-NN, then offset refinement MLP."""
    m = get_model()
    t = time.perf_counter()
    x0, x1 = m.parse_input(bgr0), m.parse_input(bgr1)
    o0 = m.detectAndComputeDense(x0, top_k=top_k)
    o1 = m.detectAndComputeDense(x1, top_k=top_k)
    idxs = m.batch_match(o0["descriptors"], o1["descriptors"])
    n_coarse = int(len(idxs[0][0]))
    mt = m.refine_matches(o0, o1, matches=idxs, batch_idx=0).cpu().numpy()
    dt = time.perf_counter() - t
    return mt[:, :2], mt[:, 2:], {
        "kpts0": int(o0["keypoints"].shape[1]), "kpts1": int(o1["keypoints"].shape[1]),
        "coarse_matches": n_coarse, "time_ms": dt * 1e3}


# ---------------------------------------------------------------- geometry
def _cam(K):
    px, py, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
    return {"model": "PINHOLE", "width": int(2 * px), "height": int(2 * py), "params": [fx, fy, px, py]}


def relative_pose_error(T_0to1, R, t):
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)
    R_gt = T_0to1[:3, :3]
    cos = np.clip((np.trace(np.dot(R.T, R_gt)) - 1) / 2, -1.0, 1.0)
    return float(t_err), float(np.rad2deg(np.abs(np.arccos(cos))))


def estimate(p0, p1, gt=None, thr=RANSAC_THR):
    """gt = dict(K0, K1, T_0to1) in *original* pixel frame (p0/p1 already scaled).
    Without gt: fundamental-matrix MAGSAC inliers only."""
    n = len(p0)
    res = {"inl": np.zeros(n, bool), "R_err": None, "t_err": None}
    if gt is not None:
        import poselib
        if n >= 5:
            try:
                M, info = poselib.estimate_relative_pose(
                    p0.astype(np.float64), p1.astype(np.float64), _cam(gt["K0"]), _cam(gt["K1"]),
                    {"max_epipolar_error": thr, "success_prob": 0.99999,
                     "min_iterations": 20, "max_iterations": 1000})
                res["inl"] = np.array(info["inliers"], bool)
                res["t_err"], res["R_err"] = relative_pose_error(gt["T_0to1"], M.R, M.t)
            except Exception:
                pass
        if res["R_err"] is None:
            res["R_err"] = res["t_err"] = float("inf")
    elif n >= 8:
        try:
            _, mask = cv2.findFundamentalMat(p0.astype(np.float32), p1.astype(np.float32),
                                             cv2.USAC_MAGSAC, thr, 0.99999, 10000)
            if mask is not None:
                res["inl"] = mask.ravel().astype(bool)
        except cv2.error:
            pass
    return res


def error_auc(errors, thresholds=(5, 10, 20)):
    """Same as modules/eval/megadepth1500.py (np.trapz was removed in NumPy 2)."""
    trapz = getattr(np, "trapezoid", None) or np.trapz
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    out = {}
    for thr in thresholds:
        last = np.searchsorted(errors, thr)
        y = recall[:last] + [recall[last - 1]]
        x = errors[:last] + [thr]
        out[thr] = float(trapz(y, x) / thr)
    return out


# ------------------------------------------------------------ full pipeline
def _finite(v):
    return None if v is None or not np.isfinite(v) else float(v)


def match_pair(bgr0, bgr1, scale0=(1, 1), scale1=(1, 1), gt=None, thr=RANSAC_THR,
               top_k_sparse=TOP_K_SPARSE, top_k_star=TOP_K_STAR):
    """bgr0/bgr1: already resized to the matching resolution.
    scale*: resized -> original pixel factors (only used with gt)."""
    out = {}
    for key, fn, k in (("xfeat", run_sparse, top_k_sparse), ("xfeat_star", run_star, top_k_star)):
        with LOCK:
            p0, p1, info = fn(bgr0, bgr1, k)
        t = thr[key] if isinstance(thr, dict) else thr      # thr may be a per-method dict
        if gt is not None:
            e = estimate(p0 * np.array(scale0), p1 * np.array(scale1), gt, t)
        else:
            e = estimate(p0, p1, None, t)
        n, ni = len(p0), int(e["inl"].sum())
        pose = None
        if e["R_err"] is not None:
            pose = max(e["R_err"], e["t_err"])
        out[key] = {
            "pts": np.round(np.concatenate([p0, p1], 1), 1).ravel().tolist(),
            "inl": e["inl"].astype(int).tolist(),
            "metrics": {**info, "matches": n, "inliers": ni, "mir": ni / n if n else 0.0,
                        "R_err": _finite(e["R_err"]), "t_err": _finite(e["t_err"]),
                        "pose_err": _finite(pose) if pose is not None else None,
                        "pose_failed": pose is not None and not np.isfinite(pose)},
        }
    return out


# ------------------------------------------------------------- homography
def estimate_homography(p0, p1, thr=3.0):
    """MAGSAC++ (OpenCV USAC_MAGSAC). Returns (H or None, inlier mask)."""
    n = len(p0)
    inl = np.zeros(n, bool)
    if n < 4:
        return None, inl
    try:
        H, mask = cv2.findHomography(p0.astype(np.float32), p1.astype(np.float32),
                                     cv2.USAC_MAGSAC, thr, maxIters=10000, confidence=0.9999)
    except cv2.error:
        return None, inl
    if H is None or mask is None:
        return None, inl
    return H, mask.ravel().astype(bool)


def corners(w, h):
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)


def warp_pts(H, pts):
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ H.T
    return p[:, :2] / p[:, 2:3]


def corner_error(H_est, H_gt, w, h):
    """ALIKE / HPatches protocol: mean distance (px) between the four reference-image corners
    warped with the ground-truth and with the estimated homography. inf if estimation failed."""
    if H_est is None:
        return float("inf")
    c = corners(w, h)
    return float(np.linalg.norm(warp_pts(H_est, c) - warp_pts(H_gt, c), axis=1).mean())
