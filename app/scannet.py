"""ScanNet-1500 test pairs (indoor). Ground-truth relative pose is built from the per-frame pose files and
intrinsic_color.txt, exactly as modules/eval/scannet1500.py does. Images are matched at VGA (640x480) as in
the paper; keypoints are scaled back to the original 1296x968 frame so the original intrinsics apply."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "datasets" / "ScanNet1500" / "scannet_test_1500"
NPZ = ROOT / "datasets" / "ScanNet1500" / "test.npz"
VGA = (640, 480)

_pairs = None
_K = {}


def available():
    return DIR.exists() and NPZ.exists()


def _load():
    global _pairs
    if _pairs is None:
        names = np.load(NPZ)["name"]
        _pairs = [(f"scene{a:04d}_{b:02d}", int(c), int(d)) for a, b, c, d in names]
    return _pairs


def count():
    return len(_load())


def _relative(p0, p1):
    R0, t0, R1, t1 = p0[:3, :3], p0[:3, [3]], p1[:3, :3], p1[:3, [3]]
    return np.concatenate([R1.T @ R0, R1.T @ (t0 - t1)], axis=-1)       # 3x4, same as the repo's script


def _intr(scene):
    if scene not in _K:
        _K[scene] = np.loadtxt(DIR / scene / "intrinsic" / "intrinsic_color.txt")
    return _K[scene]


def meta(pid):
    scene, a, b = _load()[pid]
    p0 = np.loadtxt(DIR / scene / "pose" / f"{a}.txt")
    p1 = np.loadtxt(DIR / scene / "pose" / f"{b}.txt")
    K = _intr(scene)
    return {"id": pid, "scene": scene, "frames": (a, b),
            "gt": {"K0": K, "K1": K, "T_0to1": _relative(p0, p1)}}


def _raw(pid, k):
    scene, a, b = _load()[pid]
    img = cv2.imread(str(DIR / scene / "color" / f"{a if k == 0 else b}.jpg"))
    if img is None:
        raise FileNotFoundError(f"{scene}/{a if k == 0 else b}.jpg")
    return img


def read_vga(pid, k):
    """VGA image and (sx, sy) = original/resized, to bring keypoints back to the original frame."""
    img = _raw(pid, k)
    h, w = img.shape[:2]
    return cv2.resize(img, VGA, interpolation=cv2.INTER_AREA), (w / VGA[0], h / VGA[1])


def thumb(pid, k, size=360):
    img = _raw(pid, k)
    s = size / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def samples(n=24, seed=None):
    total = count()
    ids = (np.linspace(0, total - 1, n).astype(int) if seed is None
           else np.random.RandomState(seed).choice(total, n, replace=False))
    return [{"id": int(i), "scene": _load()[i][0]} for i in ids]
