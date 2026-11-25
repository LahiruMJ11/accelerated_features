"""HPatches sequences: 116 scenes (57 illumination `i_*`, 59 viewpoint `v_*`), 5 pairs each (1 -> 2..6)."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "datasets" / "HPatches" / "hpatches-sequences-release"
SHORT_SIDE = 480

_seqs = None


def available():
    return DIR.exists()


def seqs():
    global _seqs
    if _seqs is None:
        _seqs = sorted(p.name for p in DIR.iterdir() if p.is_dir() and p.name[:2] in ("i_", "v_"))
    return _seqs


def count():
    return len(seqs()) * 5


def group(seq):
    return "illumination" if seq.startswith("i_") else "viewpoint"


def pid_to(pid):
    s = seqs()
    return s[pid // 5], pid % 5 + 2          # (sequence, target image index 2..6)


def resize_short(img, short=SHORT_SIDE):
    """Resize so the shorter side == short. Returns image and (sx, sy) = resized/original."""
    h, w = img.shape[:2]
    s = short / min(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, (nw / w, nh / h)


def _read_raw(seq, idx):
    img = cv2.imread(str(DIR / seq / f"{idx}.ppm"))
    if img is None:
        raise FileNotFoundError(f"{seq}/{idx}.ppm")
    return img


def load_pair(pid, short=SHORT_SIDE):
    """-> img1, img2 (resized), H_gt in resized coordinates (img1 -> img2), meta."""
    seq, k = pid_to(pid)
    a, sa = resize_short(_read_raw(seq, 1), short)
    b, sb = resize_short(_read_raw(seq, k), short)
    H = np.loadtxt(DIR / seq / f"H_1_{k}")
    S1, S2 = np.diag([sa[0], sa[1], 1.0]), np.diag([sb[0], sb[1], 1.0])
    Hr = S2 @ H @ np.linalg.inv(S1)
    return a, b, Hr, {"id": pid, "seq": seq, "k": k, "group": group(seq)}


def samples(n=24, seed=None):
    total = count()
    ids = (np.linspace(0, total - 1, n).astype(int) if seed is None
           else np.random.RandomState(seed).choice(total, n, replace=False))
    out = []
    for i in ids:
        seq, k = pid_to(int(i))
        out.append({"id": int(i), "seq": seq, "k": k, "group": group(seq)})
    return out


def thumb(pid, which, size=360):
    seq, k = pid_to(pid)
    img = _read_raw(seq, 1 if which == 0 else k)
    s = size / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def full(pid, which, short=SHORT_SIDE):
    seq, k = pid_to(pid)
    return resize_short(_read_raw(seq, 1 if which == 0 else k), short)[0]
