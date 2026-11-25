"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Homography estimation on HPatches (paper Sec. 4.2, Table 3), following the ALIKE protocol:
    the homography is estimated from the matches with MAGSAC++, the four corners of the reference image are warped
    with the ground-truth and the estimated homography, and a pair counts as correct when the mean corner error is
    below 3 / 5 / 7 pixels. Mean Homography Accuracy (MHA) is reported for the illumination and viewpoint sequences.

    Note: the paper does not state the resize / keypoint count / threshold; the defaults below
    (480 px short side, 4096 keypoints, 3 px MAGSAC++ threshold) are our assumptions.
"""

import argparse
import os
import time

import cv2
import numpy as np
import tqdm

# Table 3 of the paper (XFeat, MHA %)
PAPER = {'illumination': [95.0, 98.1, 98.8], 'viewpoint': [68.6, 81.1, 86.1]}
THRESHOLDS = (3, 5, 7)


def resize_short(img, short):
    """Resize so the shorter side equals `short`. Returns the image and (sx, sy) = resized/original."""
    h, w = img.shape[:2]
    s = short / min(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), (nw / w, nh / h)


def corner_error(H_est, H_gt, w, h):
    """Mean distance (px) between the reference-image corners warped with H_gt and with H_est."""
    if H_est is None:
        return np.inf
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    warp = lambda H: (lambda p: p[:, :2] / p[:, 2:3])(np.c_[c, np.ones(4)] @ H.T)
    return float(np.linalg.norm(warp(H_est) - warp(H_gt), axis=1).mean())


def estimate_homography(p0, p1, thr):
    """MAGSAC++ (OpenCV USAC_MAGSAC)."""
    if len(p0) < 4:
        return None
    H, _ = cv2.findHomography(p0.astype(np.float32), p1.astype(np.float32), cv2.USAC_MAGSAC,
                              thr, maxIters=10000, confidence=0.9999)
    return H


def run_benchmark(matcher_fn, root, short_side, thr, max_pairs=-1):
    seqs = sorted(d for d in os.listdir(root) if d[:2] in ('i_', 'v_') and os.path.isdir(os.path.join(root, d)))
    jobs = [(s, k) for s in seqs for k in range(2, 7)]
    if max_pairs > 0:
        jobs = jobs[:max_pairs]

    errors = {'illumination': [], 'viewpoint': []}
    for seq, k in tqdm.tqdm(jobs, desc='HPatches'):
        im1, s1 = resize_short(cv2.imread(f'{root}/{seq}/1.ppm'), short_side)
        im2, s2 = resize_short(cv2.imread(f'{root}/{seq}/{k}.ppm'), short_side)
        H = np.loadtxt(f'{root}/{seq}/H_1_{k}')
        H_gt = np.diag([s2[0], s2[1], 1.0]) @ H @ np.linalg.inv(np.diag([s1[0], s1[1], 1.0]))   # to resized coordinates

        p0, p1 = matcher_fn(im1, im2)
        err = corner_error(estimate_homography(p0, p1, thr), H_gt, im1.shape[1], im1.shape[0])
        errors['illumination' if seq.startswith('i_') else 'viewpoint'].append(err)
    return errors


def report(errors, args):
    print(f"\nHPatches homography estimation — XFeat (sparse)   [{sum(map(len, errors.values()))} pairs, "
          f"{args.short_side} px short side, {args.top_k} keypoints, MAGSAC++ {args.thr} px]\n")
    head = f"{'MHA (%)':<22}" + "".join(f"{g[:5].capitalize() + ' @' + str(t) + 'px':>14}" for g in errors for t in THRESHOLDS)
    print(head)
    print("-" * len(head))
    ours = [100.0 * np.mean(np.array(e) <= t) if len(e) else float('nan') for e in errors.values() for t in THRESHOLDS]
    paper = PAPER['illumination'] + PAPER['viewpoint']
    print(f"{'XFeat (paper)':<22}" + "".join(f"{v:>14.1f}" for v in paper))
    print(f"{'XFeat (this run)':<22}" + "".join(f"{v:>14.1f}" for v in ours))
    print(f"{'difference':<22}" + "".join(f"{o - p:>+14.1f}" for o, p in zip(ours, paper)))
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="HPatches homography benchmark for XFeat")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to hpatches-sequences-release (contains i_* and v_* folders)")
    parser.add_argument('--short-side', type=int, default=480, help="Resize so the shorter image side has this size")
    parser.add_argument('--thr', type=float, default=3.0, help="MAGSAC++ threshold in pixels")
    parser.add_argument('--top-k', type=int, default=4096, help="Number of XFeat keypoints")
    parser.add_argument('--max-pairs', type=int, default=-1, help="Only run the first N pairs (quick test)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    from modules.xfeat import XFeat
    xfeat = XFeat(top_k=args.top_k)

    t0 = time.time()
    errors = run_benchmark(lambda a, b: xfeat.match_xfeat(a, b, top_k=args.top_k),
                           args.dataset_dir, args.short_side, args.thr, args.max_pairs)
    report(errors, args)
    print(f"done in {time.time() - t0:.0f} s")
