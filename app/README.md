# XFeat demo UI

Upload two photos (or pick a MegaDepth-1500 sample) and compare **XFeat** (sparse) with **XFeat\*** (semi-dense).
Both use the same checkpoint, `weights/xfeat.pt`.

```bash
conda activate myenv
pip install fastapi "uvicorn[standard]" python-multipart poselib   # plus torch, opencv, kornia
python -m uvicorn app.server:app --port 8000                        # run from the repo root
# open http://localhost:8000
```

- **Any two images** → matches, inliers (fundamental-matrix MAGSAC), inlier ratio, time.
- **MegaDepth-1500 pairs** (gallery, or upload files with their original names) → also R/t pose error vs. ground truth.
- **Benchmark** → AUC@5/10/20, Acc@10, MIR, #inliers, FPS over all 1500 pairs (poselib, 2.5 px), shown next to the paper's Table 1.
  Results are cached in `app/cache/`; an interrupted run resumes.

Needs `datasets/Mega1500/megadepth_test_1500` (see `python -m modules.dataset.download -h`).

## Homography · HPatches

Switch the task to **Homography** in "Try it" or "The numbers".

- **Per pair** (HPatches gallery or any two images): XFeat matches, MAGSAC++ homography, a registration overlay
  (image 1 warped onto image 2) and, for gallery pairs, the ground-truth vs. estimated quad and the mean corner error.
- **Benchmark**: all 116 sequences (580 pairs), MHA @3/5/7 px for Illumination and Viewpoint, next to the paper's Table 3 (XFeat only, as reported).
  The paper does not state the resize / keypoint count / threshold, so the defaults are 480 px shorter side, 4096 keypoints, 3 px (adjustable in Settings).

## Relative pose · ScanNet-1500

Pick **ScanNet-1500** next to "Relative pose". Per-pair view uses the gallery (poses from `datasets/ScanNet1500`, intrinsics from `intrinsic_color.txt`).
The benchmark matches all 1500 pairs once (cached in `app/cache/scannet_matches/`, resumable), then sweeps 12 RANSAC thresholds
(0.5–6 px) per method and reports AUC@5/10/20 at the best one, next to the paper's Table 2. Images are matched at VGA (640×480).

## Visual localization · Aachen Day-Night

Third task, "Localization" (paper Sec. 4.3, Table 4 — XFeat only). Follows HLoc: XFeat features (1024 px, 4096 keypoints),
mutual-NN matching, a 3D map triangulated from the known database poses, NetVLAD top-50 retrieval, PnP + RANSAC (12 px).

- The pipeline lives in `app/loc/` and uses the vendored HLoc in `third_party/Hierarchical-Localization/` (triangulation,
  localization and the precomputed Aachen pair lists `pairs-db-covis20` / `pairs-query-netvlad50`, so no NetVLAD weights are needed).
- `pycolmap` and `torch` ship conflicting OpenMP runtimes on macOS, so the pycolmap stages run in separate processes
  (`app/loc/colmap_worker.py`, `app/loc/service.py`). Never import both in one process.
- Run it from the UI ("The numbers" → Localization → Build map and localize) or `python -m app.loc.pipeline`. About an hour on CPU;
  every stage is cached in `app/cache/aachen/` (about 4 GB) and resumable.
- Aachen's ground-truth poses are not public: the paper's percentages come from submitting
  `app/cache/aachen/Aachen_hloc_xfeat_netvlad50.txt` to visuallocalization.net (the UI explains the steps and takes the results back).
