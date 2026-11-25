"""FastAPI server for the XFeat / XFeat* demo UI.  Run from the repo root:
    uvicorn app.server:app --port 8000"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import base64

from . import benchmark, benchmark_h, benchmark_s, engine, hpatches, pairs, scannet
from .loc import live as loc_live, paths as loc_paths, pipeline as loc_pipeline

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="XFeat UI")


@app.on_event("startup")
def _warm():
    engine.get_model()


@app.get("/api/health")
def health():
    return {"device": engine.device_name(), "weights": engine.WEIGHTS.name,
            "dataset": pairs.available(), "pairs": pairs.count() if pairs.available() else 0,
            "scannet": scannet.available(), "scannet_pairs": scannet.count() if scannet.available() else 0,
            "aachen": loc_paths.IMAGES.exists() and loc_paths.PAIRS_QUERY.exists(),
            "hpatches": hpatches.available(), "hpatches_pairs": hpatches.count() if hpatches.available() else 0}


# ------------------------------------------------------------------ pairs
@app.get("/api/pairs")
def list_pairs(n: int = 24, seed: Optional[int] = None):
    if not pairs.available():
        return []
    return pairs.samples(min(n, 60), seed)


@app.get("/api/pair/{pid}/img/{k}")
def pair_image(pid: int, k: int, thumb: bool = False):
    if not pairs.available() or not (0 <= pid < pairs.count()) or k not in (0, 1):
        raise HTTPException(404)
    img = pairs.read(pid, k)
    if thumb:
        s = 360 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# ------------------------------------------------------------------ match
@app.post("/api/match")
def match(img0: Optional[UploadFile] = File(None), img1: Optional[UploadFile] = File(None),
          pair_id: Optional[int] = Form(None), max_dim: int = Form(1200),
          ransac_thr: Optional[float] = Form(None), dataset: str = Form("megadepth"),
          top_k_sparse: int = Form(engine.TOP_K_SPARSE), top_k_star: int = Form(engine.TOP_K_STAR)):
    gt, s0, s1, source = None, (1, 1), (1, 1), "upload"
    try:
        if pair_id is not None and dataset == "scannet":   # ScanNet-1500 gallery, matched at VGA
            if not scannet.available() or not (0 <= pair_id < scannet.count()):
                raise HTTPException(404, "Unknown pair")
            m = scannet.meta(pair_id)
            im0, s0 = scannet.read_vga(pair_id, 0)
            im1, s1 = scannet.read_vga(pair_id, 1)
            gt, source = m["gt"], f"scannet:{pair_id}"
            if ransac_thr is None:
                ransac_thr = benchmark_s.best_thresholds() or 1.0     # per-method best from the benchmark
        elif pair_id is not None:                      # sample from the MegaDepth-1500 gallery
            if not pairs.available() or not (0 <= pair_id < pairs.count()):
                raise HTTPException(404, "Unknown pair")
            m = pairs.meta(pair_id)
            im0, im1 = pairs.read(pair_id, 0), pairs.read(pair_id, 1)
            gt, s0, s1, source = m["gt"], m["scale0"], m["scale1"], f"megadepth:{pair_id}"
        else:
            if img0 is None or img1 is None:
                raise HTTPException(400, "Provide two images")
            raw0, raw1 = engine.decode_image(img0.file.read()), engine.decode_image(img1.file.read())
            hit = pairs.find(img0.filename or "", img1.filename or "") if pairs.available() else None
            if hit:                                   # an uploaded MegaDepth-1500 pair -> ground truth known
                pid, sw = hit
                m = pairs.meta(pid, sw)
                im0 = cv2.resize(raw0, (m["size0_hw"][1], m["size0_hw"][0]))
                im1 = cv2.resize(raw1, (m["size1_hw"][1], m["size1_hw"][0]))
                gt, s0, s1, source = m["gt"], m["scale0"], m["scale1"], f"megadepth:{pid}"
            else:
                im0, _ = engine.fit_max_dim(raw0, max_dim)
                im1, _ = engine.fit_max_dim(raw1, max_dim)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if ransac_thr is None:
        ransac_thr = engine.RANSAC_THR
    res = engine.match_pair(im0, im1, s0, s1, gt, ransac_thr, top_k_sparse, top_k_star)
    return {"source": source, "has_gt": gt is not None,
            "size0": [im0.shape[1], im0.shape[0]], "size1": [im1.shape[1], im1.shape[0]],
            "ransac_thr": ransac_thr, "methods": res}


# ----------------------------------------------------------------- ScanNet
@app.get("/api/scannet/pairs")
def sn_pairs(n: int = 24, seed: Optional[int] = None):
    return scannet.samples(min(n, 60), seed) if scannet.available() else []


@app.get("/api/scannet/img/{pid}/{k}")
def sn_image(pid: int, k: int, thumb: bool = False):
    if not scannet.available() or not (0 <= pid < scannet.count()) or k not in (0, 1):
        raise HTTPException(404)
    img = scannet.thumb(pid, k) if thumb else scannet.read_vga(pid, k)[0]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/benchmark_s/start")
def bs_start(limit: Optional[int] = None):
    if not scannet.available():
        raise HTTPException(400, "ScanNet-1500 not found in datasets/ScanNet1500")
    return {"started": benchmark_s.start(limit)}


@app.post("/api/benchmark_s/cancel")
def bs_cancel():
    benchmark_s.cancel()
    return {"ok": True}


@app.get("/api/benchmark_s/status")
def bs_status():
    return benchmark_s.status()


# -------------------------------------------------------------- benchmark
@app.post("/api/benchmark/start")
def bench_start(limit: Optional[int] = None, thr: float = engine.RANSAC_THR):
    if not pairs.available():
        raise HTTPException(400, "MegaDepth-1500 images not found in datasets/Mega1500")
    return {"started": benchmark.start(limit, thr)}


@app.post("/api/benchmark/cancel")
def bench_cancel():
    benchmark.cancel()
    return {"ok": True}


@app.get("/api/benchmark/status")
def bench_status():
    return benchmark.status()


# ------------------------------------------------------- HPatches / homography
@app.get("/api/hpatches/pairs")
def hp_pairs(n: int = 24, seed: Optional[int] = None):
    return hpatches.samples(min(n, 60), seed) if hpatches.available() else []


@app.get("/api/hpatches/img/{pid}/{k}")
def hp_image(pid: int, k: int, thumb: bool = False):
    if not hpatches.available() or not (0 <= pid < hpatches.count()) or k not in (0, 1):
        raise HTTPException(404)
    img = hpatches.thumb(pid, k) if thumb else hpatches.full(pid, k)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


def _jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.post("/api/homography")
def homography(img0: Optional[UploadFile] = File(None), img1: Optional[UploadFile] = File(None),
               pair_id: Optional[int] = Form(None), short_side: int = Form(hpatches.SHORT_SIDE),
               thr: float = Form(3.0), top_k: int = Form(engine.TOP_K_SPARSE)):
    H_gt = None
    try:
        if pair_id is not None:
            if not hpatches.available() or not (0 <= pair_id < hpatches.count()):
                raise HTTPException(404, "Unknown pair")
            im0, im1, H_gt, meta = hpatches.load_pair(pair_id, short_side)
            source = f"hpatches:{meta['seq']}:1-{meta['k']}"
        else:
            if img0 is None or img1 is None:
                raise HTTPException(400, "Provide two images")
            im0 = hpatches.resize_short(engine.decode_image(img0.file.read()), short_side)[0]
            im1 = hpatches.resize_short(engine.decode_image(img1.file.read()), short_side)[0]
            source, meta = "upload", None
    except ValueError as e:
        raise HTTPException(400, str(e))

    with engine.LOCK:
        p0, p1, info = engine.run_sparse(im0, im1, top_k)
    H, inl = engine.estimate_homography(p0, p1, thr)
    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]
    n, ni = len(p0), int(inl.sum())
    err = engine.corner_error(H, H_gt, w0, h0) if H_gt is not None else None
    quads = {"gt": engine.warp_pts(H_gt, engine.corners(w0, h0)).round(1).tolist() if H_gt is not None else None,
             "est": engine.warp_pts(H, engine.corners(w0, h0)).round(1).tolist() if H is not None else None}
    warp = None
    if H is not None:
        w = cv2.warpPerspective(im0, H, (w1, h1))
        mask = cv2.warpPerspective(np.full((h0, w0), 255, np.uint8), H, (w1, h1)) > 0
        blend = im1.copy()
        blend[mask] = (0.5 * im1[mask] + 0.5 * w[mask]).astype(np.uint8)
        warp = _jpeg_b64(blend)
    return {"source": source, "has_gt": H_gt is not None, "meta": meta, "thr": thr,
            "size0": [w0, h0], "size1": [w1, h1],
            "method": {"pts": np.round(np.concatenate([p0, p1], 1), 1).ravel().tolist(),
                       "inl": inl.astype(int).tolist(),
                       "metrics": {**info, "matches": n, "inliers": ni, "mir": ni / n if n else 0.0,
                                   "corner_err": err if err is None or np.isfinite(err) else None,
                                   "failed": H is None or (err is not None and not np.isfinite(err))}},
            "quads": quads, "warp": warp}


@app.post("/api/benchmark_h/start")
def bh_start(short_side: int = hpatches.SHORT_SIDE, thr: float = 3.0, top_k: int = engine.TOP_K_SPARSE,
             limit: Optional[int] = None):
    if not hpatches.available():
        raise HTTPException(400, "HPatches not found in datasets/HPatches")
    return {"started": benchmark_h.start(short_side, thr, top_k, limit)}


@app.post("/api/benchmark_h/cancel")
def bh_cancel():
    benchmark_h.cancel()
    return {"ok": True}


@app.get("/api/benchmark_h/status")
def bh_status():
    return benchmark_h.status()


# ------------------------------------------------- Aachen visual localization
import json as _json


@app.get("/api/aachen/status")
def aachen_status():
    st = loc_pipeline.status()
    stats_f = loc_paths.OUT / "loc_stats.json"
    summary = None
    if stats_f.exists():
        stats = _json.loads(stats_f.read_text())
        summary = {}
        for split in ("day", "night"):
            rows = [v for v in stats.values() if v["split"] == split]
            inl = [v["inliers"] for v in rows if v["localized"]]
            summary[split] = {"queries": len(rows), "localized": sum(v["localized"] for v in rows),
                              "median_inliers": float(sorted(inl)[len(inl) // 2]) if inl else 0.0,
                              "mean_inliers": float(sum(inl) / len(inl)) if inl else 0.0}
    st["summary"] = summary
    st["ready"] = loc_live.ready()
    st["has_submission"] = loc_paths.RESULTS.exists()
    sr = loc_paths.SERVER_RESULTS
    st["server_results"] = _json.loads(sr.read_text()) if sr.exists() else None
    return st


@app.post("/api/aachen/start")
def aachen_start():
    if not loc_paths.IMAGES.exists():
        raise HTTPException(400, "Aachen images not found in datasets/Aachen/images_upright")
    busy = loc_pipeline.other_running() is not None
    return {"started": loc_pipeline.start(), "busy": busy}


@app.post("/api/aachen/cancel")
def aachen_cancel():
    loc_pipeline.cancel()
    return {"ok": True}


@app.get("/api/aachen/submission")
def aachen_submission():
    if not loc_paths.RESULTS.exists():
        raise HTTPException(404, "Run the localization first")
    return FileResponse(loc_paths.RESULTS, media_type="text/plain", filename="Aachen_hloc_xfeat_netvlad50.txt")


@app.post("/api/aachen/server_results")
def aachen_server_results(body: dict):
    try:
        out = {k: [float(x) for x in body[k]] for k in ("day", "night")}
        assert all(len(v) == 3 for v in out.values())
    except Exception:
        raise HTTPException(400, "Expected {day: [a, b, c], night: [a, b, c]}")
    loc_paths.OUT.mkdir(parents=True, exist_ok=True)
    loc_paths.SERVER_RESULTS.write_text(_json.dumps(out))
    return out


@app.delete("/api/aachen/server_results")
def aachen_server_results_clear():
    if loc_paths.SERVER_RESULTS.exists():
        loc_paths.SERVER_RESULTS.unlink()
    return {"ok": True}


@app.get("/api/aachen/queries")
def aachen_queries(split: Optional[str] = None, n: int = 24, seed: Optional[int] = None):
    names = loc_live.query_names(split)
    if seed is None:
        ids = [names[i] for i in np.linspace(0, len(names) - 1, min(n, len(names))).astype(int)]
    else:
        ids = [names[i] for i in np.random.RandomState(seed).choice(len(names), min(n, len(names)), replace=False)]
    return [{"name": q, "split": loc_live._load_queries()[q][0]} for q in ids]


@app.get("/api/aachen/img/{name:path}")
def aachen_image(name: str, thumb: bool = False):
    f = (loc_paths.IMAGES / name).resolve()
    if loc_paths.IMAGES.resolve() not in f.parents or not f.exists():
        raise HTTPException(404)
    img = cv2.imread(str(f), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    s = (360 if thumb else 1024) / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/aachen/map")
def aachen_map():
    if not loc_live.ready():
        raise HTTPException(409, "The 3D map is not built yet")
    return loc_live.map_json()


@app.post("/api/aachen/localize")
def aachen_localize(body: dict):
    if not loc_live.ready():
        raise HTTPException(409, "The 3D map and query matches are not built yet — run the pipeline first")
    try:
        return loc_live.localize(body["name"])
    except KeyError:
        raise HTTPException(404, "Unknown query")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
