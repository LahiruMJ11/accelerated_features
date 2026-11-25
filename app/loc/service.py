"""pycolmap helper process for live localization: loads the triangulated 3D map once and answers PnP requests.
Runs as its own process (pycolmap and torch ship conflicting OpenMP runtimes) -- never import torch here.
Started by app.loc.live:   python -m app.loc.service <port> <authkey-hex>"""
import sys
import time
from collections import defaultdict
from multiprocessing.connection import Client

import numpy as np

from . import paths


def build_map(recon, out):
    """Top-down view: PCA of the database camera centres (the ground plane), plus a sample of 3D points."""
    names, centers = [], []
    for im in recon.images.values():
        names.append(im.name)
        centers.append(np.asarray(im.projection_center(), float))
    centers = np.array(centers)
    mu = centers.mean(0)
    _, _, vt = np.linalg.svd(centers - mu, full_matrices=False)
    basis = vt[:2]
    ids = list(recon.points3D.keys())[::25]
    pts = np.array([recon.points3D[i].xyz for i in ids])
    np.savez(out, centers=centers, names=np.array(names), pts=pts, mu=mu, basis=basis)


class Map:
    def __init__(self):
        import pycolmap
        self.pycolmap = pycolmap
        self.recon = pycolmap.Reconstruction(str(paths.SFM))
        self.name_to_id = {im.name: i for i, im in self.recon.images.items()}
        self.p3d = {}

    def point_ids(self, db_name):
        if db_name not in self.p3d:
            img = self.recon.images[self.name_to_id[db_name]]
            self.p3d[db_name] = np.array([p.point3D_id if p.has_point3D() else -1 for p in img.points2D])
        return self.p3d[db_name]

    def localize(self, req):
        pc, R = self.pycolmap, self.recon
        t0 = time.time()
        kpq = np.asarray(req["kpq"], np.float64) + 0.5                      # COLMAP pixel convention
        cam = pc.Camera(model=req["camera"]["model"], width=req["camera"]["width"],
                        height=req["camera"]["height"], params=req["camera"]["params"])
        kp3d = defaultdict(list)
        kp3d_db = defaultdict(lambda: defaultdict(list))
        per_db, total = [], 0
        for dbn, m in req["matches"].items():
            m = np.asarray(m, int).reshape(-1, 2)
            pids = self.point_ids(dbn)
            ok = pids[m[:, 1]] != -1 if len(m) else np.zeros(0, bool)
            mm = m[ok]
            total += len(mm)
            per_db.append({"name": dbn, "matches": int(len(m)), "with3d": int(len(mm)), "inliers": 0})
            for qi, di in mm:
                j = int(pids[di])
                kp3d_db[int(qi)][j].append(dbn)
                if j not in kp3d[int(qi)]:
                    kp3d[int(qi)].append(j)
        idxs = list(kp3d.keys())
        mkp = [i for i in idxs for _ in kp3d[i]]
        m3d = [j for i in idxs for j in kp3d[i]]
        out = {"ok": False, "num_corr": len(mkp), "num_matches": total, "per_db": per_db}
        if mkp:
            ret = pc.estimate_and_refine_absolute_pose(
                kpq[mkp], [R.points3D[j].xyz for j in m3d], cam,
                estimation_options={"ransac": {"max_error": float(req.get("thr", 12))}})
            if ret is not None:
                cfw = ret["cam_from_world"]
                Rm, t = cfw.rotation.matrix(), np.asarray(cfw.translation, float)
                inl = np.asarray(ret["inlier_mask"], bool)
                cnt = defaultdict(int)
                for k in np.nonzero(inl)[0]:
                    for dbn in kp3d_db[mkp[k]][m3d[k]]:
                        cnt[dbn] += 1
                for d in per_db:
                    d["inliers"] = cnt.get(d["name"], 0)
                inl_set = {(mkp[k], m3d[k]) for k in np.nonzero(inl)[0]}
                best = max(per_db, key=lambda d: d["inliers"])["name"] if per_db else None
                pairs = []
                if best is not None:
                    m = np.asarray(req["matches"][best], int).reshape(-1, 2)
                    pids = self.point_ids(best)
                    for qi, di in m:
                        pairs.append([int(qi), int(di), int((int(qi), int(pids[di])) in inl_set)])
                out.update(ok=True, num_inliers=int(ret["num_inliers"]), center=(-Rm.T @ t).tolist(),
                           forward=(Rm.T @ np.array([0, 0, 1.0])).tolist(), best=best, best_pairs=pairs,
                           cam_from_world=[Rm.tolist(), t.tolist()])
        out["pnp_ms"] = (time.time() - t0) * 1e3
        return out


def serve(port, authkey):
    conn = Client(("127.0.0.1", port), authkey=authkey)
    mp = Map()
    npz = paths.OUT / "map.npz"
    if not npz.exists():
        build_map(mp.recon, npz)
    conn.send({"ready": True, "images": len(mp.recon.images), "points": len(mp.recon.points3D)})
    while True:
        try:
            req = conn.recv()
        except EOFError:
            break
        if req.get("op") == "localize":
            try:
                conn.send(mp.localize(req))
            except Exception as ex:
                conn.send({"ok": False, "error": f"{type(ex).__name__}: {ex}"})
        elif req.get("op") == "quit":
            break


if __name__ == "__main__":
    serve(int(sys.argv[1]), bytes.fromhex(sys.argv[2]))
