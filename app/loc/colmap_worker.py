"""pycolmap-based stages, run in their own process (pycolmap and torch ship conflicting OpenMP runtimes on macOS,
so this module must never import torch).   python -m app.loc.colmap_worker <reference|triangulation|localization>"""
import json
import pickle
import sys

import numpy as np

from . import paths     # also puts the vendored HLoc on sys.path


def reference():
    from hloc.colmap_from_nvm import read_nvm_model
    from hloc.utils.read_write_model import write_model
    names = paths.db_names()
    ids = {n: i + 1 for i, n in enumerate(names)}         # one camera per database image
    model = read_nvm_model(paths.NVM, paths.DB_INTRINSICS, ids, ids, skip_points=True)
    paths.REF_MODEL.mkdir(parents=True, exist_ok=True)
    write_model(*model, path=str(paths.REF_MODEL), ext=".bin")
    print(f"reference model: {len(names)} images")


def triangulation():
    """HLoc's triangulation.main, with one fix for pycolmap 4.x: the geometrically verified pairs must be stored as
    CALIBRATED two-view geometries (HLoc leaves them 'undefined', which this COLMAP version ignores -> 0 points)."""
    import sqlite3
    import pycolmap
    from hloc import triangulation as tri
    from hloc.utils.parsers import parse_retrieval  # noqa: F401  (imported by tri)

    database = paths.SFM / "database.db"
    paths.SFM.mkdir(parents=True, exist_ok=True)
    reference = pycolmap.Reconstruction(str(paths.REF_MODEL))
    image_ids = tri.create_db_from_model(reference, database)
    with pycolmap.Database.open(database) as db:
        tri.import_features(image_ids, db, paths.FEATURES)
        tri.import_matches(image_ids, db, paths.PAIRS_DB, paths.MATCHES_DB, None, False)
    with pycolmap.Database.open(database) as db:
        tri.geometric_verification(image_ids, reference, db, paths.FEATURES, paths.PAIRS_DB, paths.MATCHES_DB)
    con = sqlite3.connect(database)
    con.execute("UPDATE two_view_geometries SET config=2 WHERE rows>0")      # 2 = CALIBRATED
    con.commit()
    con.close()
    rec = tri.run_triangulation(paths.SFM, database, paths.IMAGES, reference, False, None)
    print(rec.summary())
    assert rec.num_points3D() > 1000, "triangulation produced (almost) no 3D points"


def localization():
    from hloc import localize_sfm
    localize_sfm.main(paths.SFM, paths.DATA / "queries" / "*_time_queries_with_intrinsics.txt", paths.PAIRS_QUERY,
                      paths.FEATURES, paths.MATCHES_QUERY, paths.RESULTS, covisibility_clustering=False)
    stats()


def stats():
    """Per-query summary (split, PnP inliers, matches) from HLoc's log, for the UI."""
    with open(f"{paths.RESULTS}_logs.pkl", "rb") as f:
        logs = pickle.load(f)
    split = dict(paths.query_list())
    out = {}
    for q, log in logs["loc"].items():
        ret = log.get("PnP_ret")
        ok = ret is not None
        out[q] = {"split": split.get(q), "localized": bool(ok),
                  "inliers": int(ret["num_inliers"]) if ok else 0, "matches": int(log.get("num_matches", 0))}
    (paths.OUT / "loc_stats.json").write_text(json.dumps(out))
    print(f"stats for {len(out)} queries")


if __name__ == "__main__":
    {"reference": reference, "triangulation": triangulation, "localization": localization, "stats": stats}[sys.argv[1]]()
