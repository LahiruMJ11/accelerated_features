"""Paths and names for the Aachen Day-Night (v1.0) visual localization pipeline (HLoc-style)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HLOC_DIR = ROOT / "third_party" / "Hierarchical-Localization"      # vendored HLoc (pairs + triangulation/localization)
if str(HLOC_DIR) not in sys.path:
    sys.path.insert(0, str(HLOC_DIR))

DATA = ROOT / "datasets" / "Aachen"
IMAGES = DATA / "images_upright"
NVM = DATA / "3D-models" / "aachen_cvpr2018_db.nvm"
DB_INTRINSICS = DATA / "3D-models" / "database_intrinsics.txt"
QUERY_FILES = {"day": DATA / "queries" / "day_time_queries_with_intrinsics.txt",
               "night": DATA / "queries" / "night_time_queries_with_intrinsics.txt"}

PAIRS_DB = HLOC_DIR / "pairs" / "aachen" / "pairs-db-covis20.txt"          # 20 most covisible per db image
PAIRS_QUERY = HLOC_DIR / "pairs" / "aachen" / "pairs-query-netvlad50.txt"   # NetVLAD top-50 per query

OUT = ROOT / "app" / "cache" / "aachen"
REF_MODEL = OUT / "sfm_reference"          # db poses + intrinsics (no points)
SFM = OUT / "sfm_xfeat"                    # triangulated with XFeat matches
FEATURES = OUT / "feats-xfeat.h5"
MATCHES_DB = OUT / "matches-xfeat-db.h5"
MATCHES_QUERY = OUT / "matches-xfeat-query.h5"
RESULTS = OUT / "Aachen_hloc_xfeat_netvlad50.txt"
STATUS = OUT / "status.json"
SERVER_RESULTS = OUT / "server_results.json"

MAX_DIM = 1024       # paper: images resized so the max dimension is 1024
TOP_K = 4096         # paper: top 4096 keypoints


def db_names():
    return [l.split()[0] for l in DB_INTRINSICS.read_text().splitlines() if l.strip()]


def query_list():
    """[(name, split)] for the 824 day + 98 night queries."""
    out = []
    for split, f in QUERY_FILES.items():
        out += [(l.split()[0], split) for l in f.read_text().splitlines() if l.strip()]
    return out
