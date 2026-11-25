"""Staged, resumable HLoc-style visual localization pipeline on Aachen Day-Night v1.0 with XFeat features.

stages: reference -> features -> matches_db -> triangulation -> matches_query -> localization
Every stage skips work that is already on disk, so a stopped run resumes where it left off.
CLI:  python -m app.loc.pipeline            (run from the repo root, in the `myenv` environment)
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

from . import features, matching, paths

STAGES = ["reference", "features", "matches_db", "triangulation", "matches_query", "localization"]
TITLES = {"reference": "Reference poses", "features": "XFeat features", "matches_db": "Database matches",
          "triangulation": "3D map", "matches_query": "Query matches", "localization": "Localization"}

_state = {"status": "idle", "stage": None, "error": None, "eta_s": None,
          "stages": {s: {"state": "pending", "done": 0, "total": 0} for s in STAGES}}
_cancel = threading.Event()
_thread = None
_proc = None


LOCK = paths.OUT / "run.lock"
PROGRESS = paths.OUT / "progress.json"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def other_running():
    """PID of another process (e.g. the CLI) that is running the pipeline, else None."""
    try:
        pid = int(LOCK.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid != os.getpid() and _pid_alive(pid) else None


def _progress_writer(stop):
    while not stop.wait(2):
        try:
            PROGRESS.write_text(json.dumps({**_state, "pid": os.getpid(), "t": time.time()}, default=str))
        except Exception:
            pass


def _marker(stage):
    return {
        "reference": paths.REF_MODEL / "images.bin",
        "triangulation": paths.SFM / "points3D.bin",
        "localization": paths.RESULTS,
    }.get(stage)


def _is_done(stage):
    m = _marker(stage)
    if m is not None:
        return m.exists()
    return _state["stages"][stage].get("complete", False)


def refresh():
    """Re-derive finished stages from what is on disk (so the UI is right after a server restart)."""
    for st in STAGES:
        info = _state["stages"][st]
        if _marker(st) is not None and _marker(st).exists():
            info["state"] = "done"
    flag = paths.OUT / "complete.json"
    if flag.exists():
        for k in json.loads(flag.read_text()):
            _state["stages"][k].update(state="done", complete=True)


def _mark_complete(stage):
    flag = paths.OUT / "complete.json"
    cur = json.loads(flag.read_text()) if flag.exists() else []
    if stage not in cur:
        cur.append(stage)
    flag.write_text(json.dumps(cur))
    _state["stages"][stage].update(state="done", complete=True)


# --------------------------------------------------------------------- stages
def _worker(stage, info):
    """Run a pycolmap stage in its own process (never import pycolmap here: it clashes with torch's OpenMP)."""
    global _proc
    info.update(done=0, total=1)
    log = open(paths.OUT / f"{stage}.log", "w")
    _proc = subprocess.Popen([sys.executable, "-m", "app.loc.colmap_worker", stage], cwd=paths.ROOT,
                             stdout=log, stderr=subprocess.STDOUT)
    while _proc.poll() is None:
        if _cancel.is_set():
            _proc.terminate()
            return False
        time.sleep(1)
    log.close()
    if _proc.returncode != 0:
        tail = (paths.OUT / f"{stage}.log").read_text().strip().splitlines()[-3:]
        raise RuntimeError(f"{stage} failed: " + " | ".join(tail))
    info.update(done=1, total=1)
    return True


def stage_reference(info):
    return _worker("reference", info)


def stage_features(info):
    names = paths.db_names() + [n for n, _ in paths.query_list()]
    return features.run(names, info, _cancel)


def stage_matches_db(info):
    return matching.run(paths.PAIRS_DB, paths.MATCHES_DB, info, _cancel)


def stage_triangulation(info):
    return _worker("triangulation", info)


def stage_matches_query(info):
    return matching.run(paths.PAIRS_QUERY, paths.MATCHES_QUERY, info, _cancel)


def stage_localization(info):
    return _worker("localization", info)


FUNCS = {"reference": stage_reference, "features": stage_features, "matches_db": stage_matches_db,
         "triangulation": stage_triangulation, "matches_query": stage_matches_query,
         "localization": stage_localization}


def _run():
    stop = threading.Event()
    try:
        paths.OUT.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(str(os.getpid()))
        threading.Thread(target=_progress_writer, args=(stop,), daemon=True).start()
        _state.update(status="running", error=None)
        for st in STAGES:
            info = _state["stages"][st]
            if _is_done(st):
                info["state"] = "done"
                continue
            if _cancel.is_set():
                _state["status"] = "cancelled"
                return
            _state["stage"] = st
            info["state"] = "running"
            ok = FUNCS[st](info)
            if not ok:
                info["state"] = "pending"
                _state["status"] = "cancelled"
                return
            if _marker(st) is None:
                _mark_complete(st)
            info["state"] = "done"
        _state.update(status="done", stage=None, eta_s=None)
        paths.STATUS.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds")}))
    except Exception as ex:
        import traceback
        traceback.print_exc()
        _state.update(status="error", error=f"{type(ex).__name__}: {ex}")
        _state["stages"][_state["stage"]]["state"] = "error"
    finally:
        stop.set()
        try:
            PROGRESS.write_text(json.dumps({**_state, "pid": os.getpid(), "t": time.time()}, default=str))
            if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
                LOCK.unlink()
        except Exception:
            pass


def start():
    global _thread
    if _thread and _thread.is_alive():
        return False
    if other_running():
        return False
    refresh()
    _cancel.clear()
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    return True


def cancel():
    _cancel.set()


def status():
    refresh()
    pid = other_running()
    if pid:                                   # the pipeline runs in another process (e.g. started from the CLI)
        ext = {}
        try:
            ext = json.loads(PROGRESS.read_text())
            if time.time() - ext.get("t", 0) > 30 or ext.get("pid") != pid:
                ext = {}
        except Exception:
            pass
        out = dict(_state)
        out.update(status="running", error=None, external=True, pid=pid)
        if ext.get("stages"):
            out["stages"] = ext["stages"]
            out["stage"] = ext.get("stage")
        else:                                 # no progress info (older run): trust what is on disk
            out["stages"] = {k: dict(v) for k, v in _state["stages"].items()}
            first = next((k for k in STAGES if out["stages"][k]["state"] != "done"), None)
            if first:
                out["stages"][first]["state"] = "running"
            out["stage"] = first
        cur = out.get("stage")
        out["eta_s"] = out["stages"][cur].get("eta_s") if cur else None
        return out
    out = dict(_state)
    cur = _state.get("stage")
    out["eta_s"] = _state["stages"][cur].get("eta_s") if cur else None
    return out


if __name__ == "__main__":
    start()
    _thread.join()
    print(json.dumps(status(), default=str, indent=1))
