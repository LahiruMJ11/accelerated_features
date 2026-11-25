"""Terminal front-end for the XFeat benchmarks (same code as the UI), with the paper's numbers next to ours.

Run from the repo root in the `myenv` environment:
    python -m app.cli megadepth | scannet | hpatches | aachen | all   [--rerun] [--pairs N]
"""
import argparse
import json
import sys
import time

ROOT_NOTE = "XFeat: Accelerated Features for Lightweight Image Matching (CVPR 2024)"


# ---------------------------------------------------------------- formatting
def table(title, head, rows, note=None):
    """rows: list of lists of str, or None for a separator line."""
    cols = len(head)
    w = [max(len(str(head[i])), *(len(str(r[i])) for r in rows if r)) for i in range(cols)]
    line = lambda l, m, r: l + m.join("─" * (x + 2) for x in w) + r
    fmt = lambda cells: "│" + "│".join(f" {str(c):<{w[i]}} " if i == 0 else f" {str(c):>{w[i]}} " for i, c in enumerate(cells)) + "│"
    print()
    print(title)
    print(line("┌", "┬", "┐"))
    print(fmt(head))
    print(line("├", "┼", "┤"))
    for r in rows:
        print(line("├", "┼", "┤") if r is None else fmt(r))
    print(line("└", "┴", "┘"))
    if note:
        print(note)


def f1(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def delta(a, b, d=1):
    x = round(a - b, d)
    return f"{'+' if x > 0 else ''}{0.0 if x == 0 else x:.{d}f}"


def paper(name):
    from pathlib import Path
    return json.loads((Path(__file__).parent / "static" / "data" / name).read_text())


def banner(text):
    print("\n" + "═" * 78)
    print(text)
    print("═" * 78)


# --------------------------------------------------------------------- jobs
def run_job(mod, label, **kw):
    """Run a benchmark module's background job in the foreground, with a progress line."""
    if not mod.start(**kw):
        sys.exit("A job is already running.")
    t0 = time.time()
    while mod._thread.is_alive():
        s = mod._state
        eta = f"  ~{int(s['eta_s'] // 60)} min {int(s['eta_s'] % 60):02d} s left" if s.get("eta_s") else ""
        phase = f"[{s['phase']}] " if s.get("phase") else ""
        print(f"\r{label}: {phase}{s['done']:>5} / {s['total']} pairs{eta}      ", end="", flush=True)
        time.sleep(1)
    print(f"\r{label}: finished in {time.time() - t0:.0f} s{' ' * 40}")
    if mod._state["status"] != "done":
        sys.exit(f"Failed: {mod._state.get('error') or mod._state['status']}")
    return mod._state["last"]


# ----------------------------------------------------------------- megadepth
def cmd_megadepth(a):
    from app import benchmark
    res = None if a.rerun or a.pairs else (json.loads(benchmark.RESULT.read_text()) if benchmark.RESULT.exists() else None)
    if res is None:
        res = run_job(benchmark, "MegaDepth-1500", limit=a.pairs)
    banner("Relative pose estimation · MegaDepth-1500 · XFeat (sparse) and XFeat* (semi-dense)")
    pr = {r["key"]: r for r in paper("paper.json")["rows"] if r.get("key")}
    rows = []
    for key, name in (("xfeat", "XFeat"), ("xfeat_star", "XFeat*")):
        m, p = res["methods"][key], pr[key]
        rows.append([f"{name}  paper", f1(p["auc5"]), f1(p["auc10"]), f1(p["auc20"]), f1(p["acc10"]), f1(p["mir"], 2), f"{p['inliers']:,}", f1(p["fpsv"])])
        rows.append([f"{name}  this run", f1(m["auc5"]), f1(m["auc10"]), f1(m["auc20"]), f1(m["acc10"]), f1(m["mir"], 2), f"{m['inliers']:,.0f}", f1(m["fps"])])
        rows.append(["   Δ", delta(m["auc5"], p["auc5"]), delta(m["auc10"], p["auc10"]), delta(m["auc20"], p["auc20"]),
                     delta(m["acc10"], p["acc10"]), delta(m["mir"], p["mir"], 2), delta(m["inliers"], p["inliers"], 0), delta(m["fps"], p["fpsv"])])
        rows.append(None)
    table(f"{res['pairs']} pairs · RANSAC {res['ransac_thr']} px · {res['device'].upper()} · {res['date'].replace('T', ' ')}",
          ["Method", "AUC@5°", "AUC@10°", "AUC@20°", "Acc@10°", "MIR", "#inliers", "FPS"], rows[:-1],
          "Paper: Table 1. FPS is measured at VGA on this machine (the paper used an i5-1135G7 laptop CPU).")


# ------------------------------------------------------------------- scannet
def cmd_scannet(a):
    from app import benchmark_s
    res = None if a.rerun or a.pairs else (json.loads(benchmark_s.RESULT.read_text()) if benchmark_s.RESULT.exists() else None)
    if res is None:
        res = run_job(benchmark_s, "ScanNet-1500", limit=a.pairs)
    banner("Relative pose estimation · ScanNet-1500 (indoor) · XFeat and XFeat*")
    pr = {r["key"]: r for r in paper("paper_scannet.json")["rows"] if r.get("key")}
    rows = []
    for key, name in (("xfeat", "XFeat"), ("xfeat_star", "XFeat*")):
        m, p = res["methods"][key], pr[key]["v"]
        rows.append([f"{name}  paper", f1(p[0]), f1(p[1]), f1(p[2]), "—"])
        rows.append([f"{name}  this run", f1(m["auc5"]), f1(m["auc10"]), f1(m["auc20"]), f"{m['best_thr']} px"])
        rows.append(["   Δ", delta(m["auc5"], p[0]), delta(m["auc10"], p[1]), delta(m["auc20"], p[2]), ""])
        rows.append(None)
    table(f"{res['pairs']} pairs · {res['resolution']} · best of 12 RANSAC thresholds · {res['device'].upper()} · {res['date'].replace('T', ' ')}",
          ["Method", "AUC@5°", "AUC@10°", "AUC@20°", "RANSAC thr."], rows[:-1], "Paper: Table 2 (AUC only).")


# ------------------------------------------------------------------- hpatches
def cmd_hpatches(a):
    from app import benchmark_h
    res = None if a.rerun or a.pairs else (json.loads(benchmark_h.RESULT.read_text()) if benchmark_h.RESULT.exists() else None)
    if res is None:
        res = run_job(benchmark_h, "HPatches", limit=a.pairs)
    banner("Homography estimation · HPatches · XFeat (sparse)")
    p = next(r for r in paper("paper_hpatches.json")["rows"] if r.get("key"))
    g = res["groups"]
    mine = [g["illumination"][f"mha{t}"] for t in (3, 5, 7)] + [g["viewpoint"][f"mha{t}"] for t in (3, 5, 7)]
    pv = p["i"] + p["v"]
    rows = [["XFeat  paper"] + [f1(x) for x in pv],
            ["XFeat  this run"] + [f1(x) for x in mine],
            ["   Δ"] + [delta(m, x) if m is not None else "—" for m, x in zip(mine, pv)]]
    c = res["cfg"]
    table(f"{res['pairs']} pairs ({g['illumination']['n']} illumination + {g['viewpoint']['n']} viewpoint) · MAGSAC++ {c['thr']} px · "
          f"{c['short_side']} px short side · {c['top_k']} keypoints · {res['device'].upper()} · {res['date'].replace('T', ' ')}",
          ["MHA (%)", "Illum @3px", "@5px", "@7px", "View @3px", "@5px", "@7px"], rows, "Paper: Table 3 (XFeat only).")


# --------------------------------------------------------------------- aachen
def cmd_aachen(a):
    from app.loc import paths, pipeline
    if a.set_results:
        v = [float(x) for x in a.set_results]
        paths.OUT.mkdir(parents=True, exist_ok=True)
        paths.SERVER_RESULTS.write_text(json.dumps({"day": v[:3], "night": v[3:]}))
        print("Saved the benchmark-server results.")
    pipeline.refresh()
    done = all(s["state"] == "done" for s in pipeline._state["stages"].values())
    if a.run or not done:
        if not a.run:
            sys.exit("The Aachen pipeline has not been run yet. Run it once with:  python -m app.cli aachen --run   (about an hour on CPU)")
        t0 = time.time()
        if not pipeline.start():
            sys.exit("A pipeline run is already in progress (another process).")
        while pipeline._thread.is_alive():
            st = pipeline._state
            cur = st["stage"]
            s = st["stages"].get(cur, {}) if cur else {}
            eta = f"  ~{int(s['eta_s'] // 60)} min left" if s.get("eta_s") else ""
            print(f"\r{pipeline.TITLES.get(cur, '…'):<18} {s.get('done', 0):>6} / {s.get('total', 0)}{eta}      ", end="", flush=True)
            time.sleep(2)
        print(f"\rPipeline finished in {(time.time() - t0) / 60:.0f} min{' ' * 40}")
        if pipeline._state["status"] != "done":
            sys.exit(f"Failed: {pipeline._state.get('error')}")

    banner("Visual localization · Aachen Day-Night v1.0 · XFeat (sparse) + HLoc")
    stats = json.loads((paths.OUT / "loc_stats.json").read_text())
    rows = []
    for split in ("day", "night"):
        v = [x for x in stats.values() if x["split"] == split]
        inl = sorted(x["inliers"] for x in v if x["localized"])
        rows.append([split.capitalize(), len(v), f"{sum(x['localized'] for x in v)} ({100 * sum(x['localized'] for x in v) / len(v):.1f}%)",
                     f"{inl[len(inl) // 2]:,}", f"{sum(inl) / len(inl):,.0f}"])
    table("Local summary — 4,328 database images, 3D map from known poses, NetVLAD top-50 retrieval, PnP + RANSAC",
          ["Queries", "Count", "Localized", "Median inliers", "Mean inliers"], rows)
    print(f"\nPoses file for visuallocalization.net:\n  {paths.RESULTS}")
    if paths.SERVER_RESULTS.exists():
        sr = json.loads(paths.SERVER_RESULTS.read_text())
        p = next(r for r in paper("paper_aachen.json")["rows"] if r.get("key"))
        pv, mine = p["day"] + p["night"], sr["day"] + sr["night"]
        table("Official benchmark-server results", ["", "Day 0.25m,2°", "0.5m,5°", "5m,10°", "Night 0.25m,2°", "0.5m,5°", "5m,10°"],
              [["XFeat  paper"] + [f1(x) for x in pv], ["XFeat  this run"] + [f1(x) for x in mine],
               ["   Δ"] + [delta(m, x) for m, x in zip(mine, pv)]], "Paper: Table 4 (XFeat only).")
    else:
        print("\nThe percentages (Table 4) come from visuallocalization.net. After submitting, add them with:\n"
              "  python -m app.cli aachen --set-results D1 D2 D3 N1 N2 N3")

    if a.query:
        from app.loc import live
        name = a.query
        if name in ("day", "night"):
            name = live.query_names(name)[0]
        r = live.localize(name)
        banner(f"Localizing one query live · {name}")
        if not r["ok"]:
            print("PnP + RANSAC found no pose:", r.get("error"))
            return
        table("", ["Metric", "Value"], [
            ["Split", r["split"]], ["Retrieved database images", r["retrieved"]], ["Mutual matches (all retrieved)", f"{r['num_matches']:,}"],
            ["2D–3D correspondences", f"{r['num_corr']:,}"], ["PnP inliers", f"{r['num_inliers']:,}"], ["Inlier ratio", f"{r['inlier_ratio']:.2f}"],
            ["Matching (ms)", f"{r['timings']['match_ms']:.0f}"], ["PnP + RANSAC (ms)", f"{r['timings']['pnp_ms']:.0f}"],
            ["Total (ms)", f"{r['timings']['total_ms']:.0f}"], ["Best database image", r["best"]["name"]]])


def main():
    ap = argparse.ArgumentParser(prog="python -m app.cli", description=ROOT_NOTE)
    ap.add_argument("task", choices=["megadepth", "scannet", "hpatches", "aachen", "all"])
    ap.add_argument("--rerun", action="store_true", help="run the benchmark again instead of printing the saved result")
    ap.add_argument("--pairs", type=int, help="quick demo on the first N pairs (not saved)")
    ap.add_argument("--run", action="store_true", help="aachen: build the map and localize all queries (about an hour)")
    ap.add_argument("--query", help="aachen: localize one query live (a query path, or 'day' / 'night')")
    ap.add_argument("--set-results", nargs=6, metavar=("D1", "D2", "D3", "N1", "N2", "N3"),
                    help="aachen: save the percentages from visuallocalization.net")
    a = ap.parse_args()
    for task, fn in (("megadepth", cmd_megadepth), ("scannet", cmd_scannet), ("hpatches", cmd_hpatches), ("aachen", cmd_aachen)):
        if a.task in (task, "all"):
            fn(a)
    print()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
