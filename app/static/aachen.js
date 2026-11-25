"use strict";
/* Visual localization on Aachen Day-Night (paper Sec. 4.3, Table 4) — XFeat sparse only.
   Relies on globals from app.js: $, $$, fmt, int, S, table, showErr, loadImg, paintMatches. */

const LS = { res: null, imgs: [null, null], map: null, name: null, split: "", mapView: "zoom", ready: false };

/* ------------------------------------------------------------------ gallery */
async function lLoadSamples(seed) {
  const q = new URLSearchParams({ n: 24, split: LS.split });
  if (!LS.split) q.delete("split");
  if (seed) q.set("seed", seed);
  const list = await (await fetch("/api/aachen/queries?" + q)).json();
  const row = $("#loc-samples"); row.innerHTML = "";
  list.forEach(p => {
    const b = document.createElement("button"); b.className = "sample q"; b.dataset.name = p.name;
    b.innerHTML = `<div style="grid-template-columns:1fr"><img loading="lazy" src="/api/aachen/img/${p.name}?thumb=true"></div><span>${p.split} · ${p.name.split("/").pop().replace(/\.jpg$/, "")}</span>`;
    if (p.name === LS.name) b.classList.add("on");
    b.onclick = () => {
      $$("#loc-samples .sample").forEach(s => s.classList.remove("on")); b.classList.add("on");
      LS.name = p.name; lUpdateRun();
    };
    row.appendChild(b);
  });
}
function lUpdateRun() { $("#loc-run").disabled = !(LS.name && LS.ready); }
$("#loc-shuffle").onclick = () => lLoadSamples(Math.floor(Math.random() * 1e6));
$$("#loc-split button").forEach(b => b.onclick = () => {
  $$("#loc-split button").forEach(x => x.classList.toggle("on", x === b)); LS.split = b.dataset.s; lLoadSamples();
});

/* ------------------------------------------------------------------ localize */
$("#loc-run").onclick = async () => {
  const btn = $("#loc-run"); showErr($("#loc-err"), "");
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Localizing…';
  try {
    const r = await fetch("/api/aachen/localize", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: LS.name }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const res = await r.json();
    if (!LS.map) LS.map = await (await fetch("/api/aachen/map")).json();
    LS.res = res;
    if (res.ok) LS.imgs = await Promise.all([loadImg(`/api/aachen/img/${res.name}`), loadImg(`/api/aachen/img/${res.best.name}`)]);
    lRender();
    $("#l-results").hidden = false; $("#nav-results").hidden = false; $("#nav-results").href = "#l-results";
    requestAnimationFrame(() => { lDraw(); $("#l-results").scrollIntoView({ behavior: "smooth" }); });
  } catch (e) { showErr($("#loc-err"), "Localization failed: " + e.message); }
  btn.textContent = "Localize"; lUpdateRun();
};

/* ------------------------------------------------------------------ drawing */
function lDraw() {
  const R = LS.res; if (!R) return;
  if (R.ok) {
    const b = R.best, txt = `inliers: ${R.num_inliers} | 2D-3D: ${R.num_corr} | ratio: ${fmt(R.inlier_ratio, 2)}`;
    paintMatches($("#cv-l-match"), { pts: b.pts, inl: b.inl }, R.image_size, b.image_size, LS.imgs, txt, "XFeat");
    $("#cv-l-match").parentElement.hidden = false;
    drawMap();
  } else {
    $("#cv-l-match").parentElement.hidden = true;
  }
}
function drawMap() {
  const R = LS.res, M = LS.map, cv = $("#cv-l-map");
  const W = cv.clientWidth, H = Math.round(W * 0.62), dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = "#0b0b0d"; c.fillRect(0, 0, W, H);
  const q = R.pose.center;
  let x0, x1, y0, y1;
  if (LS.mapView === "zoom") {
    const near = R.per_db.map(d => d.center);
    const r = Math.max(25, ...near.map(p => Math.hypot(p[0] - q[0], p[1] - q[1]))) * 1.5;
    const asp = W / H; x0 = q[0] - r * asp; x1 = q[0] + r * asp; y0 = q[1] - r; y1 = q[1] + r;
  } else {
    const all = M.centers.concat(M.points);
    x0 = Math.min(...all.map(p => p[0])); x1 = Math.max(...all.map(p => p[0]));
    y0 = Math.min(...all.map(p => p[1])); y1 = Math.max(...all.map(p => p[1]));
    const padx = (x1 - x0) * .05, pady = (y1 - y0) * .05; x0 -= padx; x1 += padx; y0 -= pady; y1 += pady;
  }
  const s = Math.min(W / (x1 - x0), H / (y1 - y0)), ox = (W - s * (x1 - x0)) / 2, oy = (H - s * (y1 - y0)) / 2;
  const X = p => ox + (p[0] - x0) * s, Y = p => H - oy - (p[1] - y0) * s;
  const zoom = LS.mapView === "zoom";
  c.fillStyle = "rgba(255,255,255,.16)";
  M.points.forEach(p => { const x = X(p), y = Y(p); if (x > -2 && x < W + 2 && y > -2 && y < H + 2) c.fillRect(x, y, zoom ? 1.6 : 1, zoom ? 1.6 : 1); });
  c.fillStyle = "rgba(41,151,255,.55)";
  M.centers.forEach(p => { const x = X(p), y = Y(p); if (x > -4 && x < W + 4 && y > -4 && y < H + 4) { c.beginPath(); c.arc(x, y, zoom ? 2.6 : 1.5, 0, 7); c.fill(); } });
  // retrieved images that supported the pose
  R.per_db.forEach((d, i) => {
    c.strokeStyle = i === 0 ? "#30d158" : "rgba(255,255,255,.8)"; c.lineWidth = 1.5; c.beginPath();
    c.arc(X(d.center), Y(d.center), zoom ? 6 : 4, 0, 7); c.stroke();
  });
  // query: position + viewing direction
  const f = R.pose.forward, qx = X(q), qy = Y(q), len = zoom ? 34 : 18;
  c.strokeStyle = "#ff9f0a"; c.fillStyle = "#ff9f0a"; c.lineWidth = 3; c.lineCap = "round";
  c.beginPath(); c.moveTo(qx, qy); c.lineTo(qx + f[0] * len, qy - f[1] * len); c.stroke();
  c.beginPath(); c.arc(qx, qy, zoom ? 7 : 5, 0, 7); c.fill();
  c.strokeStyle = "#fff"; c.lineWidth = 2; c.stroke();
  $("#l-legend").innerHTML =
    `<span><i style="background:#ff9f0a"></i>Estimated query camera</span>` +
    `<span><i style="background:#30d158"></i>Best retrieved image</span>` +
    `<span><i style="background:#2997ff"></i>Database cameras</span>` +
    `<span><i style="background:rgba(255,255,255,.4)"></i>3D points (top-down)</span>`;
}
$$("#l-seg-show button").forEach(b => b.onclick = () => {
  $$("#l-seg-show button").forEach(x => x.classList.toggle("on", x === b)); S.show = b.dataset.v; lDraw();
});
$$("#l-seg-map button").forEach(b => b.onclick = () => {
  $$("#l-seg-map button").forEach(x => x.classList.toggle("on", x === b)); LS.mapView = b.dataset.v; if (LS.res && LS.res.ok) drawMap();
});
$("#l-density").oninput = e => { S.density = +e.target.value; lDraw(); };
$("#l-save-m").onclick = () => $("#cv-l-match").toBlob(bl => {
  const a = document.createElement("a"); a.href = URL.createObjectURL(bl); a.download = "xfeat_aachen_matches.png"; a.click();
});

/* -------------------------------------------------------------------- tables */
function lRender() {
  const R = LS.res, one = (label, v) => ({ label, dir: 0, raw: [0, 0], cells: [v] });
  if (!R.ok) {
    $("#l-sub").textContent = "PnP + RANSAC could not find a pose for this query" + (R.error ? ` (${R.error})` : ".");
    table($("#l-table"), ["", "XFeat"], [one("Matches with the retrieved images", int(R.num_matches)),
      one("Time (ms)", fmt((R.timings.match_ms || 0) + (R.timings.pnp_ms || 0), 0))]);
    $("#l-db-table").innerHTML = ""; return;
  }
  $("#l-sub").textContent = `${R.split === "night" ? "Night" : "Day"} query · matched against ${R.retrieved} retrieved database images, then localized in the 3D map.`;
  $("#l-match-title").textContent = `Query vs. best retrieved image (${R.best.name.split("/").pop()})`;
  table($("#l-table"), ["", "XFeat"], [
    { group: "Matching" },
    one("Retrieved images (NetVLAD top-50)", int(R.retrieved)),
    one("Mutual matches with all of them", int(R.num_matches)),
    one("2D–3D correspondences", int(R.num_corr)),
    { group: "Pose (PnP + RANSAC, 12 px)" },
    one("Inliers", int(R.num_inliers)),
    one("Inlier ratio", fmt(R.inlier_ratio, 2)),
    { group: "Time" },
    one("Matching (ms)", fmt(R.timings.match_ms, 0)),
    one("PnP + RANSAC (ms)", fmt(R.timings.pnp_ms, 0)),
    one("Total (ms)", fmt(R.timings.total_ms, 0)),
  ]);
  let h = `<thead><tr><th>Database image</th><th>Matches</th><th>With 3D point</th><th>Inliers</th></tr></thead><tbody>`;
  R.per_db.forEach((d, i) => { h += `<tr${i === 0 ? ' class="hl"' : ""}><td>${d.name}</td><td>${int(d.matches)}</td><td>${int(d.with3d)}</td><td>${int(d.inliers)}</td></tr>`; });
  $("#l-db-table").innerHTML = h + "</tbody>";
}
window.addEventListener("resize", () => { if (S.task === "localization" && LS.res) { clearTimeout(lDraw.t); lDraw.t = setTimeout(lDraw, 120); } });

/* ----------------------------------------------------------------- benchmark */
let aPaper = null, aTimer = null;
const ATH = ["0.25 m, 2°", "0.5 m, 5°", "5 m, 10°"];

fetch("/static/data/paper_aachen.json").then(r => r.json()).then(p => {
  aPaper = p;
  const head = `<thead><tr><th rowspan="2">Method</th><th colspan="3" style="text-align:center">Day</th><th colspan="3" style="text-align:center">Night</th></tr>` +
    `<tr>${[...ATH, ...ATH].map(t => `<th>${t}</th>`).join("")}</tr></thead><tbody>`;
  let h = head, cls = "";
  p.rows.forEach(r => {
    if (r.cls !== cls) { cls = r.cls; h += `<tr class="grp"><td colspan="7">${cls}</td></tr>`; }
    h += `<tr${r.key ? ' class="hl"' : ""}><td>${r.m}</td>${[...r.day, ...r.night].map(v => `<td>${v.toFixed(1)}</td>`).join("")}</tr>`;
  });
  $("#la-paper-table").innerHTML = h + "</tbody>";
  aRenderCmp(null);
});

function aRenderCmp(sr) {
  if (!aPaper) return;
  const row = aPaper.rows.find(r => r.key), pv = [...row.day, ...row.night];
  let h = `<thead><tr><th rowspan="2">Method</th><th colspan="3" style="text-align:center">Day</th><th colspan="3" style="text-align:center">Night</th></tr>` +
    `<tr>${[...ATH, ...ATH].map(t => `<th>${t}</th>`).join("")}</tr></thead><tbody>`;
  h += `<tr class="hl"><td>XFeat — paper</td>${pv.map(v => `<td>${fmt(v, 1)}</td>`).join("")}</tr>`;
  if (sr) {
    const mine = [...sr.day, ...sr.night];
    h += `<tr class="hl"><td>XFeat — this run (benchmark server)</td>${mine.map(v => `<td class="best">${fmt(v, 1)}</td>`).join("")}</tr>`;
    h += `<tr class="sub"><td>Δ</td>${mine.map((v, i) => { const d = v - pv[i]; return `<td class="${d >= 0 ? "pos" : "neg"}">${d >= 0 ? "+" : ""}${fmt(d, 1)}</td>`; }).join("")}</tr>`;
  }
  $("#la-cmp-table").innerHTML = h + "</tbody>";
  $("#la-cmp-note").textContent = sr
    ? "Percentages of queries localized within (0.25 m, 2°) / (0.5 m, 5°) / (5 m, 10°), as reported by visuallocalization.net for your submission."
    : "Enter the benchmark server's results above to fill in this table.";
}

const STAGE_NOTE = {
  reference: "4,328 database poses and intrinsics from the NVM model",
  features: "XFeat, 1,024 px, 4,096 keypoints — 4,328 database + 922 query images",
  matches_db: "mutual nearest neighbours on 85,514 covisible database pairs",
  triangulation: "3D points triangulated from the known database poses",
  matches_query: "mutual nearest neighbours on 46,099 retrieved query pairs",
  localization: "PnP + RANSAC for all 922 queries",
};
const TITLES = { reference: "Reference poses", features: "XFeat features", matches_db: "Database matches", triangulation: "3D map", matches_query: "Query matches", localization: "Localization" };

async function aPoll() {
  let s;
  try { s = await (await fetch("/api/aachen/status")).json(); } catch { return; }
  const running = s.status === "running";
  LS.ready = !!s.ready; lUpdateRun();
  $("#loc-note").textContent = s.ready ? "" : "The 3D map is not built yet — open “The numbers” → Localization and run the pipeline once.";
  $("#la-run").hidden = running; $("#la-cancel").hidden = !running || !!s.external;
  const done = Object.values(s.stages).every(x => x.state === "done");
  $("#la-run").textContent = done ? "Run again" : (Object.values(s.stages).some(x => x.state === "done") ? "Resume" : "Build map and localize");
  showErr($("#la-err"), s.status === "error" ? s.error : "");
  $("#la-stages").innerHTML = Object.entries(s.stages).map(([k, v]) => {
    const pct = v.state === "done" ? 100 : v.total > 1 ? 100 * v.done / v.total : 0;
    const sub = v.state === "done" ? "done" : v.state === "running" ? (v.total > 1 ? `${int(v.done)} / ${int(v.total)}` + (v.eta_s ? ` · ~${v.eta_s > 90 ? Math.ceil(v.eta_s / 60) + " min" : Math.ceil(v.eta_s) + " s"}` : "") : "working…") : v.state === "error" ? "failed" : "waiting";
    return `<li class="${v.state}"><span class="dot"></span><span class="nm">${TITLES[k]}<br><span class="sub" style="text-align:left">${STAGE_NOTE[k]}</span></span><span class="sub">${sub}</span><span class="pb"><i style="width:${pct}%"></i></span></li>`;
  }).join("");
  $("#la-info").textContent = s.external ? "Running in another process (started from the command line)." : s.status === "cancelled" ? "Stopped — progress is saved." : "";
  // local summary
  $("#la-summary-card").hidden = !s.summary;
  if (s.summary) {
    let h = `<thead><tr><th></th><th>Queries</th><th>Localized</th><th>Median inliers</th><th>Mean inliers</th></tr></thead><tbody>`;
    ["day", "night"].forEach(k => { const x = s.summary[k]; h += `<tr><td>${k[0].toUpperCase() + k.slice(1)}</td><td>${x.queries}</td><td>${x.localized} (${fmt(100 * x.localized / Math.max(1, x.queries), 1)}%)</td><td>${int(x.median_inliers)}</td><td>${int(x.mean_inliers)}</td></tr>`; });
    $("#la-summary").innerHTML = h + "</tbody>";
  }
  $("#la-submit-card").hidden = !s.has_submission;
  const sr = s.server_results;
  if (sr) {
    const f = $("#la-form"); ["d", "n"].forEach((p, j) => [0, 1, 2].forEach(i => { f.elements[p + i].value = sr[j ? "night" : "day"][i]; }));
  }
  $("#la-clear").hidden = !sr;
  aRenderCmp(sr);
  if (running && !aTimer) aTimer = setInterval(aPoll, 2000);
  if (!running && aTimer) { clearInterval(aTimer); aTimer = null; }
}
$("#la-run").onclick = async () => {
  const r = await fetch("/api/aachen/start", { method: "POST" });
  if (!r.ok) showErr($("#la-err"), (await r.json().catch(() => ({}))).detail || "Could not start");
  else if ((await r.json()).busy) showErr($("#la-err"), "The pipeline is already running in another process (started from the command line). Its progress shows here.");
  setTimeout(aPoll, 500);
};
$("#la-cancel").onclick = async () => { await fetch("/api/aachen/cancel", { method: "POST" }); };
$("#la-form").onsubmit = async e => {
  e.preventDefault(); const f = e.target.elements;
  const body = { day: [f.d0.value, f.d1.value, f.d2.value], night: [f.n0.value, f.n1.value, f.n2.value] };
  if ([...body.day, ...body.night].some(v => v === "")) return showErr($("#la-err"), "Fill in all six percentages.");
  const r = await fetch("/api/aachen/server_results", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  showErr($("#la-err"), r.ok ? "" : "Could not save the results."); aPoll();
};
$("#la-clear").onclick = async () => { await fetch("/api/aachen/server_results", { method: "DELETE" }); $("#la-form").reset(); aPoll(); };
aPoll();
