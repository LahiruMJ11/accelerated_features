"use strict";
/* Homography estimation on HPatches (paper Sec. 4.2, Table 3) — XFeat sparse only.
   Relies on globals from app.js: $, $$, fmt, int, S, table, setSlot, loadSamples, updateRun, showErr, loadImg, paintMatches. */

const HS = { res: null, imgs: [null, null], warp: null };

/* ---------------------------------------------------------- task switching */
function setTask(t) {
  if (S.task === t) return;
  S.task = t;
  $$("#task-seg button, #bench-task-seg button").forEach(b => b.classList.toggle("on", b.dataset.t === t));
  const hom = t === "homography", loc = t === "localization", pose = t === "pose";
  $$(".t-pose").forEach(e => e.hidden = !pose);
  $$(".t-hom").forEach(e => e.hidden = !hom);
  $("#try-pairs").hidden = loc; $("#try-loc").hidden = !loc;
  $("#try-title").textContent = loc ? "Localize a photo." : "Match two photos.";
  $("#try-lead").textContent = loc
    ? "Pick an Aachen day or night photo — XFeat finds its place in a 3D map of the old town, like HLoc."
    : hom
    ? "Drop two views of the same planar scene — or start from an HPatches sample to get ground-truth corner error."
    : S.ds === "scannet"
    ? "Drop any pair of images — or start from a ScanNet-1500 indoor sample to get ground-truth pose metrics."
    : "Drop any pair of images — or start from a MegaDepth-1500 sample to get ground-truth pose metrics.";
  $("#bench-pose").hidden = !pose; $("#bench-hom").hidden = !hom; $("#bench-loc").hidden = !loc;
  if (!loc) {   // reset the pair slots
    S.pairId = null; S.files = [null, null];
    $$(".drop").forEach(d => { d.classList.remove("has"); $("img", d).removeAttribute("src"); });
    updateRun();
  }
  // results sections follow the task
  $("#results").hidden = !pose || !S.result;
  $("#h-results").hidden = !hom || !HS.res;
  $("#l-results").hidden = !loc || !LS.res;
  const has = pose ? S.result : hom ? HS.res : LS.res;
  $("#nav-results").hidden = !has;
  $("#nav-results").href = pose ? "#results" : hom ? "#h-results" : "#l-results";
  if (loc) { lLoadSamples(); aPoll(); }
  else {
    if (hom ? S.hasHP : true) loadSamples();
    if (hom) hPollBench(); else if (S.ds === "scannet") sPollBench(); else pollBench();
  }
  if (loc && LS.res) requestAnimationFrame(lDraw);
  else if (hom && HS.res) requestAnimationFrame(hDraw);
  else if (pose && S.result) requestAnimationFrame(drawAll);
}
$$("#task-seg button, #bench-task-seg button").forEach(b => b.onclick = () => setTask(b.dataset.t));

/* ------------------------------------------------------------------- match */
async function hRun() {
  const btn = $("#run"); showErr($("#err"), "");
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Matching…';
  const fd = new FormData();
  if (S.pairId !== null) fd.append("pair_id", S.pairId);
  else { fd.append("img0", S.files[0]); fd.append("img1", S.files[1]); }
  fd.append("short_side", $("#oh-short").value);
  fd.append("thr", $("#oh-thr").value);
  fd.append("top_k", $("#oh-k").value);
  try {
    const r = await fetch("/api/homography", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    HS.res = await r.json();
    const src = S.pairId !== null ? [`/api/hpatches/img/${S.pairId}/0`, `/api/hpatches/img/${S.pairId}/1`] : S.urls;
    HS.imgs = await Promise.all(src.map(loadImg));
    HS.warp = HS.res.warp ? await loadImg(HS.res.warp) : null;
    hRender();
    $("#h-results").hidden = false; $("#nav-results").hidden = false;
    requestAnimationFrame(() => { hDraw(); $("#h-results").scrollIntoView({ behavior: "smooth" }); });
  } catch (e) { showErr($("#err"), "Matching failed: " + e.message); }
  btn.textContent = "Match"; updateRun();
}

/* ------------------------------------------------------------------ drawing */
function hDraw() {
  const R = HS.res; if (!R) return;
  const m = R.method.metrics;
  const txt = [R.has_gt ? `corner err: ${m.corner_err === null ? "failed" : fmt(m.corner_err, 2) + " px"}` : null,
    `inliers: ${m.inliers}`, `MIR: ${fmt(m.mir, 2)}`].filter(Boolean).join(" | ");
  paintMatches($("#cv-h-match"), R.method, R.size0, R.size1, HS.imgs, txt, "XFeat");
  drawRegistration();
}
function drawRegistration() {
  const R = HS.res, cv = $("#cv-h-reg"), [w1, h1] = R.size1;
  const W = cv.clientWidth, dpr = window.devicePixelRatio || 1, s = W / w1, H = h1 * s;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.drawImage(HS.warp || HS.imgs[1], 0, 0, W, H);
  const quad = (q, color, dash) => {
    if (!q) return;
    c.strokeStyle = color; c.lineWidth = 3; c.setLineDash(dash); c.lineJoin = "round"; c.beginPath();
    q.forEach(([x, y], i) => i ? c.lineTo(x * s, y * s) : c.moveTo(x * s, y * s)); c.closePath(); c.stroke();
    c.setLineDash([]);
  };
  quad(R.quads.gt, "#30d158", []);
  quad(R.quads.est, "#2997ff", [10, 6]);
  const leg = [];
  if (R.quads.gt) leg.push(`<span><i style="background:#30d158"></i>Ground truth</span>`);
  if (R.quads.est) leg.push(`<span><i style="background:#2997ff"></i>XFeat estimate</span>`);
  if (!R.quads.est) leg.push(`<span>Homography could not be estimated</span>`);
  else leg.push(`<span>Blend: image 2 + warped image 1</span>`);
  $("#h-legend").innerHTML = leg.join("");
}
$$("#h-seg-show button").forEach(b => b.onclick = () => {
  $$("#h-seg-show button").forEach(x => x.classList.toggle("on", x === b)); S.show = b.dataset.v; hDraw();
});
$("#h-density").oninput = e => { S.density = +e.target.value; hDraw(); };
const savePng = (id, name) => $(id).toBlob(bl => {
  const a = document.createElement("a"); a.href = URL.createObjectURL(bl); a.download = name; a.click();
});
$("#h-save-m").onclick = () => savePng("#cv-h-match", "xfeat_hpatches_matches.png");
$("#h-save-r").onclick = () => savePng("#cv-h-reg", "xfeat_hpatches_registration.png");
window.addEventListener("resize", () => { if (S.task === "homography") { clearTimeout(hDraw.t); hDraw.t = setTimeout(hDraw, 120); } });

/* -------------------------------------------------------------- pair table */
function hRender() {
  const R = HS.res, m = R.method.metrics, gt = R.has_gt;
  $("#h-sub").textContent = gt
    ? "Ground-truth homography is available, so the corner error is reported as in the paper."
    : "No ground truth for uploaded images. The homography is estimated with MAGSAC++ and shown as an overlay.";
  const one = (label, v) => ({ label, dir: 0, raw: [0, 0], cells: [v] });
  const rows = [
    { group: "Matching" },
    one("Features per image", `${int(m.kpts0)} / ${int(m.kpts1)}`),
    one("Matches (mutual NN)", int(m.matches)),
    one("Inliers (MAGSAC++)", int(m.inliers)),
    one("Inlier ratio (MIR)", fmt(m.mir, 2)),
    one("Time (ms)", fmt(m.time_ms, 0)),
  ];
  if (gt) {
    const e = m.corner_err;
    rows.push({ group: "Homography accuracy" },
      one("Mean corner error (px)", e === null ? "failed" : fmt(e, 2)),
      ...[3, 5, 7].map(t => one(`Within ${t} px`, e !== null && e <= t ? "✓" : "✗")));
  }
  table($("#h-pair-table"), ["", "XFeat"], rows);
  $("#h-pair-note").textContent = `Matched at ${R.size0[0]}×${R.size0[1]} and ${R.size1[0]}×${R.size1[1]} · MAGSAC++ threshold ${R.thr} px` +
    (gt ? ` · ${R.meta.seq}, image 1 → ${R.meta.k} (${R.meta.group})` : "") + " · MHA needs many pairs — see the benchmark below.";
}

/* ---------------------------------------------------------------- benchmark */
let hPaper = null;
const GROUPS = [["i", "illumination", "Illumination"], ["v", "viewpoint", "Viewpoint"]];
const T = [3, 5, 7];

fetch("/static/data/paper_hpatches.json").then(r => r.json()).then(p => {
  hPaper = p;
  let h = `<thead><tr><th rowspan="2">Method</th><th colspan="3" style="text-align:center">Illumination · MHA</th><th colspan="3" style="text-align:center">Viewpoint · MHA</th></tr>` +
    `<tr>${[...T, ...T].map(t => `<th>@${t}</th>`).join("")}</tr></thead><tbody>`;
  let cls = "";
  p.rows.forEach(r => {
    if (r.cls !== cls) { cls = r.cls; h += `<tr class="grp"><td colspan="7">${cls}</td></tr>`; }
    h += `<tr${r.key ? ' class="hl"' : ""}><td>${r.m}</td>${[...r.i, ...r.v].map(v => `<td>${v.toFixed(1)}</td>`).join("")}</tr>`;
  });
  $("#bh-paper-table").innerHTML = h + "</tbody>";
  hRenderCmp(null);
});

function hRenderCmp(res, partial) {
  if (!hPaper) return;
  const row = hPaper.rows.find(r => r.key);
  const mine = res ? GROUPS.flatMap(([, g]) => T.map(t => res.groups[g]["mha" + t])) : null;
  const pv = [...row.i, ...row.v];
  let h = `<thead><tr><th rowspan="2">Method</th><th colspan="3" style="text-align:center">Illumination · MHA</th><th colspan="3" style="text-align:center">Viewpoint · MHA</th></tr>` +
    `<tr>${[...T, ...T].map(t => `<th>@${t} px</th>`).join("")}</tr></thead><tbody>`;
  h += `<tr class="hl"><td>XFeat — paper</td>${pv.map(v => `<td>${fmt(v, 1)}</td>`).join("")}</tr>`;
  if (mine) {
    h += `<tr class="hl"><td>XFeat — this run</td>${mine.map(v => `<td class="best">${fmt(v, 1)}</td>`).join("")}</tr>`;
    h += `<tr class="sub"><td>Δ</td>${mine.map((v, i) => {
      if (v === null) return "<td></td>";
      const d = v - pv[i]; return `<td class="${d >= 0 ? "pos" : "neg"}">${d >= 0 ? "+" : ""}${fmt(d, 1)}</td>`;
    }).join("")}</tr>`;
  }
  $("#bh-cmp-table").innerHTML = h + "</tbody>";
  if (res) {
    const n = GROUPS.map(([, g, name]) => `${res.groups[g].n} ${name.toLowerCase()}`).join(" + ");
    const c = res.cfg;
    $("#bh-cmp-note").textContent = partial
      ? `In progress — ${res.pairs} pairs so far (${n}).`
      : `${res.pairs} pairs (${n}) · ${c.short_side}px shorter side · ${c.top_k} keypoints · MAGSAC++ ${c.thr} px · ${res.device.toUpperCase()} · ${res.date.replace("T", " ")}`;
  }
}

let hPoll = null;
async function hPollBench() {
  let s;
  try { s = await (await fetch("/api/benchmark_h/status")).json(); } catch { return; }
  const running = s.status === "running";
  $("#bh-run").hidden = running; $("#bh-cancel").hidden = !running;
  $("#bh-run").textContent = s.resumable ? "Resume benchmark" : (s.result ? "Run again" : "Run benchmark");
  $("#bh-bar").hidden = !running;
  showErr($("#bh-err"), s.status === "error" ? s.error : "");
  if (running) {
    $("#bh-bar i").style.width = (100 * s.done / Math.max(1, s.total)) + "%";
    const eta = s.eta_s ? ` · about ${s.eta_s > 90 ? Math.ceil(s.eta_s / 60) + " min" : Math.ceil(s.eta_s) + " s"} left` : "";
    $("#bh-info").textContent = `${s.done.toLocaleString()} / ${s.total.toLocaleString()} pairs${eta}`;
    if (s.live) hRenderCmp({ groups: s.live, pairs: s.done, cfg: { short_side: 480, top_k: 4096, thr: 3 }, device: "cpu", date: "" }, true);
    if (!hPoll) hPoll = setInterval(hPollBench, 1500);
  } else {
    if (hPoll) { clearInterval(hPoll); hPoll = null; }
    $("#bh-info").textContent = s.status === "cancelled" ? "Stopped — progress is saved." : "";
    if (s.result) hRenderCmp(s.result);
  }
}
$("#bh-run").onclick = async () => {
  const q = new URLSearchParams({ short_side: $("#oh-short").value, thr: $("#oh-thr").value, top_k: $("#oh-k").value });
  const r = await fetch("/api/benchmark_h/start?" + q, { method: "POST" });
  if (!r.ok) showErr($("#bh-err"), (await r.json().catch(() => ({}))).detail || "Could not start");
  setTimeout(hPollBench, 400);
};
$("#bh-cancel").onclick = async () => { await fetch("/api/benchmark_h/cancel", { method: "POST" }); };
