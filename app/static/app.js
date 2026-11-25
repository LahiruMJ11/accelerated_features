"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = (v, d = 1) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const int = v => (v === null || v === undefined ? "—" : Math.round(v).toLocaleString());

/* ------------------------------------------------------------ reveal on scroll */
const io = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))),
  { threshold: .12 });
$$(".reveal").forEach(el => io.observe(el));

/* ------------------------------------------------------------------- state */
const S = {
  files: [null, null], urls: [null, null], pairId: null,
  result: null, imgs: [null, null], show: "inl", density: 100,
  task: "pose", ds: "megadepth", hasHP: false, hasSN: false,
};

/* -------------------------------------------------------------- health/chip */
fetch("/api/health").then(r => r.json()).then(h => {
  $("#device-chip").textContent = h.device.toUpperCase();
  S.hasHP = !!h.hpatches; S.hasSN = !!h.scannet;
  if (!h.scannet) $$("[data-ds=scannet]").forEach(b => b.hidden = true);
  if (!h.aachen) $$("[data-t=localization]").forEach(b => b.hidden = true);
  if (h.dataset || h.hpatches) { $("#samples-wrap").hidden = false; loadSamples(); }
  if (!h.hpatches) $$("[data-t=homography]").forEach(b => b.hidden = true);
}).catch(() => { $("#device-chip").textContent = "offline"; });

/* ------------------------------------------------------------------ uploads */
function setSlot(k, src) {
  const d = $("#drop" + k);
  $("img", d).src = src; d.classList.add("has");
}
function setFile(k, f) {
  if (!f || !f.type.startsWith("image/")) return;
  S.files[k] = f; S.pairId = null; $$(".sample").forEach(s => s.classList.remove("on"));
  if (S.urls[k]) URL.revokeObjectURL(S.urls[k]);
  S.urls[k] = URL.createObjectURL(f);
  setSlot(k, S.urls[k]); updateRun();
}
$$(".drop").forEach(d => {
  const k = +d.dataset.k, inp = $("input", d);
  inp.addEventListener("change", () => setFile(k, inp.files[0]));
  ["dragenter", "dragover"].forEach(ev => d.addEventListener(ev, e => { e.preventDefault(); d.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => d.addEventListener(ev, e => { e.preventDefault(); d.classList.remove("over"); }));
  d.addEventListener("drop", e => setFile(k, e.dataTransfer.files[0]));
});
function updateRun() { $("#run").disabled = !(S.pairId !== null || (S.files[0] && S.files[1])); }

/* ----------------------------------------------------------------- samples */
async function loadSamples(seed) {
  const hp = S.task === "homography", sn = !hp && S.ds === "scannet";
  const base = hp ? "/api/hpatches" : sn ? "/api/scannet" : "/api";
  const list = await (await fetch(`${base}/pairs?n=24` + (seed ? "&seed=" + seed : ""))).json();
  $("#samples-title").textContent = hp ? "HPatches samples" : sn ? "ScanNet-1500 samples" : "MegaDepth-1500 samples";
  const row = $("#samples"); row.innerHTML = "";
  list.forEach(p => {
    const u = k => hp ? `/api/hpatches/img/${p.id}/${k}` : sn ? `/api/scannet/img/${p.id}/${k}` : `/api/pair/${p.id}/img/${k}`;
    const b = document.createElement("button"); b.className = "sample"; b.dataset.id = p.id;
    const cap = hp ? `${p.seq} · 1→${p.k}` : `Scene ${p.scene} · #${p.id}`;
    b.innerHTML = `<div><img loading="lazy" src="${u(0)}?thumb=true"><img loading="lazy" src="${u(1)}?thumb=true"></div><span>${cap}</span>`;
    b.onclick = () => {
      $$(".sample").forEach(s => s.classList.remove("on")); b.classList.add("on");
      S.pairId = p.id; S.files = [null, null];
      setSlot(0, u(0)); setSlot(1, u(1)); updateRun();
    };
    row.appendChild(b);
  });
}
$("#shuffle").onclick = () => loadSamples(Math.floor(Math.random() * 1e6));

/* ------------------------------------------------------------------- match */
function showErr(el, msg) { el.textContent = msg; el.hidden = !msg; }
function loadImg(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; }); }

$("#run").onclick = async () => {
  if (S.task === "homography") return hRun();
  const btn = $("#run"); showErr($("#err"), "");
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Matching…';
  const fd = new FormData();
  if (S.pairId !== null) fd.append("pair_id", S.pairId);
  else { fd.append("img0", S.files[0]); fd.append("img1", S.files[1]); }
  fd.append("max_dim", $("#opt-maxdim").value);
  fd.append("dataset", S.ds);
  if ($("#opt-thr").value !== "") fd.append("ransac_thr", $("#opt-thr").value);   // empty = auto (ScanNet: best from benchmark)
  fd.append("top_k_sparse", $("#opt-k1").value);
  fd.append("top_k_star", $("#opt-k2").value);
  try {
    const r = await fetch("/api/match", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    S.result = await r.json();
    const pb = S.ds === "scannet" ? "/api/scannet/img" : "/api/pair";
    const pu = k => S.ds === "scannet" ? `${pb}/${S.pairId}/${k}` : `${pb}/${S.pairId}/img/${k}`;
    const src = [S.pairId !== null ? pu(0) : S.urls[0], S.pairId !== null ? pu(1) : S.urls[1]];
    S.imgs = await Promise.all(src.map(loadImg));
    renderResults();
    $("#results").hidden = false; $("#nav-results").hidden = false;
    requestAnimationFrame(() => { drawAll(); $("#results").scrollIntoView({ behavior: "smooth" }); });
  } catch (e) { showErr($("#err"), "Matching failed: " + e.message); }
  btn.textContent = "Match"; updateRun();
};

/* -------------------------------------------------------------- drawing */
const LABEL = { xfeat: "XFeat", xfeat_star: "XFeat*" };
function banner(m, gt) {
  const p = [];
  if (gt) p.push(`R_err: ${fmt(m.R_err, 2)}`, `t_err: ${fmt(m.t_err, 2)}`);
  p.push(`inliers: ${m.inliers}`, `MIR: ${fmt(m.mir, 2)}`);
  return p.join(" | ");
}
function paintMatches(cv, d, size0, size1, imgs, bannerTxt, tag) {
  const [i0, i1] = imgs, [w0, h0] = size0, [w1, h1] = size1;
  const W = cv.clientWidth, dpr = window.devicePixelRatio || 1, gap = 6;
  const s = (W - gap) / (w0 + w1), H = Math.max(h0, h1) * s;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = "#000"; c.fillRect(0, 0, W, H);
  c.drawImage(i0, 0, 0, w0 * s, h0 * s);
  const ox = w0 * s + gap; c.drawImage(i1, ox, 0, w1 * s, h1 * s);

  const n = d.inl.length, step = Math.max(1, Math.round(100 / S.density));
  const draw = (wantInl, color, lw) => {
    c.strokeStyle = color; c.lineWidth = lw; c.beginPath();
    for (let i = 0; i < n; i += step) {
      if (!!d.inl[i] !== wantInl) continue;
      const o = i * 4;
      c.moveTo(d.pts[o] * s, d.pts[o + 1] * s); c.lineTo(ox + d.pts[o + 2] * s, d.pts[o + 3] * s);
    }
    c.stroke();
  };
  if (S.show === "all") draw(false, "rgba(255,69,58,.4)", .5);
  draw(true, "rgba(48,255,90,.4)", .55);

  // paper-style banner + method tag
  c.font = "600 " + Math.max(11, Math.min(15, W / 70)) + "px SF Mono, Menlo, monospace";
  const tw = c.measureText(bannerTxt).width + 20, bh = 26;
  c.fillStyle = "rgba(0,0,0,.72)"; c.fillRect(0, 0, tw, bh);
  c.fillStyle = "#fff"; c.textBaseline = "middle"; c.fillText(bannerTxt, 10, bh / 2 + 1);
  c.font = "700 " + Math.max(13, Math.min(18, W / 55)) + "px -apple-system, Helvetica, sans-serif";
  const tg = c.measureText(tag).width + 22, x = W - tg - 12, y = 12;
  c.fillStyle = "rgba(255,255,255,.92)"; c.beginPath(); c.roundRect(x, y, tg, 30, 8); c.fill();
  c.fillStyle = "#000"; c.fillText(tag, x + 11, y + 16);
}
function drawMatches(key) {
  const R = S.result, d = R.methods[key];
  paintMatches($("#cv-" + key), d, R.size0, R.size1, S.imgs, banner(d.metrics, R.has_gt), LABEL[key]);
}
function drawAll() { if (S.task === "localization") return lDraw(); if (S.task === "homography") return hDraw(); if (S.result) Object.keys(LABEL).forEach(drawMatches); }
window.addEventListener("resize", () => { clearTimeout(drawAll.t); drawAll.t = setTimeout(drawAll, 120); });
$$("#seg-show button").forEach(b => b.onclick = () => {
  $$("#seg-show button").forEach(x => x.classList.toggle("on", x === b)); S.show = b.dataset.v; drawAll();
});
$("#density").oninput = e => { S.density = +e.target.value; drawAll(); };
$$("[data-save]").forEach(b => b.onclick = () => {
  $("#cv-" + b.dataset.save).toBlob(bl => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(bl);
    a.download = `${b.dataset.save}_matches.png`; a.click();
  });
});

/* ------------------------------------------------------------- pair table */
function table(el, head, rows) {
  const best = (vals, dir) => {
    if (!dir) return -1; const nums = vals.map(Number);
    if (nums.some(isNaN)) return -1;
    const t = dir > 0 ? Math.max(...nums) : Math.min(...nums);
    return nums[0] === nums[1] ? -1 : nums.indexOf(t);
  };
  let h = `<thead><tr>${head.map(x => `<th>${x}</th>`).join("")}</tr></thead><tbody>`;
  rows.forEach(r => {
    if (r.group) { h += `<tr class="grp"><td colspan="${head.length}">${r.group}</td></tr>`; return; }
    const b = best(r.raw, r.dir);
    h += `<tr><td>${r.label}</td>${r.cells.map((c, i) => `<td class="${i === b ? "best" : ""}">${c}</td>`).join("")}</tr>`;
  });
  el.innerHTML = h + "</tbody>";
}
function renderResults() {
  const R = S.result, a = R.methods.xfeat.metrics, b = R.methods.xfeat_star.metrics, gt = R.has_gt;
  $("#res-sub").textContent = gt
    ? "Ground-truth camera pose is available for this pair, so pose errors are reported as in the paper."
    : "No ground-truth pose for this pair. Inliers come from a fundamental-matrix MAGSAC fit.";
  const row = (label, f, dir, d = 0) => ({ label, dir, raw: [f(a), f(b)], cells: [f(a), f(b)].map(v => d ? fmt(v, d) : int(v)) });
  const rows = [
    { group: "Matching" },
    { label: "Features per image", dir: 0, raw: [0, 0], cells: [`${int(a.kpts0)} / ${int(a.kpts1)}`, `${int(b.kpts0)} / ${int(b.kpts1)}`] },
    { label: "Coarse mutual matches", dir: 0, raw: [0, 0], cells: ["—", int(b.coarse_matches)] },
    row("Final matches", m => m.matches, 1),
    row("Inliers", m => m.inliers, 1),
    row("Inlier ratio (MIR)", m => m.mir, 1, 2),
    row("Time (ms)", m => m.time_ms, -1, 0),
  ];
  if (gt) {
    const ok = (m, t) => m.pose_err !== null && m.pose_err <= t;
    rows.push({ group: "Pose accuracy" },
      row("Rotation error (°)", m => m.R_err, -1, 2),
      row("Translation error (°)", m => m.t_err, -1, 2),
      row("Pose error, max (°)", m => m.pose_err, -1, 2),
      ...[5, 10, 20].map(t => ({ label: `Within ${t}°`, dir: 0, raw: [0, 0], cells: [ok(a, t), ok(b, t)].map(v => v ? "✓" : "✗") })));
  }
  table($("#pair-table"), ["", "XFeat", "XFeat*"], rows);
  $("#pair-note").textContent = `Matched at ${R.size0[0]}×${R.size0[1]} and ${R.size1[0]}×${R.size1[1]} · RANSAC threshold ${typeof R.ransac_thr === "object" ? Object.entries(R.ransac_thr).map(([k, v]) => (k === "xfeat" ? "XFeat " : "XFeat* ") + v).join(", ") : R.ransac_thr} px · AUC needs many pairs — see the benchmark below.`;
}

/* ------------------------------------------------------------- benchmark */
let paper = null;
const METRICS = [["auc5", "AUC@5°", 1], ["auc10", "AUC@10°", 1], ["auc20", "AUC@20°", 1], ["acc10", "Acc@10°", 1], ["mir", "MIR", 2], ["inliers", "#inliers", 0], ["fps", "FPS", 1]];

fetch("/static/data/paper.json").then(r => r.json()).then(p => {
  paper = p;
  const head = ["Method", "AUC@5°", "AUC@10°", "AUC@20°", "Acc@10°", "MIR", "#inliers", "dim", "FPS"];
  let h = `<thead><tr>${head.map(x => `<th>${x}</th>`).join("")}</tr></thead><tbody>`;
  let cls = "";
  p.rows.forEach(r => {
    if (r.cls !== cls) { cls = r.cls; h += `<tr class="grp"><td colspan="9">${cls}</td></tr>`; }
    const hl = r.key ? ' class="hl"' : "";
    h += `<tr${hl}><td>${r.m}</td><td>${r.auc5}</td><td>${r.auc10}</td><td>${r.auc20}</td><td>${r.acc10}</td><td>${r.mir.toFixed(2)}</td><td>${r.inliers.toLocaleString()}</td><td>${r.dim}</td><td>${r.fps}</td></tr>`;
  });
  $("#paper-table").innerHTML = h + "</tbody>";
  renderCmp(null);
});

function renderCmp(res) {
  if (!paper) return;
  const head = ["Method", ...METRICS.map(m => m[1])];
  let h = `<thead><tr>${head.map(x => `<th>${x}</th>`).join("")}</tr></thead><tbody>`;
  paper.rows.filter(r => r.key).forEach(r => {
    const mine = res && res.methods[r.key];
    const pv = k => k === "fps" ? r.fpsv : r[k];
    h += `<tr class="hl"><td>${r.m} — paper</td>${METRICS.map(([k, , d]) => `<td>${fmt(pv(k), d)}</td>`).join("")}</tr>`;
    if (mine) {
      h += `<tr class="hl"><td>${r.m} — this run</td>${METRICS.map(([k, , d]) => `<td class="best">${fmt(mine[k], d)}</td>`).join("")}</tr>`;
      h += `<tr class="sub"><td>Δ</td>${METRICS.map(([k, , d]) => {
        if (mine[k] === null || mine[k] === undefined) return "<td></td>";
        const dv = mine[k] - pv(k); const good = k === "fps" ? null : dv >= 0;
        return `<td class="${good === null ? "" : good ? "pos" : "neg"}">${dv >= 0 ? "+" : ""}${fmt(dv, d)}</td>`;
      }).join("")}</tr>`;
    }
  });
  $("#cmp-table").innerHTML = h + "</tbody>";
  if (res) $("#cmp-note").textContent =
    `${res.pairs.toLocaleString()} pairs · RANSAC ${res.ransac_thr} px · run on ${res.device.toUpperCase()} · ${res.date.replace("T", " ")}. FPS is per frame at VGA (extract + match against a cached reference) on this machine, so it will not equal the paper's i5 laptop figures.`;
}

let poll = null;
async function pollBench() {
  let s;
  try { s = await (await fetch("/api/benchmark/status")).json(); } catch { return; }
  const running = s.status === "running";
  $("#bench-run").hidden = running; $("#bench-cancel").hidden = !running;
  $("#bench-run").textContent = s.resumable ? "Resume benchmark" : (s.result ? "Run again" : "Run benchmark");
  $("#bench-bar").hidden = !running;
  showErr($("#bench-err"), s.status === "error" ? s.error : "");
  if (running) {
    $("#bench-bar i").style.width = (100 * s.done / Math.max(1, s.total)) + "%";
    const eta = s.eta_s ? ` · about ${s.eta_s > 90 ? Math.ceil(s.eta_s / 60) + " min" : Math.ceil(s.eta_s) + " s"} left` : "";
    $("#bench-info").textContent = `${s.done.toLocaleString()} / ${s.total.toLocaleString()} pairs${eta}`;
    if (s.live) renderCmp({ methods: Object.fromEntries(Object.entries(s.live).map(([k, v]) => [k, { ...v, fps: null }])),
      pairs: s.done, ransac_thr: 2.5, device: "cpu", date: "in progress" });
    if (!poll) poll = setInterval(pollBench, 1500);
  } else {
    if (poll) { clearInterval(poll); poll = null; }
    $("#bench-info").textContent = s.status === "cancelled" ? "Stopped — progress is saved." : "";
    if (s.result) renderCmp(s.result);
  }
}
$("#bench-run").onclick = async () => {
  const r = await fetch("/api/benchmark/start", { method: "POST" });
  if (!r.ok) showErr($("#bench-err"), (await r.json().catch(() => ({}))).detail || "Could not start");
  setTimeout(pollBench, 400);
};
$("#bench-cancel").onclick = async () => { await fetch("/api/benchmark/cancel", { method: "POST" }); };
pollBench();
