"use strict";
/* ScanNet-1500 relative pose (paper Table 2). Relies on globals from app.js / homography.js. */

/* ------------------------------------------------------- dataset switching */
function setDataset(ds) {
  if (S.ds === ds) return;
  S.ds = ds;
  $$("#ds-seg button, #bench-ds-seg button").forEach(b => b.classList.toggle("on", b.dataset.ds === ds));
  const sn = ds === "scannet";
  $("#bench-md").hidden = sn; $("#bench-sn").hidden = !sn;
  $("#opt-thr").value = sn ? "" : "2.5";                    // empty = auto (best threshold from the benchmark)
  $("#try-lead").textContent = sn
    ? "Drop any pair of images — or start from a ScanNet-1500 indoor sample to get ground-truth pose metrics."
    : "Drop any pair of images — or start from a MegaDepth-1500 sample to get ground-truth pose metrics.";
  S.pairId = null; S.files = [null, null]; S.result = null;
  $$(".drop").forEach(d => { d.classList.remove("has"); $("img", d).removeAttribute("src"); });
  $("#results").hidden = true; $("#nav-results").hidden = true;
  updateRun(); loadSamples();
  if (sn) sPollBench(); else pollBench();
}
$$("#ds-seg button, #bench-ds-seg button").forEach(b => b.onclick = () => setDataset(b.dataset.ds));

/* ---------------------------------------------------------------- benchmark */
let sPaper = null;
const AUCS = [["auc5", "AUC@5°"], ["auc10", "AUC@10°"], ["auc20", "AUC@20°"]];

fetch("/static/data/paper_scannet.json").then(r => r.json()).then(p => {
  sPaper = p;
  let h = `<thead><tr><th>Method</th>${AUCS.map(a => `<th>${a[1]}</th>`).join("")}</tr></thead><tbody>`;
  let cls = "";
  p.rows.forEach(r => {
    if (r.cls !== cls) { cls = r.cls; h += `<tr class="grp"><td colspan="4">${cls}</td></tr>`; }
    h += `<tr${r.key ? ' class="hl"' : ""}><td>${r.m}</td>${r.v.map(v => `<td>${v.toFixed(1)}</td>`).join("")}</tr>`;
  });
  $("#bs-paper-table").innerHTML = h + "</tbody>";
  sRenderCmp(null);
});

function sRenderCmp(res) {
  if (!sPaper) return;
  let h = `<thead><tr><th>Method</th>${AUCS.map(a => `<th>${a[1]}</th>`).join("")}<th>RANSAC px</th></tr></thead><tbody>`;
  sPaper.rows.filter(r => r.key).forEach(r => {
    const mine = res && res.methods[r.key];
    h += `<tr class="hl"><td>${r.m} — paper</td>${r.v.map(v => `<td>${fmt(v, 1)}</td>`).join("")}<td>—</td></tr>`;
    if (mine) {
      h += `<tr class="hl"><td>${r.m} — this run</td>${AUCS.map(([k]) => `<td class="best">${fmt(mine[k], 1)}</td>`).join("")}<td>${mine.best_thr}</td></tr>`;
      h += `<tr class="sub"><td>Δ</td>${AUCS.map(([k], i) => {
        const d = mine[k] - r.v[i]; return `<td class="${d >= 0 ? "pos" : "neg"}">${d >= 0 ? "+" : ""}${fmt(d, 1)}</td>`;
      }).join("")}<td></td></tr>`;
    }
  });
  $("#bs-cmp-table").innerHTML = h + "</tbody>";
  if (res) $("#bs-cmp-note").textContent =
    `${res.pairs.toLocaleString()} pairs · ${res.resolution} · best RANSAC threshold of 12 (0.5–6 px) per method, by mean AUC · ${res.device.toUpperCase()} · ${res.date.replace("T", " ")}`;
}

let sPoll = null;
async function sPollBench() {
  let s;
  try { s = await (await fetch("/api/benchmark_s/status")).json(); } catch { return; }
  const running = s.status === "running";
  $("#bs-run").hidden = running; $("#bs-cancel").hidden = !running;
  $("#bs-run").textContent = s.result ? "Run again" : (s.resumable ? "Resume benchmark" : "Run benchmark");
  $("#bs-bar").hidden = !running;
  showErr($("#bs-err"), s.status === "error" ? s.error : "");
  if (running) {
    $("#bs-bar i").style.width = (100 * s.done / Math.max(1, s.total)) + "%";
    const eta = s.eta_s ? ` · about ${s.eta_s > 90 ? Math.ceil(s.eta_s / 60) + " min" : Math.ceil(s.eta_s) + " s"} left` : "";
    const what = s.phase === "sweeping" ? `RANSAC thresholds ${s.done} / ${s.total}` : `Matching ${s.done.toLocaleString()} / ${s.total.toLocaleString()} pairs`;
    $("#bs-info").textContent = what + eta;
    if (!sPoll) sPoll = setInterval(sPollBench, 1500);
  } else {
    if (sPoll) { clearInterval(sPoll); sPoll = null; }
    $("#bs-info").textContent = s.status === "cancelled" ? "Stopped — matches are saved, a re-run resumes." : "";
    if (s.result) sRenderCmp(s.result);
  }
}
$("#bs-run").onclick = async () => {
  const r = await fetch("/api/benchmark_s/start", { method: "POST" });
  if (!r.ok) showErr($("#bs-err"), (await r.json().catch(() => ({}))).detail || "Could not start");
  setTimeout(sPollBench, 400);
};
$("#bs-cancel").onclick = async () => { await fetch("/api/benchmark_s/cancel", { method: "POST" }); };
sPollBench();
