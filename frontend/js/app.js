import { api, followJob } from "./api.js";
import { barChart, historyChart, trainValChart } from "./charts.js";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const VIEWS = { home: "Home", preview: "Preview", performance: "Performance Analysis", chart: "Chart", about: "About", data: "Data", train: "Train" };
const ALIASES = { overview: "home", predict: "preview", evaluate: "performance" };  // old links keep working
const PIPELINE_VIEWS = ["data", "train"];
const AUTH_KEY = "bce_auth";
const RING_C = 2 * Math.PI * 52;

const state = {
  view: "overview", status: null, job: null, followingId: null, stopFollow: null, lastStage: null,
  results: null, file: null, aboutLoaded: false, backbonesBuilt: false, cmKey: null, chartKey: null, started: false,
};

// ---------- helpers ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 4) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(d)}%`);
const dn = (k) => state.status?.display_names?.[k] || k;
const short = (c) => c.replace("[Malignant] ", "");
const inline = (md) => esc(md).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
const jobActive = () => !!state.status?.active_job || ["queued", "running"].includes(state.job?.state);

let toastTimer;
function toast(msg, kind = "info") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 5000);
}

function setMsg(el, text, kind = "") {
  el.textContent = text;
  el.className = `form-msg ${kind}`;
}

function md(text) {
  // Minimal markdown: headings, pipe tables, paragraphs, inline code/bold/italics.
  const out = [];
  const lines = (text || "").split("\n");
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i].trim();
    if (!ln) continue;
    if (ln.startsWith("#")) { out.push(`<h4>${inline(ln.replace(/^#+\s*/, ""))}</h4>`); continue; }
    if (ln.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++].trim());
      i--;
      const cells = rows.filter((r) => !/^\|[\s|:-]+\|$/.test(r)).map((r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
      const [head, ...body] = cells;
      out.push(`<table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join("")}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    out.push(`<p>${inline(ln)}</p>`);
  }
  return out.join("");
}

// ---------- routing ----------
function route() {
  if (!isAuthed()) return showLogin();
  const raw = location.hash.replace(/^#\/?/, "") || "home";
  const v = ALIASES[raw] || raw;
  show(VIEWS[v] ? v : "home");
}

function show(view) {
  state.view = view;
  $$("[data-view]").forEach((s) => { s.hidden = s.dataset.view !== view; });
  $$("[data-nav]").forEach((a) => {
    if (a.dataset.nav === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("#view-title").textContent = VIEWS[view];
  if (PIPELINE_VIEWS.includes(view)) setPipelineOpen(true);
  document.title = `${VIEWS[view]} · ALL Hybrid Ensemble`;
  closeMenu();
  render(view);
}

async function render(view = state.view) {
  try {
    await ({ home: renderOverview, data: renderData, train: renderTrain, performance: renderEvaluate, preview: renderPredict, chart: renderChart, about: renderAbout })[view]();
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------- status ----------
async function refreshStatus() {
  state.status = await api.status();
  const s = state.status;
  $("#demo-banner").hidden = !s.demo_mode;
  $("#smoke-banner").hidden = !s.smoke;
  if (s.active_job && state.followingId !== s.active_job) track(s.active_job);
  if (!state.job || !jobActive()) setChip(null);
  $$("[data-job-button]").forEach((b) => { b.disabled = jobActive() || b.dataset.blocked === "1"; });
  renderStepper();
  renderSettings();
  return s;
}

const isSingle = () => state.status?.settings?.active_mode === "mobilenetv2";

function renderSettings() {
  const st = state.status.settings;
  $("#single-mode").checked = st.active_mode === "mobilenetv2";
  $("#mode-label").textContent = `Predict uses: ${st.label}`;
}

async function changeActiveMode() {
  const box = $("#single-mode");
  try {
    const st = await api.setActiveMode(box.checked ? "mobilenetv2" : "ensemble");
    toast(`Predict now uses ${st.label}.`);
    state.results = null;
    await refreshStatus();
    render();
  } catch (e) {
    box.checked = !box.checked;
    toast(e.message, "error");
  }
}

function renderStepper() {
  const s = state.status;
  if (!s) return;
  const icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7"/></svg>';
  $("#stepper").innerHTML = s.steps.map((st, i) => `
    <li class="step" data-step="${st.key}" data-state="${st.state}">
      <span class="step-dot">${st.state === "done" ? icon : i + 1}</span>
      <span>${esc(st.label)}</span>
      <span class="step-state">${st.state}</span>
    </li>`).join("");
  $("#next-step").textContent = s.next_step;
}

// ---------- jobs ----------
async function startJob(call, label) {
  try {
    const r = await call();
    toast(`${label} started.`);
    track(r.job_id);
    onJob(r.job);
    await refreshStatus();
  } catch (e) {
    toast(e.message, "error");
  }
}

function track(id) {
  state.stopFollow?.();
  state.followingId = id;
  state.stopFollow = followJob(id, onJob);
}

function setChip(job) {
  const chip = $("#job-chip");
  const running = job && ["queued", "running"].includes(job.state);
  chip.classList.toggle("running", !!running);
  $("#job-chip-text").textContent = running ? `${job.kind}: ${job.stage} · ${Math.round(job.progress)}%` : "Idle";
}

function onJob(j) {
  state.job = j;
  setChip(j);
  updateJobPanel(j);
  if (state.view === "train") updateTrainLive(j);
  if (j.stage !== state.lastStage) {
    state.lastStage = j.stage;
    refreshStatus().catch(() => {});
  }
  if (j.state === "done" || j.state === "failed") {
    state.followingId = null;
    if (j.state === "done") toast(`${j.kind} finished.`);
    else toast(`${j.kind} failed: ${j.error}`, "error");
    state.results = null;
    refreshStatus().then(() => render()).catch(() => {});
  }
}

function updateJobPanel(j) {
  const badge = $("#job-state");
  badge.textContent = j.state;
  badge.className = `badge ${j.state}`;
  $("#job-bar").style.setProperty("--p", j.progress / 100);
  $("#job-panel .progress").setAttribute("aria-valuenow", Math.round(j.progress));
  $("#job-stage").textContent = j.state === "failed" ? `Failed: ${j.error}` : `${j.kind} · stage: ${j.stage} · ${Math.round(j.progress)}%`;
  const log = $("#job-log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  log.textContent = j.logs.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
}

// ---------- empty states ----------
const FIXES = {
  data: { title: "No dataset yet", text: "Download the Kaggle dataset on the Data view, or generate 200 synthetic demo images to try the pipeline.", label: "Use demo data", run: () => startJob(api.demo, "Demo data") },
  splits: { title: "Dataset not split yet", text: "Create the stratified 70 / 15 / 15 train, validation and test splits.", label: "Make splits", run: () => makeSplits(false) },
  models: { title: "No trained backbones", text: "Train MobileNetV2, EfficientNetB0 and DenseNet121. Smoke mode takes about a minute on CPU.", label: "Train all backbones (smoke)", run: () => startJob(() => api.train(state.status.backbones, 2, true), "Training") },
  ensemble: { title: "Ensemble not fitted", text: "Fit weighted soft voting and the stacked meta-classifier on the validation split.", label: "Fit ensemble", run: () => startJob(api.ensemble, "Ensemble") },
  single: { title: "MobileNetV2 is not trained", text: "Single-model mode predicts with MobileNetV2 alone. Train it, or switch the sidebar setting back to the ensemble.", label: "Train MobileNetV2 (smoke)", run: () => startJob(() => api.train(["mobilenetv2"], 2, true), "Training") },
  metrics: { title: "No evaluation results", text: "Run the three backbones and both ensembles once on the test split.", label: "Run evaluation", run: () => startJob(api.evaluate, "Evaluation") },
};

function firstMissing(s, need) {
  const anyModel = Object.values(s.models).some(Boolean);
  if (need === "predict") {
    const ready = isSingle() ? s.models.mobilenetv2 : anyModel;
    return ready ? null : !s.has_data && !s.splits ? "data" : !s.splits ? "splits" : isSingle() ? "single" : "models";
  }
  const chain = { train: ["data", "splits"], chart: ["data", "splits", "models"], evaluate: ["data", "splits", "models", "ensemble", "metrics"], all: ["data", "splits", "models", "ensemble", "metrics"] }[need];
  const ok = { data: s.has_data || s.splits, splits: s.splits, models: anyModel, ensemble: s.voting && s.stacking, metrics: s.metrics };
  return chain.find((k) => !ok[k]) || null;
}

function emptyState(container, need) {
  const missing = firstMissing(state.status, need);
  if (!missing) {
    container.innerHTML = "";
    return false;
  }
  const f = FIXES[missing];
  container.innerHTML = `
    <div class="empty">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5M12 16v.01"/></svg>
      <div>
        <h3>${esc(f.title)}</h3>
        <p>${esc(f.text)}</p>
        <div class="actions"><button class="btn btn-primary" type="button" data-job-button>${esc(f.label)}</button>
        ${missing === "data" ? '<a class="btn" href="#/data">Go to Data</a>' : ""}</div>
      </div>
    </div>`;
  const b = $("button", container);
  b.disabled = jobActive();
  b.addEventListener("click", f.run);
  return true;
}

// ---------- overview ----------
async function renderOverview() {
  await refreshStatus();
  emptyState($("#overview-fix"), "all");
  if (state.job) updateJobPanel(state.job);
}

// ---------- data ----------
async function makeSplits(smoke) {
  const msg = $("#splits-msg");
  try {
    const r = await api.splits(smoke);
    setMsg(msg, `Splits written from ${r.source} data.`, "ok");
    toast("Splits written.");
    await refreshStatus();
    render();
  } catch (e) {
    setMsg(msg, e.message, "error");
    toast(e.message, "error");
  }
}

async function renderData() {
  const s = await refreshStatus();
  const creds = $("#creds-badge");
  creds.textContent = s.credentials ? "credentials found" : "no credentials";
  creds.className = `badge ${s.credentials ? "done" : ""}`;
  $("#creds-source").textContent = s.credential_source ? `Using: ${s.credential_source}` : "No credentials found yet.";
  $("#btn-download").dataset.blocked = s.credentials ? "0" : "1";
  $("#btn-download").disabled = !s.credentials || jobActive();
  const data = $("#data-badge");
  data.textContent = s.kaggle_images ? `${s.kaggle_images} Kaggle images` : s.demo_images ? `${s.demo_images} demo images`
    : s.source === "sample" ? "sample images only" : "no data";
  data.className = `badge ${s.has_data ? "done" : ""}`;

  const { samples, source, counts } = await api.samples();
  $("#samples-source").textContent = `source: ${source}`;
  $("#sample-grid").innerHTML = samples.map((x) => `
    <figure><img src="${x.url}" alt="${esc(x.label)} example" loading="lazy" width="160" height="160"><figcaption title="${esc(x.label)}">${esc(short(x.label))}</figcaption></figure>`).join("");

  const canvas = $("#chart-counts");
  if (counts && s.splits) {
    $("#counts-empty").innerHTML = "";
    canvas.parentElement.hidden = false;
    $("#counts-source").textContent = `${s.source} data · ${Object.values(counts.train).reduce((a, b) => a + b, 0)} train images`;
    barChart(canvas, s.class_names.map(short), Object.entries(counts).map(([split, c]) => ({ label: split, data: s.class_names.map((n) => c[n]) })));
  } else {
    canvas.parentElement.hidden = true;
    $("#counts-source").textContent = "";
    emptyState($("#counts-empty"), "train");
  }
}

// ---------- train ----------
function selectedBackbones() {
  return $$("#backbone-checks input:checked").map((i) => i.value);
}

function updateEstimate() {
  const s = state.status;
  const smoke = $("#train-smoke").checked;
  const bbs = selectedBackbones();
  const epochs = Number($("#epochs").value);
  let secs = smoke ? bbs.length * s.estimate.smoke_sec_per_backbone
    : bbs.reduce((a, b) => a + s.estimate.cpu_sec_per_image_epoch[b], 0) * s.train_images * epochs;
  const out = $("#estimate");
  if (!bbs.length) { out.textContent = "select a backbone"; return; }
  if (!smoke && !s.train_images) { out.textContent = "make splits first"; return; }
  out.textContent = secs < 90 ? `≈ ${Math.max(5, Math.round(secs / 5) * 5)} s` : secs < 5400 ? `≈ ${Math.round(secs / 60)} min` : `≈ ${(secs / 3600).toFixed(1)} h (use a GPU)`;
  $("#btn-train").disabled = !bbs.length || jobActive();
}

function buildTrainControls() {
  if (state.backbonesBuilt) return;
  state.backbonesBuilt = true;
  const s = state.status;
  $("#backbone-checks").insertAdjacentHTML("beforeend", s.backbones.map((b) =>
    `<label class="check"><input type="checkbox" value="${b}" checked>${esc(dn(b))}</label>`).join(""));
  $("#epochs").value = s.default_epochs;
  $("#epochs-out").textContent = s.default_epochs;
  $("#backbone-checks").addEventListener("change", updateEstimate);
  $("#epochs").addEventListener("input", () => { $("#epochs-out").textContent = $("#epochs").value; updateEstimate(); });
  $("#train-smoke").addEventListener("change", () => { $("#epochs").disabled = $("#train-smoke").checked; updateEstimate(); });
  $("#epochs").disabled = $("#train-smoke").checked;
  $("#btn-train").addEventListener("click", () =>
    startJob(() => api.train(selectedBackbones(), Number($("#epochs").value), $("#train-smoke").checked), "Training"));
}

function renderBackboneProgress(j) {
  const s = state.status;
  const live = j && ["train", "pipeline"].includes(j.kind) ? j : null;
  $("#backbone-progress").innerHTML = s.backbones.map((b) => {
    const m = live?.models?.[b];
    const inJob = live?.plan?.includes(b);
    const p = m ? m.progress : inJob ? 0 : s.models[b] ? 100 : 0;
    const text = m ? `phase ${m.phase} · epoch ${m.epoch}/${m.epochs_total}` : inJob ? "queued" : s.models[b] ? "trained" : "not trained";
    return `<div><div class="bb-row-head"><strong>${esc(dn(b))}</strong><span>${text}</span></div>
      <div class="progress" role="progressbar" aria-label="${esc(dn(b))} progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(p)}"><div class="bar" style="--p:${p / 100}"></div></div></div>`;
  }).join("");
  const badge = $("#train-state");
  badge.textContent = live ? live.state : "idle";
  badge.className = `badge ${live ? live.state : ""}`;
}

let lastHistoryVersion = -1;
function updateTrainLive(j) {
  renderBackboneProgress(j);
  if (["train", "pipeline"].includes(j.kind) && Object.keys(j.history).length && j.version !== lastHistoryVersion) {
    lastHistoryVersion = j.version;
    historyChart($("#chart-loss"), j.history, "loss", state.status.display_names);
    historyChart($("#chart-acc"), j.history, "accuracy", state.status.display_names);
  }
}

async function renderTrain() {
  await refreshStatus();
  const missing = emptyState($("#train-empty"), "train");
  $("#train-body").hidden = missing;
  if (missing) return;
  buildTrainControls();
  updateEstimate();
  const live = state.job && ["queued", "running"].includes(state.job.state) && ["train", "pipeline"].includes(state.job.kind);
  if (live) {
    lastHistoryVersion = -1;
    updateTrainLive(state.job);
    return;
  }
  renderBackboneProgress(null);
  const r = await getResults();
  historyChart($("#chart-loss"), r.histories, "loss", state.status.display_names);
  historyChart($("#chart-acc"), r.histories, "accuracy", state.status.display_names);
}

// ---------- evaluate ----------
async function getResults() {
  if (!state.results) state.results = await api.results();
  return state.results;
}

function setImg(img, url) {
  img.hidden = !url;
  if (url) img.src = url;
}

function renderCm(r, key) {
  state.cmKey = key;
  $$("#cm-tabs .tab").forEach((t) => {
    const on = t.dataset.key === key;
    t.setAttribute("aria-selected", on);
    t.tabIndex = on ? 0 : -1;
  });
  const cm = r.confusion[key];
  const max = Math.max(1, ...cm.flat());
  const names = r.class_names.map(short);
  $("#cm-panel").setAttribute("aria-labelledby", `tab-${key}`);
  $("#cm-panel").innerHTML = `
    <div class="table-wrap"><table class="cm" aria-label="Confusion matrix for ${esc(dn(key))}: rows are true classes, columns predicted">
      <thead><tr><th></th>${names.map((n) => `<th scope="col">${esc(n)}</th>`).join("")}</tr></thead>
      <tbody>${cm.map((row, i) => `<tr><th scope="row" class="row">${esc(names[i])}</th>${row.map((v) => {
        const a = v / max;
        return `<td class="${a > 0.55 ? "hot" : ""}" style="background: color-mix(in srgb, var(--accent) ${Math.round(a * 100)}%, var(--surface))">${v}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="cm-axis">rows: true class · columns: predicted class</p>`;
}

async function renderEvaluate() {
  await refreshStatus();
  const s = state.status;
  const missing = emptyState($("#eval-empty"), "evaluate");
  const anyModel = Object.values(s.models).some(Boolean);
  $("#eval-toolbar").hidden = !anyModel;
  $("#btn-ensemble").disabled = jobActive();
  $("#btn-evaluate").disabled = jobActive() || !(s.voting && s.stacking);
  $("#downloads").hidden = !s.metrics;
  const r = await getResults();
  $("#eval-body").hidden = !r.available || (missing && !s.metrics);
  if (!r.available) return;

  const best = r.rows.find((x) => x.key === r.best_ensemble);
  const active = r.rows.find((x) => x.selected);
  $("#cmp-note").textContent = r.active_mode === "mobilenetv2"
    ? `single-model mode: Predict uses ${active?.name ?? "MobileNetV2"}` : `Predict uses ${active?.name}`;
  $("#eval-meta").textContent = `Test split: ${r.meta.n_test} images · source: ${r.meta.source}${r.meta.smoke ? " (smoke)" : ""} · generated ${r.meta.generated_at}`;
  const gain = r.gain_pp;
  $("#perf-tiles").innerHTML = [
    ["Accuracy", pct(best.accuracy, 2), best.name],
    ["Precision", pct(best.macro_precision, 2), "macro average"],
    ["Recall", pct(best.macro_recall, 2), "macro average"],
    ["F-measure", pct(best.macro_f1, 2), "macro F1"],
  ].map(([l, v, sub]) => `<div class="kpi"><div class="kpi-label">${l}</div><div class="kpi-value">${v}</div><div class="kpi-sub">${esc(sub)}</div></div>`).join("");
  $("#cm-note").textContent = `test split, ${r.meta.n_test} images`;
  $("#kpis").innerHTML = [
    ["Cohen's kappa", num(best.kappa, 3), "agreement beyond chance"],
    ["ROC-AUC", num(best.roc_auc_ovr, 3), "one vs rest, macro"],
    ["Gain over best single", `${gain >= 0 ? "+" : ""}${gain.toFixed(2)} pp`, `vs ${dn(r.best_backbone)} accuracy`, gain > 0 ? "pos" : gain < 0 ? "neg" : ""],
    ["Test images", String(r.meta.n_test), `source: ${r.meta.source}`],
  ].map(([l, v, sub, cls = ""]) => `<div class="kpi"><div class="kpi-label">${l}</div><div class="kpi-value ${cls}">${v}</div><div class="kpi-sub">${esc(sub)}</div></div>`).join("");

  const cols = [["accuracy", "Accuracy"], ["macro_precision", "Macro P"], ["macro_recall", "Macro R"], ["macro_f1", "Macro F1"], ["weighted_f1", "Weighted F1"], ["kappa", "Kappa"], ["roc_auc_ovr", "ROC-AUC"]];
  // Bold the best value per column, unless every model ties.
  const maxOf = Object.fromEntries(cols.map(([k]) => {
    const vals = r.rows.map((x) => x[k] ?? -1);
    return [k, new Set(vals).size > 1 ? Math.max(...vals) : null];
  }));
  $("#cmp-table").innerHTML = `<thead><tr><th scope="col">Model</th>${cols.map(([, l]) => `<th scope="col" class="num">${l}</th>`).join("")}</tr></thead>
    <tbody>${r.rows.map((x) => `<tr class="${x.highlighted ? "highlight" : ""}"><td>${esc(x.name)} ${x.selected ? '<span class="badge badge-accent">selected</span>' : ""}</td>
      ${cols.map(([k]) => `<td class="num">${x[k] === maxOf[k] && x[k] !== null ? `<strong class="best">${num(x[k])}</strong>` : num(x[k])}</td>`).join("")}</tr>`).join("")}</tbody>`;

  barChart($("#chart-metrics"), r.rows.map((x) => x.name), [
    { label: "accuracy", data: r.rows.map((x) => x.accuracy) },
    { label: "macro-F1", data: r.rows.map((x) => x.macro_f1) },
  ], 1);

  $("#cm-tabs").innerHTML = r.rows.map((x) => `<button class="tab" role="tab" type="button" id="tab-${x.key}" data-key="${x.key}" aria-controls="cm-panel">${esc(x.name)}</button>`).join("");
  renderCm(r, r.rows.some((x) => x.key === state.cmKey) ? state.cmKey : r.best_ensemble);

  setImg($("#roc-img"), r.figures["roc_ensemble.png"]);
  $("#roc-model").textContent = dn(r.best_ensemble);
  setImg($("#gradcam-img"), r.figures["gradcam_examples.png"]);

  const sel = $("#pc-model");
  const keep = sel.value || r.best_ensemble;
  sel.innerHTML = r.rows.map((x) => `<option value="${x.key}">${esc(x.name)}</option>`).join("");
  sel.value = r.per_class[keep] ? keep : r.best_ensemble;
  renderPerClass(r);
  $("#notes").innerHTML = r.notes ? md(r.notes) : '<p class="muted">No target check yet.</p>';
}

function renderPerClass(r) {
  const pc = r.per_class[$("#pc-model").value];
  $("#pc-table").innerHTML = `<thead><tr><th scope="col">Class</th><th scope="col" class="num">Precision</th><th scope="col" class="num">Recall</th><th scope="col" class="num">F1</th><th scope="col" class="num">Support</th></tr></thead>
    <tbody>${Object.entries(pc).map(([c, v]) => `<tr><td>${esc(c)}</td><td class="num">${num(v.precision, 3)}</td><td class="num">${num(v.recall, 3)}</td><td class="num">${num(v["f1-score"], 3)}</td><td class="num">${v.support}</td></tr>`).join("")}</tbody>`;
}

// ---------- predict ----------
function predictError(text) {
  $("#predict-error").textContent = text || "";
}

function selectFile(file) {
  predictError("");
  // A new image invalidates the previous result; never show one image's prediction next to another.
  $("#result-body").hidden = true;
  $("#result-empty").hidden = false;
  const s = state.status;
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (!s.upload_exts.includes(ext)) {
    state.file = null;
    $("#btn-submit").disabled = true;
    return predictError(`Unsupported file type ".${ext}". Use ${s.upload_exts.join(", ")}.`);
  }
  if (file.size > s.max_upload_mb * 1024 * 1024) {
    state.file = null;
    $("#btn-submit").disabled = true;
    return predictError(`File is larger than ${s.max_upload_mb} MB.`);
  }
  state.file = file;
  const prev = $("#preview");
  prev.src = URL.createObjectURL(file);
  prev.hidden = false;
  $("#dz-text").hidden = true;
  $("#btn-submit").disabled = false;
}

async function analyze() {
  if (!state.file) return;
  const btn = $("#btn-submit");
  btn.disabled = true;
  btn.textContent = "Submitting…";
  predictError("");
  try {
    showResult(await api.predict(state.file));
  } catch (e) {
    predictError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Submit";
  }
}

function showResult(r) {
  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;
  $("#result-class").textContent = r.class;
  $("#ring-fg").style.strokeDasharray = RING_C;
  $("#ring-fg").style.strokeDashoffset = RING_C * (1 - r.confidence);
  $("#ring-label").textContent = `${Math.round(100 * r.confidence)}%`;
  $("#mode-badge").textContent = r.mode_label;
  const top = r.class;
  $("#probs").innerHTML = Object.entries(r.probs).map(([c, p]) => `
    <div class="prob ${c === top ? "top" : ""}"><div class="prob-head"><span>${esc(c)}</span><span>${pct(p)}</span></div>
    <div class="progress" role="progressbar" aria-label="${esc(c)} probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(100 * p)}"><div class="bar" style="--p:${p}"></div></div></div>`).join("");
  $("#img-original").src = `data:image/png;base64,${r.original_png_base64}`;
  $("#img-cam").src = `data:image/png;base64,${r.gradcam_png_base64}`;
  $("#img-cam").style.opacity = $("#cam-opacity").value / 100;
  $("#cam-model").textContent = `Grad-CAM from ${r.display_names[r.gradcam_model]}${r.mode === "single" ? "" : " (highest ensemble weight)"}. Slide to compare with the input.`;
  $("#votes").innerHTML = `<thead><tr><th scope="col">Backbone</th><th scope="col">Vote</th><th scope="col" class="num">Confidence</th><th scope="col" class="num">Voting weight</th></tr></thead>
    <tbody>${Object.entries(r.per_model_probs).map(([n, p]) => `<tr><td>${esc(r.display_names[n])}</td><td>${esc(r.votes[n])}</td>
      <td class="num">${pct(Math.max(...Object.values(p)))}</td><td class="num">${r.voting_weights[n] === undefined ? "–" : num(r.voting_weights[n], 3)}</td></tr>`).join("")}</tbody>`;
}

async function renderPredict() {
  await refreshStatus();
  const missing = emptyState($("#predict-empty"), "predict");
  $("#predict-body").hidden = missing;
  if (missing) return;
  await loadTestImages();
  const strip = $("#sample-strip");
  if (strip.childElementCount) return;
  const { samples } = await api.samples();
  strip.innerHTML = samples.slice(0, 8).map((x, i) => `<button type="button" data-i="${i}" aria-pressed="false" aria-label="Use sample: ${esc(x.label)}" title="${esc(x.label)}"><img src="${x.url}" alt="" width="56" height="56"></button>`).join("");
  strip.addEventListener("click", async (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    const x = samples[Number(b.dataset.i)];
    $$("button", strip).forEach((o) => o.setAttribute("aria-pressed", o === b));
    await useImageUrl(x.url, x.name);
  });
}

async function useImageUrl(url, name) {
  // Until the fetch lands, Submit must not send the previously selected image.
  state.file = null;
  $("#btn-submit").disabled = true;
  $("#preview").hidden = true;
  const blob = await (await fetch(url)).blob();
  selectFile(new File([blob], name, { type: blob.type }));
}

async function loadTestImages() {
  const { images } = await api.testImages();
  const sel = $("#test-image-select");
  const keep = sel.value;
  sel.innerHTML = images.length
    ? `<option value="">Try a test image… (${images.length})</option>` + state.status.class_names.map((c) => {
      const opts = images.filter((x) => x.label === c);
      return opts.length ? `<optgroup label="${esc(c)}">${opts.map((x) => `<option value="${esc(x.url)}" data-name="${esc(x.name)}">${esc(x.name)}</option>`).join("")}</optgroup>` : "";
    }).join("")
    : '<option value="">No test images exported yet</option>';
  sel.disabled = !images.length;
  sel.value = keep;
  $("#btn-export-tests").hidden = images.length > 0 || !state.status.splits;
}

// ---------- chart ----------
async function renderChart() {
  await refreshStatus();
  const r = await getResults();
  const names = state.status.backbones.filter((b) => r.histories[b]?.length);
  $("#chart-body").hidden = !names.length;
  if (!names.length) {
    // Prerequisite missing -> card with the fixing button; otherwise models exist without history CSVs.
    if (!emptyState($("#chart-empty"), "chart")) {
      $("#chart-empty").innerHTML = '<div class="empty"><div><h3>No training history yet</h3><p>Curves appear here after the backbones are trained.</p></div></div>';
    }
    return;
  }
  $("#chart-empty").innerHTML = "";
  if (!names.includes(state.chartKey)) state.chartKey = names[0];
  $("#chart-tabs").innerHTML = names.map((n) => `<button class="tab" role="tab" type="button" data-key="${n}" aria-selected="${n === state.chartKey}" tabindex="${n === state.chartKey ? 0 : -1}">${esc(dn(n))}</button>`).join("");
  const rows = r.histories[state.chartKey];
  const bestRow = rows.reduce((a, b) => (b.val_loss < a.val_loss ? b : a));
  $("#chart-meta").textContent = `${rows.length} epochs · best val loss ${num(bestRow.val_loss, 4)} at epoch ${bestRow.epoch} (val acc ${pct(bestRow.val_accuracy)})`;
  trainValChart($("#chart-model-loss"), rows, "loss");
  trainValChart($("#chart-model-acc"), rows, "accuracy");
  const phase2 = rows.findIndex((x, i) => i > 0 && x.learning_rate < rows[0].learning_rate);
  $("#chart-phase-note").textContent = phase2 > 0
    ? `Phase 1 (frozen base) ran epochs 1–${phase2}; phase 2 (top 30 % unfrozen, lower learning rate) from epoch ${phase2 + 1}.` : "";
}

// ---------- login ----------
function isAuthed() {
  try { return sessionStorage.getItem(AUTH_KEY) === "1"; } catch { return false; }
}

function showLogin() {
  state.stopFollow?.();
  state.followingId = null;
  $("#app").hidden = true;
  $("#login-screen").hidden = false;
  $("#login-pass").value = "";
  $("#login-user").focus();
}

function startApp() {
  $("#login-screen").hidden = true;
  $("#app").hidden = false;
  if (!state.started) {
    state.started = true;
    refreshStatus().then(route).catch((e) => { toast(e.message, "error"); route(); });
  } else {
    refreshStatus().then(route).catch((e) => toast(e.message, "error"));
  }
}

function setPipelineOpen(open) {
  $("#pipeline-toggle").setAttribute("aria-expanded", open);
  $("#pipeline-sub").hidden = !open;
}

// ---------- about ----------
async function renderAbout() {
  if (state.aboutLoaded) return;
  const a = await api.about();
  $("#mapping").innerHTML = `<thead><tr>${a.mapping.header.map((h) => `<th scope="col">${inline(h)}</th>`).join("")}</tr></thead>
    <tbody>${a.mapping.rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  state.aboutLoaded = true;
}

// ---------- chrome: theme, menu ----------
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  $("#theme-toggle").setAttribute("aria-pressed", t === "dark");
  $(".theme-label").textContent = t === "dark" ? "Light mode" : "Dark mode";
  try { localStorage.setItem("theme", t); } catch { /* storage blocked */ }
}

function openMenu() {
  $("#sidebar").classList.add("open");
  $("#scrim").hidden = false;
  $("#menu-btn").setAttribute("aria-expanded", "true");
  $("#menu-btn").setAttribute("aria-label", "Close navigation");
  $(".nav a")?.focus();
}

function closeMenu() {
  $("#sidebar").classList.remove("open");
  $("#scrim").hidden = true;
  $("#menu-btn").setAttribute("aria-expanded", "false");
  $("#menu-btn").setAttribute("aria-label", "Open navigation");
}

function wire() {
  window.addEventListener("hashchange", route);
  $("#login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#login-error");
    err.textContent = "";
    try {
      await api.login($("#login-user").value.trim(), $("#login-pass").value);
      try { sessionStorage.setItem(AUTH_KEY, "1"); } catch { /* storage blocked: session lasts this page only */ }
      toast("Login successful");
      if (location.hash !== "#/home") history.replaceState(null, "", "#/home");
      startApp();
    } catch (ex) {
      err.textContent = ex.message;
      $("#login-pass").select();
    }
  });
  $("#logout").addEventListener("click", () => {
    try { sessionStorage.removeItem(AUTH_KEY); } catch { /* ignore */ }
    closeMenu();
    toast("Logged out.");
    showLogin();
  });
  $("#pipeline-toggle").addEventListener("click", () =>
    setPipelineOpen($("#pipeline-toggle").getAttribute("aria-expanded") !== "true"));
  $("#chart-tabs").addEventListener("click", (e) => {
    const t = e.target.closest(".tab");
    if (t) { state.chartKey = t.dataset.key; renderChart(); }
  });
  $("#btn-choose").addEventListener("click", () => $("#file-input").click());
  $("#test-image-select").addEventListener("change", (e) => {
    const opt = e.target.selectedOptions[0];
    if (opt?.value) useImageUrl(opt.value, opt.dataset.name);
  });
  $("#btn-export-tests").addEventListener("click", async () => {
    try {
      const r = await api.exportTestImages();
      toast(`Exported ${r.count} test images to test_images/.`);
      await loadTestImages();
    } catch (ex) {
      toast(ex.message, "error");
    }
  });
  $("#single-mode").addEventListener("change", changeActiveMode);
  $("#theme-toggle").addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    render();
  });
  setTheme(document.documentElement.dataset.theme || "light");
  $("#menu-btn").addEventListener("click", () => ($("#sidebar").classList.contains("open") ? closeMenu() : openMenu()));
  $("#scrim").addEventListener("click", closeMenu);
  // Tapping the current page's link fires no hashchange, so close the drawer here too.
  $$("[data-nav]").forEach((a) => a.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

  // overview
  $("#run-all").dataset.jobButton = "";
  $("#run-all").addEventListener("click", () => startJob(() => api.pipelineAll($("#all-smoke").checked), "Full pipeline"));
  const allHint = () => { $("#all-hint").hidden = $("#all-smoke").checked; };
  $("#all-smoke").addEventListener("change", allHint);
  allHint();

  // data
  $("#kg-reveal").addEventListener("click", () => {
    const k = $("#kg-key");
    const show = k.type === "password";
    k.type = show ? "text" : "password";
    $("#kg-reveal").setAttribute("aria-pressed", show);
    $("#kg-reveal").setAttribute("aria-label", show ? "Hide API key" : "Show API key");
  });
  $("#kg-token-reveal").addEventListener("click", () => {
    const k = $("#kg-token");
    const show = k.type === "password";
    k.type = show ? "text" : "password";
    $("#kg-token-reveal").setAttribute("aria-pressed", show);
    $("#kg-token-reveal").setAttribute("aria-label", show ? "Hide access token" : "Show access token");
  });
  $("#token-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const msg = $("#token-msg");
    const token = $("#kg-token").value.trim();
    if (!token) return setMsg(msg, "Paste the access token first.", "error");
    try {
      const r = await api.saveToken(token);
      setMsg(msg, `Saved to ${r.saved}`, "ok");
      $("#kg-token").value = "";
      render();
    } catch (err) {
      setMsg(msg, err.message, "error");
    }
  });
  $("#creds-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const msg = $("#creds-msg");
    const user = $("#kg-user").value.trim();
    const key = $("#kg-key").value.trim();
    if (!user || !key) return setMsg(msg, "Enter both the username and the API key.", "error");
    try {
      const r = await api.saveCreds(user, key);
      setMsg(msg, `Saved to ${r.saved}`, "ok");
      $("#kg-key").value = "";
      render();
    } catch (err) {
      setMsg(msg, err.message, "error");
    }
  });
  for (const [id, call, label] of [["#btn-download", api.download, "Download"], ["#btn-demo", api.demo, "Demo data"]]) {
    $(id).dataset.jobButton = "";
    $(id).addEventListener("click", () => startJob(call, label));
  }
  $("#btn-splits").dataset.jobButton = "";
  $("#btn-splits").addEventListener("click", () => makeSplits($("#split-smoke").checked));

  // train / evaluate
  $("#btn-train").dataset.jobButton = "";
  $("#btn-ensemble").addEventListener("click", () => startJob(api.ensemble, "Ensemble"));
  $("#btn-evaluate").addEventListener("click", () => startJob(api.evaluate, "Evaluation"));
  $("#pc-model").addEventListener("change", async () => renderPerClass(await getResults()));
  $("#cm-tabs").addEventListener("click", async (e) => {
    const t = e.target.closest(".tab");
    if (t) renderCm(await getResults(), t.dataset.key);
  });
  $("#cm-tabs").addEventListener("keydown", async (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const tabs = $$("#cm-tabs .tab");
    const i = tabs.findIndex((t) => t.dataset.key === state.cmKey);
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    renderCm(await getResults(), next.dataset.key);
    next.focus();
  });

  // predict
  const dz = $("#dropzone");
  $("#file-input").addEventListener("change", (e) => e.target.files[0] && selectFile(e.target.files[0]));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
  });
  $("#btn-submit").addEventListener("click", analyze);
  $("#cam-opacity").addEventListener("input", (e) => {
    $("#img-cam").style.opacity = e.target.value / 100;
    $("#cam-out").textContent = `${e.target.value}%`;
  });
}

wire();
if (isAuthed()) startApp();
else showLogin();
