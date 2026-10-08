// Chart.js helpers. Colours come from the CSS variables so charts follow the theme.
// If the CDN script did not load, a readable note replaces the canvas instead of an error.

const charts = new Map();

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Series colours: the accent first, then fixed hues readable on both themes.
function palette() {
  return [css("--accent"), "#c2410c", "#7c3aed", "#15803d", "#b45309"];
}

function missing(canvas, text) {
  const box = canvas.parentElement;
  canvas.hidden = true;
  let note = box.querySelector(".chart-missing");
  if (!note) {
    note = document.createElement("div");
    note.className = "chart-missing";
    box.appendChild(note);
  }
  note.textContent = text;
}

function draw(canvas, config) {
  charts.get(canvas.id)?.destroy();
  charts.delete(canvas.id);
  if (!window.Chart) {
    missing(canvas, "Charts unavailable: Chart.js could not be loaded from the CDN.");
    return;
  }
  canvas.hidden = false;
  canvas.parentElement.querySelector(".chart-missing")?.remove();
  const text = css("--text");
  const grid = css("--border");
  const muted = css("--muted");
  config.options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    plugins: { legend: { labels: { color: text, boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: "circle" } } },
    scales: {
      x: { ticks: { color: muted }, grid: { color: grid }, title: { display: !!config.xTitle, text: config.xTitle, color: muted } },
      y: { ticks: { color: muted }, grid: { color: grid }, beginAtZero: true, suggestedMax: config.yMax },
    },
    ...config.options,
  };
  charts.set(canvas.id, new window.Chart(canvas, config));
}

export function barChart(canvas, labels, series, yMax) {
  const colours = palette();
  draw(canvas, {
    type: "bar",
    yMax,
    data: {
      labels,
      datasets: series.map((s, i) => ({ label: s.label, data: s.data, backgroundColor: colours[i % colours.length], borderRadius: 4 })),
    },
  });
}

/** histories: {name: [{epoch, loss, val_loss, accuracy, val_accuracy}]}; metric: "loss" | "accuracy". */
export function historyChart(canvas, histories, metric, displayNames) {
  const colours = palette();
  const names = Object.keys(histories).filter((n) => histories[n]?.length);
  if (!names.length) {
    charts.get(canvas.id)?.destroy();
    charts.delete(canvas.id);
    missing(canvas, "No training history yet. Curves appear here during and after training.");
    return;
  }
  const maxEpoch = Math.max(...names.flatMap((n) => histories[n].map((r) => r.epoch)));
  const labels = Array.from({ length: maxEpoch }, (_, i) => i + 1);
  const datasets = names.flatMap((n, i) => {
    const c = colours[i % colours.length];
    const at = (key) => labels.map((e) => histories[n].find((r) => r.epoch === e)?.[key] ?? null);
    return [
      { label: `${displayNames[n] || n} val`, data: at(`val_${metric}`), borderColor: c, backgroundColor: c, tension: 0.25, spanGaps: true, pointRadius: 3 },
      { label: `${displayNames[n] || n} train`, data: at(metric), borderColor: c, backgroundColor: c, borderDash: [5, 4], tension: 0.25, spanGaps: true, pointRadius: 2 },
    ];
  });
  draw(canvas, { type: "line", xTitle: "epoch", yMax: metric === "accuracy" ? 1 : undefined, data: { labels, datasets } });
}
