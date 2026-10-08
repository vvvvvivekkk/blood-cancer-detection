// Thin fetch wrapper. Every error carries the server's readable {detail} message.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch {
    throw new ApiError("Cannot reach the server. Is `python run.py` still running?", 0);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(data.detail || `Request failed (${res.status})`, res.status);
  return data;
}

export const api = {
  status: () => request("GET", "/api/status"),
  results: () => request("GET", "/api/results"),
  samples: () => request("GET", "/api/data/samples"),
  about: () => request("GET", "/api/about"),
  job: (id) => request("GET", `/api/jobs/${id}`),
  saveToken: (token) => request("POST", "/api/data/kaggle-token", { token }),
  saveCreds: (username, key) => request("POST", "/api/data/kaggle-creds", { username, key }),
  download: () => request("POST", "/api/data/download"),
  demo: () => request("POST", "/api/data/demo"),
  splits: (smoke) => request("POST", "/api/data/splits", { smoke }),
  train: (backbones, epochs, smoke) => request("POST", "/api/train", { backbones, epochs, smoke }),
  ensemble: () => request("POST", "/api/ensemble"),
  evaluate: () => request("POST", "/api/evaluate"),
  setActiveMode: (active_mode) => request("POST", "/api/settings", { active_mode }),
  pipelineAll: (smoke) => request("POST", "/api/pipeline/all", { smoke }),
  predict: (file) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    return request("POST", "/api/predict", fd);
  },
};

/** Follow a job: Server-Sent Events first, 1 s polling if SSE fails. onUpdate(job) on every change. */
export function followJob(id, onUpdate) {
  let stopped = false;
  let timer = null;
  const finished = (j) => j.state === "done" || j.state === "failed";

  const poll = async () => {
    if (stopped) return;
    try {
      const j = await api.job(id);
      onUpdate(j);
      if (finished(j)) return;
    } catch (e) {
      if (e.status === 404) return;
    }
    timer = setTimeout(poll, 1000);
  };

  if ("EventSource" in window) {
    const es = new EventSource(`/api/jobs/${id}/stream`);
    es.onmessage = (ev) => {
      const j = JSON.parse(ev.data);
      onUpdate(j);
      if (finished(j)) { es.close(); stopped = true; }
    };
    es.onerror = () => {
      es.close();
      if (!stopped) poll();
    };
  } else {
    poll();
  }
  return () => { stopped = true; clearTimeout(timer); };
}
