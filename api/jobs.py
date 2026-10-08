"""Tiny in-process job runner: one background thread at a time, fed by src's progress_cb."""
import logging
import threading
import time
import uuid

from src import config as C

log = logging.getLogger("api.jobs")
MAX_LOG_LINES = 500
# Steps per stage when the callback carries no epochs_total.
KNOWN_STEPS = {"download": 1, "ensemble": 1, "evaluate": 2}


class Busy(Exception):
    pass


def friendly(e):
    if isinstance(e, (FileNotFoundError, ValueError)):
        return str(e)
    return f"{type(e).__name__}: {e}"


class Job:
    def __init__(self, kind, plan):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.plan = plan  # [(stage, weight)] in execution order
        self.state = "queued"
        self.stage = plan[0][0] if plan else kind
        self.progress = 0.0
        self.logs = []
        self.error = None
        self.result = None
        self.started_at = time.time()
        self.finished_at = None
        self.models = {}   # backbone -> {epoch, epochs_total, phase, progress}
        self.history = {}  # backbone -> [{epoch, loss, val_loss, accuracy, val_accuracy}]
        self.version = 0   # bumps on every change (SSE uses it)
        self._lock = threading.Lock()

    def log(self, line):
        with self._lock:
            self.logs.append(f"{time.strftime('%H:%M:%S')}  {line}")
            del self.logs[:-MAX_LOG_LINES]
            self.version += 1

    def begin_stage(self, stage):
        with self._lock:
            self.stage = stage
            self._set_progress(stage, 0.0)
            self.version += 1
        self.log(f"stage started: {stage}")

    def _set_progress(self, stage, frac):
        names = [s for s, _ in self.plan]
        if stage not in names:
            return
        total = sum(w for _, w in self.plan) or 1
        i = names.index(stage)
        done = sum(w for _, w in self.plan[:i])
        self.progress = max(self.progress, 100 * (done + self.plan[i][1] * min(frac, 1.0)) / total)

    def callback(self, stage, epoch, logs):
        """progress_cb(stage, epoch, logs) as used throughout src/."""
        logs = dict(logs or {})
        total = logs.get("epochs_total")
        frac = epoch / total if total else epoch / KNOWN_STEPS.get(stage, 1)
        with self._lock:
            self.stage = stage
            if stage in C.BACKBONES and total:
                self.models[stage] = {"epoch": epoch, "epochs_total": total, "phase": logs.get("phase"),
                                      "progress": round(100 * frac, 1)}
                self.history.setdefault(stage, []).append(
                    {"epoch": epoch, **{k: logs.get(k) for k in ("loss", "val_loss", "accuracy", "val_accuracy")}})
            self._set_progress(stage, frac)
            self.version += 1
        shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in logs.items()
                 if not isinstance(v, dict)}
        self.log(f"[{stage}] step {epoch} {shown}")

    def to_dict(self):
        with self._lock:
            return {"id": self.id, "kind": self.kind, "state": self.state, "stage": self.stage,
                    "plan": [s for s, _ in self.plan],
                    "progress": round(self.progress, 1), "logs": list(self.logs), "error": self.error,
                    "result": self.result, "started_at": self.started_at, "finished_at": self.finished_at,
                    "models": dict(self.models), "history": {k: list(v) for k, v in self.history.items()},
                    "version": self.version}


class JobRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self.jobs = {}
        self.active = None

    def active_id(self):
        a = self.active
        return a.id if a and a.state in ("queued", "running") else None

    def get(self, job_id):
        return self.jobs.get(job_id)

    def start(self, kind, steps):
        """steps: [(stage, weight, fn(job))]. Raises Busy if a job is already running."""
        with self._lock:
            if self.active_id():
                raise Busy(f"Job {self.active.id} ({self.active.kind}) is still running. Wait for it to finish.")
            job = Job(kind, [(s, w) for s, w, _ in steps])
            self.jobs[job.id] = job
            self.active = job
        threading.Thread(target=self._run, args=(job, steps), daemon=True, name=f"job-{job.id}").start()
        return job

    def _run(self, job, steps):
        job.state = "running"
        job.log(f"job {job.kind} started")
        try:
            out = []
            for stage, _, fn in steps:
                job.begin_stage(stage)
                r = fn(job)
                if r:
                    out.append(str(r))
                    job.log(str(r).splitlines()[0])
            job.result = "\n\n".join(out)
            job.progress = 100.0
            job.state = "done"
            job.log("job finished")
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job.id)
            job.error = friendly(e)
            job.state = "failed"
            job.log(f"job failed: {job.error}")
        finally:
            job.finished_at = time.time()
            job.version += 1
