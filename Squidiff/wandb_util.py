"""
Minimal wandb helpers shared across training code.
"""

import json
import os

try:
    import wandb
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None

from . import dist_util


_LOCAL_RUN_DIR = None
_LOCAL_METRICS_PATH = None
_LOCAL_SUMMARY_PATH = None
_LOCAL_METRIC_PLOTS = {
    "training_loss": "training_loss.png",
    "training_mse": "training_mse.png",
    "val_loss": "val_loss.png",
    "val_mse": "val_mse.png",
}


def is_available():
    return wandb is not None


def _default_run_name(args):
    if args.get("run_name"):
        return args["run_name"]

    checkpoint_dir = args.get("resume_checkpoint", "")
    if checkpoint_dir:
        return os.path.basename(os.path.abspath(checkpoint_dir))

    return None


def _default_wandb_dir(args):
    return args.get("wandb_dir") or "wandb/"


def _default_logger_dir(args):
    return args.get("logger_path") or "logger/"


def _local_run_dir(args):
    run_name = _default_run_name(args) or "run"
    return os.path.join(_default_logger_dir(args), run_name)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _init_local_run(args):
    global _LOCAL_RUN_DIR, _LOCAL_METRICS_PATH, _LOCAL_SUMMARY_PATH

    _LOCAL_RUN_DIR = _local_run_dir(args)
    os.makedirs(_LOCAL_RUN_DIR, exist_ok=True)
    _LOCAL_METRICS_PATH = os.path.join(_LOCAL_RUN_DIR, "metrics.jsonl")
    _LOCAL_SUMMARY_PATH = os.path.join(_LOCAL_RUN_DIR, "summary.json")

    _write_json(os.path.join(_LOCAL_RUN_DIR, "config.json"), args)
    if not os.path.exists(_LOCAL_SUMMARY_PATH):
        _write_json(_LOCAL_SUMMARY_PATH, {})


def _append_local_metrics(metrics, step=None):
    if _LOCAL_METRICS_PATH is None:
        return
    payload = dict(metrics)
    if step is not None:
        payload["_step"] = step
    with open(_LOCAL_METRICS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True))
        handle.write("\n")


def _update_local_summary(summary):
    if _LOCAL_SUMMARY_PATH is None:
        return
    payload = _read_json(_LOCAL_SUMMARY_PATH)
    payload.update(_json_safe(summary))
    _write_json(_LOCAL_SUMMARY_PATH, payload)


def _plot_local_metric(metric_name, filename):
    if _LOCAL_RUN_DIR is None or _LOCAL_METRICS_PATH is None or not os.path.exists(_LOCAL_METRICS_PATH):
        return
    if plt is None:
        print(f"Local logger: matplotlib not available, skipping {filename} generation.")
        return

    xs = []
    ys = []
    with open(_LOCAL_METRICS_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if metric_name not in payload:
                continue
            xs.append(payload.get("_step", len(xs) + 1))
            ys.append(payload[metric_name])

    if not ys:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(xs, ys, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_LOCAL_RUN_DIR, filename))
    plt.close(fig)


def init_run(args):
    if not dist_util.is_main_process():
        return

    if not is_available():
        _init_local_run(args)
        return

    os.makedirs(_default_wandb_dir(args), exist_ok=True)
    init_kwargs = {
        "project": args.get("project_name", "Squidiff"),
        "name": _default_run_name(args),
        "dir": _default_wandb_dir(args),
        "config": args,
    }
    wandb.init(**init_kwargs)


def log(metrics, *, step=None):
    if is_available() and dist_util.is_main_process() and wandb.run is not None:
        wandb.log(metrics, step=step)
    elif dist_util.is_main_process():
        _append_local_metrics(metrics, step=step)


def update_summary(summary):
    if is_available() and dist_util.is_main_process() and wandb.run is not None:
        wandb.run.summary.update(summary)
    elif dist_util.is_main_process():
        _update_local_summary(summary)


def finish():
    if is_available() and dist_util.is_main_process() and wandb.run is not None:
        wandb.finish()
    elif dist_util.is_main_process():
        for metric_name, filename in _LOCAL_METRIC_PLOTS.items():
            _plot_local_metric(metric_name, filename)


def log_figure(name, fig, *, step=None, filename=None):
    if not dist_util.is_main_process():
        return

    if is_available() and wandb.run is not None:
        wandb.log({name: wandb.Image(fig)}, step=step)
        return

    if _LOCAL_RUN_DIR is None:
        return

    figure_filename = filename or f"{name}.png"
    figure_path = os.path.join(_LOCAL_RUN_DIR, figure_filename)
    fig.savefig(figure_path, bbox_inches="tight")
