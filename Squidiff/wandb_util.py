"""
Minimal wandb helpers shared across training code.
"""

import os

import wandb

from . import dist_util


def _default_run_name(args):
    if args.get("wandb_run_name"):
        return args["wandb_run_name"]

    checkpoint_dir = args.get("resume_checkpoint", "")
    if checkpoint_dir:
        return os.path.basename(os.path.abspath(checkpoint_dir))

    return None


def _default_wandb_dir(args):
    return (
        args.get("wandb_dir")
        or args.get("logger_path")
        or args.get("resume_checkpoint")
        or os.getcwd()
    )


def init_run(args):
    if not dist_util.is_main_process():
        return

    os.makedirs(_default_wandb_dir(args), exist_ok=True)
    wandb.init(
        project=args.get("wandb_project", "Squidiff"),
        entity=args.get("wandb_entity") or None,
        name=_default_run_name(args),
        mode=args.get("wandb_mode", "online"),
        dir=_default_wandb_dir(args),
        config=args,
    )


def log(metrics, *, step=None):
    if dist_util.is_main_process() and wandb.run is not None:
        wandb.log(metrics, step=step)


def update_summary(summary):
    if dist_util.is_main_process() and wandb.run is not None:
        wandb.run.summary.update(summary)


def finish():
    if dist_util.is_main_process() and wandb.run is not None:
        wandb.finish()
