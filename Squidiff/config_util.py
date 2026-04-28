from copy import deepcopy
import importlib.util
from pathlib import Path
import os
from datetime import datetime
from Squidiff.script_util import (
    model_and_diffusion_defaults,
)

import yaml

VALID_LOSS_TYPES = {"mse", "mse-kl", "mse-gmm", "kl", "rescaled-mse"}

_ALLOWED_TOP_LEVEL_SECTIONS = {
    "experiment",
    "dataset",
    "training",
    "runtime",
    "model",
    "diffusion",
    "conditioning",
    "logging",
}

_ALLOWED_SECTION_KEYS = {
    "experiment": {"project_name", "run_name", "seed"},
    "dataset": {"data_path", "val_data_path", "rna_only", "control_data_path"},
    "training": {
        "batch_size",
        "microbatch",
        "lr",
        "weight_decay",
        "lr_anneal_steps",
        "ema_rate",
        "log_interval",
        "save_interval",
        "resume_checkpoint",
    },
    "runtime": {"use_fp16", "fp16_scale_growth", "schedule_sampler", "use_ddim"},
    "model": {
        "gene_size",
        "num_layers",
        "num_channels",
        "dropout",
        "class_cond",
        "use_checkpoint",
        "use_scale_shift_norm",
        "use_encoder",
        "loss_type",
        "alpha",
        "gmm_num_components",
    },
    "diffusion": {
        "learn_sigma",
        "diffusion_steps",
        "noise_schedule",
        "timestep_respacing",
        "predict_xstart",
        "rescale_timesteps",
    },
    "conditioning": {
        "use_drug_structure",
        "comb_num",
        "drug_dimension",
        "paired_latent_dim",
        "hidden_rna",
        "hidden_atac",
        "paired_dropout",
    },
    "logging": {"wandb_dir", "logger_path", "val_recon_interval_epochs"},
}


def training_defaults():
    defaults = {}
    defaults.update(model_and_diffusion_defaults())
    defaults.update(
        {
            "data_path": "",
            "val_data_path": "",
            "rna_only": None,
            "control_data_path": "",
            "schedule_sampler": "uniform",
            "lr": 1e-4,
            "weight_decay": 0.0,
            "lr_anneal_steps": 1e5,
            "batch_size": 128,
            "microbatch": -1,
            "ema_rate": "0.9999",
            "log_interval": 1e4,
            "save_interval": 1e4,
            "resume_checkpoint": "",
            "use_fp16": False,
            "fp16_scale_growth": 1e-3,
            "gene_size": None,
            "num_layers": 3,
            "class_cond": False,
            "use_encoder": True,
            "diffusion_steps": 1000,
            "logger_path": "",
            "wandb_dir": "",
            "val_recon_interval_epochs": 50,
            "use_drug_structure": False,
            "paired_latent_dim": 128,
            "hidden_rna": "1024,512",
            "hidden_atac": "1024,512",
            "paired_dropout": 0.2,
            "comb_num": 1,
            "use_ddim": True,
            "seed": 42,
            "project_name": "Squidiff",
            "run_name": "",
        }
    )
    return defaults


def _default_checkpoint_dir():
    return os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))


def _wandb_is_available():
    return importlib.util.find_spec("wandb") is not None


def validate_args(args):
    if not args.get("data_path"):
        raise ValueError("Dataset path is required. Pass --data_path or set dataset.data_path in config.")
    if args.get("rna_only") is None:
        raise ValueError("dataset.rna_only is required. Pass it in config or via --rna_only.")

    if args.get("gene_size") is not None and int(args["gene_size"]) <= 0:
        raise ValueError("--gene_size must be positive when provided.")
    if int(args.get("val_recon_interval_epochs", 0)) < 0:
        raise ValueError("logging.val_recon_interval_epochs must be non-negative.")
    if args.get("loss_type") not in VALID_LOSS_TYPES:
        valid = ", ".join(sorted(VALID_LOSS_TYPES))
        raise ValueError(f"loss_type must be one of: {valid}.")
    if float(args.get("alpha", 0.0)) < 0:
        raise ValueError("alpha must be non-negative.")
    if int(args.get("gmm_num_components", 1)) <= 0:
        raise ValueError("gmm_num_components must be positive.")

    if not args["wandb_dir"] and not args["logger_path"]:
        if _wandb_is_available():
            args["wandb_dir"] = "wandb/"
        else:
            args["logger_path"] = "logger/"
    if not args["resume_checkpoint"]:
        args["resume_checkpoint"] = _default_checkpoint_dir()

    return args


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_base_path(base_ref, source_path):
    base_path = Path(base_ref)
    candidates = []
    if not base_path.is_absolute():
        candidates.append((source_path.parent / base_path).resolve())
    candidates.append(base_path.resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve base config '{base_ref}' referenced from '{source_path}'."
    )


def load_config(config_path, _stack=None):
    config_path = Path(config_path).resolve()
    if _stack is None:
        _stack = []
    if config_path in _stack:
        cycle = " -> ".join(str(path) for path in (*_stack, config_path))
        raise ValueError(f"Circular config inheritance detected: {cycle}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")

    bases = config.pop("__base__", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise TypeError(f"__base__ must be a string or list: {config_path}")

    merged = {}
    for base_ref in bases:
        base_path = _resolve_base_path(base_ref, config_path)
        merged = _deep_merge(merged, load_config(base_path, [*_stack, config_path]))

    merged = _deep_merge(merged, config)
    unknown_sections = sorted(set(merged) - _ALLOWED_TOP_LEVEL_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"Unknown top-level config sections in {config_path}: {', '.join(unknown_sections)}"
        )
    return merged


def _section(config, name):
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    unknown_keys = sorted(set(value) - _ALLOWED_SECTION_KEYS[name])
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in config section '{name}': {', '.join(unknown_keys)}"
        )
    return value


def normalize_training_config(config, defaults):
    args = deepcopy(defaults)

    experiment = _section(config, "experiment")
    dataset = _section(config, "dataset")
    training = _section(config, "training")
    runtime = _section(config, "runtime")
    model = _section(config, "model")
    diffusion = _section(config, "diffusion")
    conditioning = _section(config, "conditioning")
    logging = _section(config, "logging")

    for section in (
        dataset,
        training,
        runtime,
        model,
        diffusion,
        conditioning,
        logging,
    ):
        args.update(section)

    args["project_name"] = experiment.get("project_name", args["project_name"])
    args["run_name"] = experiment.get("run_name", args["run_name"])
    args["seed"] = experiment.get("seed", args["seed"])

    return args
