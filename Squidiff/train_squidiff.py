# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2026-04-19

import argparse
import os
from datetime import datetime

from Squidiff import dist_util, wandb_util
from Squidiff.resample import create_named_schedule_sampler
from Squidiff.scrna_datasets import prepared_data
from Squidiff.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from Squidiff.train_util import TrainLoop


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _arg_type(value):
    if isinstance(value, bool):
        return _str2bool
    return type(value)


def _default_checkpoint_dir():
    return os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))


def run_training(args):
    accelerator = dist_util.setup_accelerator(use_fp16=args["use_fp16"])
    wandb_util.init_run(args)

    accelerator.print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)

    accelerator.print("creating data loader...")
    data = prepared_data(
        data_dir=args["data_path"],
        control_data_dir=args["control_data_path"],
        batch_size=args["batch_size"],
        use_drug_structure=args["use_drug_structure"],
        comb_num=args["comb_num"],
    )

    start_time = datetime.now()
    accelerator.print(f"training started at {start_time.isoformat()}")
    wandb_util.update_summary(
        {
            "train_start": start_time.isoformat(),
            "device": str(dist_util.dev()),
            "num_processes": dist_util.num_processes(),
            "checkpoint_dir": args["resume_checkpoint"],
        }
    )

    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args["batch_size"],
        microbatch=args["microbatch"],
        lr=args["lr"],
        ema_rate=args["ema_rate"],
        log_interval=args["log_interval"],
        save_interval=args["save_interval"],
        resume_checkpoint=args["resume_checkpoint"],
        use_fp16=args["use_fp16"],
        fp16_scale_growth=args["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=args["weight_decay"],
        lr_anneal_steps=args["lr_anneal_steps"],
        use_drug_structure=args["use_drug_structure"],
        comb_num=args["comb_num"],
    )
    train_.run_loop()

    end_time = datetime.now()
    duration_min = (end_time - start_time).total_seconds() / 60
    final_step = train_.step + train_.resume_step
    accelerator.print(
        f"training finished at {end_time.isoformat()} after {duration_min:.2f} minutes"
    )
    wandb_util.log({"train_duration_min": duration_min}, step=final_step)
    wandb_util.update_summary(
        {
            "train_end": end_time.isoformat(),
            "train_duration_min": duration_min,
            "final_step": final_step,
        }
    )
    wandb_util.finish()

    return train_.loss_list


def parse_args():
    """Parse command-line arguments and update with default values."""
    default_args = {}
    default_args.update(model_and_diffusion_defaults())
    updated_args = {
        "data_path": "",
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
        "gene_size": 100,
        "output_dim": 100,
        "num_layers": 3,
        "class_cond": False,
        "use_encoder": True,
        "diffusion_steps": 1000,
        "logger_path": "",
        "wandb_project": "Squidiff",
        "wandb_run_name": "",
        "wandb_entity": "",
        "wandb_mode": "online",
        "wandb_dir": "",
        "use_drug_structure": False,
        "comb_num": 1,
        "use_ddim": True,
    }
    default_args.update(updated_args)

    parser = argparse.ArgumentParser(
        description="Perturbation-conditioned generative diffusion model"
    )
    for key, value in default_args.items():
        parser.add_argument(
            f"--{key}",
            default=value,
            type=_arg_type(value),
            help=f"{key} (default: {value})",
        )

    args = vars(parser.parse_args())
    if args["data_path"] == "":
        raise ValueError(
            "Dataset path is required. Please specify the path where the training adata is."
        )

    if not args["wandb_dir"] and args["logger_path"]:
        args["wandb_dir"] = args["logger_path"]

    if not args["resume_checkpoint"]:
        args["resume_checkpoint"] = _default_checkpoint_dir()

    return args


def main():
    args_train = parse_args()
    run_training(args_train)


if __name__ == "__main__":
    main()
