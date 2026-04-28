# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2026-04-19

import argparse
from datetime import datetime

from Squidiff.config_util import (
    load_config, 
    normalize_training_config, 
    training_defaults, 
    validate_args
)
from Squidiff import dist_util, wandb_util
from Squidiff.model_spec import build_model_spec
from Squidiff.resample import create_named_schedule_sampler
from Squidiff.scrna_datasets import prepared_data
from Squidiff.seed_util import seed_everything
from Squidiff.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from Squidiff.train_util import TrainLoop

def run_training(args):
    seed_everything(args["seed"])
    accelerator = dist_util.setup_accelerator(use_fp16=args["use_fp16"])

    accelerator.print("creating data loader...")
    data, resolved_spec = prepared_data(
        data_path=args["data_path"],
        control_data_dir=args["control_data_path"],
        batch_size=args["batch_size"],
        use_drug_structure=args["use_drug_structure"],
        comb_num=args["comb_num"],
        rna_only=args["rna_only"],
        gene_size=args.get("gene_size"),
        shuffle=True,
    )
    val_data = None
    if args.get("val_data_path"):
        accelerator.print("creating validation data loader...")
        val_data, _ = prepared_data(
            data_path=args["val_data_path"],
            control_data_dir=args["control_data_path"],
            batch_size=args["batch_size"],
            use_drug_structure=args["use_drug_structure"],
            comb_num=args["comb_num"],
            rna_only=args["rna_only"],
            gene_size=None,
            rna_feature_names=resolved_spec.rna_feature_names,
            atac_feature_names=resolved_spec.atac_feature_names,
            shuffle=False,
        )
    args["gene_size"] = resolved_spec.rna_dim
    args["atac_input_size"] = resolved_spec.atac_dim
    args["rna_feature_names"] = resolved_spec.rna_feature_names
    args["rna_features_file"] = "rna_features.txt"
    args["atac_feature_names"] = resolved_spec.atac_feature_names
    args["atac_features_file"] = "atac_features.txt"
    args["model_spec"] = build_model_spec(args)
    wandb_util.init_run(args)

    accelerator.print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)

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
        model_spec=args["model_spec"],
        rna_feature_names=resolved_spec.rna_feature_names,
        rna_features_file=args["rna_features_file"],
        atac_feature_names=resolved_spec.atac_feature_names,
        atac_features_file=args["atac_features_file"],
        val_data=val_data,
        use_ddim=args["use_ddim"],
        val_recon_interval_epochs=args["val_recon_interval_epochs"],
    )
    train_.run_loop()

    end_time = datetime.now()
    duration_min = (end_time - start_time).total_seconds() / 60
    final_step = train_.step + train_.resume_step
    accelerator.print(
        f"training finished at {end_time.isoformat()} after {duration_min:.2f} minutes"
    )
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
    """Parse command-line arguments from YAML plus a few optional overrides."""
    parser = argparse.ArgumentParser(
        description="Perturbation-conditioned generative diffusion model"
    )
    parser.add_argument("--config", required=True, help="Path to YAML training config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--data_path", default=None, help="Optional dataset path override.")
    parser.add_argument("--val_data_path", default=None, help="Optional validation dataset path override.")
    parser.add_argument(
        "--rna_only",
        default=None,
        choices=["true", "false", "True", "False"],
        help="Required modality mode override.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Optional checkpoint directory override.",
    )

    cli_args = parser.parse_args()
    config = load_config(cli_args.config)
    args = normalize_training_config(config, training_defaults())

    overrides = {
        "seed": cli_args.seed,
        "data_path": cli_args.data_path,
        "val_data_path": cli_args.val_data_path,
        "rna_only": None if cli_args.rna_only is None else cli_args.rna_only.lower() == "true",
        "resume_checkpoint": cli_args.resume_checkpoint,
    }
    for key, value in overrides.items():
        if value is not None:
            args[key] = value

    return validate_args(args)


def main():
    args_train = parse_args()
    run_training(args_train)


if __name__ == "__main__":
    main()
