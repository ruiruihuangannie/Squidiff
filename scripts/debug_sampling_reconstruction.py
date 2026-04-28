from pathlib import Path

import pandas as pd

from debug_mouse_common import (
    align_rna,
    common_parser,
    encode_adata,
    expression_summary,
    load_data,
    load_sampler,
    mean_metric_rows,
    maybe_subsample_by_group,
    prepare_run,
    sample_from_z,
    scatter_pred_truth,
    subset_by_group,
    torch_z,
)


def existing_checkpoint_variants(model_path):
    directory = Path(model_path).parent
    names = ["best_model.pt", "best_model_0.9999.pt", "model.pt", "model_0.9999.pt"]
    return [directory / name for name in names if (directory / name).exists()]


def main():
    parser = common_parser("Compare reconstruction across checkpoint and sampler variants.")
    parser.add_argument("--replicates", type=int, default=3)
    args = prepare_run(parser.parse_args())

    rows = []
    scale_rows = []
    for checkpoint in existing_checkpoint_variants(args.model_path):
        sampler = load_sampler(checkpoint, args.seed)
        train = maybe_subsample_by_group(align_rna(sampler, load_data(args.train_path)), args.max_cells_per_group, seed=args.seed)
        test = maybe_subsample_by_group(align_rna(sampler, load_data(args.test_path)), args.max_cells_per_group, seed=args.seed)
        device = next(sampler.model.parameters()).device

        for split, adata in [("train", train), ("test", test)]:
            for day in [0, 3]:
                truth = subset_by_group(adata, day)
                z = encode_adata(sampler, truth, batch_size=args.batch_size, mode="eval")
                for use_ddim in [True, False]:
                    for rep in range(args.replicates):
                        name = f"{checkpoint.name}_{split}_day{day}_{'ddim' if use_ddim else 'ancestral'}_rep{rep}"
                        pred = sample_from_z(sampler, torch_z(z, device), seed=args.seed + rep, use_ddim=use_ddim)
                        metric = mean_metric_rows(name, pred, truth)
                        metric.update(
                            {
                                "checkpoint": checkpoint.name,
                                "split": split,
                                "day": day,
                                "sampler": "ddim" if use_ddim else "ancestral",
                                "replicate": rep,
                            }
                        )
                        rows.append(metric)
                        scale = expression_summary(name, pred)
                        scale.update({"checkpoint": checkpoint.name, "split": split, "day": day, "sampler": "ddim" if use_ddim else "ancestral", "replicate": rep})
                        scale_rows.append(scale)
                        if rep == 0:
                            scatter_pred_truth(pred, truth, name, args.out_dir / f"{name}_scatter.png")

    pd.DataFrame(rows).to_csv(args.out_dir / "sampling_reconstruction_metrics.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.out_dir / "sampling_reconstruction_scale_metrics.csv", index=False)
    print(f"Wrote {args.out_dir / 'sampling_reconstruction_metrics.csv'}")


if __name__ == "__main__":
    main()
