import numpy as np
import pandas as pd

from debug_mouse_common import (
    align_rna,
    common_parser,
    encode_adata,
    encode_adata_by_group,
    latent_metric_rows,
    load_data,
    load_sampler,
    maybe_subsample_by_group,
    pca_coords,
    prepare_run,
    scatter_by_label,
    umap_coords,
)


def main():
    parser = common_parser("Compare diffusion encoder latent modes and BatchNorm sensitivity.")
    args = prepare_run(parser.parse_args())

    sampler = load_sampler(args.model_path, args.seed)
    datasets = {
        "train": maybe_subsample_by_group(align_rna(sampler, load_data(args.train_path)), args.max_cells_per_group, seed=args.seed),
        "test": maybe_subsample_by_group(align_rna(sampler, load_data(args.test_path)), args.max_cells_per_group, seed=args.seed),
    }

    rows = []
    for split, adata in datasets.items():
        variants = {
            "eval_all_batched": encode_adata(sampler, adata, batch_size=args.batch_size, mode="eval"),
            "train_all_batched": encode_adata(sampler, adata, batch_size=args.batch_size, mode="train"),
            "eval_per_group": encode_adata_by_group(sampler, adata, batch_size=args.batch_size, mode="eval"),
            "train_per_group": encode_adata_by_group(sampler, adata, batch_size=args.batch_size, mode="train"),
        }

        reference = variants["eval_all_batched"]
        for name, z in variants.items():
            full_name = f"{split}_{name}"
            rows.extend(latent_metric_rows(full_name, z, adata.obs))
            rows.append(
                {
                    "name": full_name,
                    "label_key": "reference_delta",
                    "n_cells": int(z.shape[0]),
                    "latent_dim": int(z.shape[1]),
                    "latent_mean": float(z.mean()),
                    "latent_std": float(z.std()),
                    "latent_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
                    "latent_norm_std": float(np.linalg.norm(z, axis=1).std()),
                    "n_labels": 0,
                    "silhouette": np.nan,
                    "centroid_distance_min": np.nan,
                    "centroid_distance_mean": float(np.linalg.norm(z - reference, axis=1).mean()),
                    "centroid_distance_max": float(np.linalg.norm(z - reference, axis=1).max()),
                }
            )

            label_key = "stage" if "stage" in adata.obs.columns else "Group"
            pca = pca_coords(z, seed=args.seed)
            scatter_by_label(pca, adata.obs[label_key], f"{full_name} PCA by {label_key}", args.out_dir / f"{full_name}_pca_{label_key}.png")
            umap = umap_coords(z, seed=args.seed)
            scatter_by_label(umap, adata.obs[label_key], f"{full_name} UMAP by {label_key}", args.out_dir / f"{full_name}_umap_{label_key}.png")

    out_csv = args.out_dir / "latent_mode_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
