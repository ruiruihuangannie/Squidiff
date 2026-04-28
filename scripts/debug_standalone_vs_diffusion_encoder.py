from pathlib import Path

import numpy as np
import pandas as pd

from debug_mouse_common import (
    REPO_ROOT,
    align_rna,
    common_parser,
    encode_adata,
    latent_metric_rows,
    load_data,
    load_sampler,
    maybe_subsample_by_group,
    pca_coords,
    prepare_run,
    scatter_by_label,
    to_numpy_matrix,
    umap_coords,
)


def build_standalone_autoencoder_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class EncoderMLPModel(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size=60, use_fp16=False):
            super().__init__()
            self.dtype = torch.float16 if use_fp16 else torch.float32
            self.fc1 = nn.Linear(input_size, hidden_sizes)
            self.bn1 = nn.BatchNorm1d(hidden_sizes)
            self.bn2 = nn.BatchNorm1d(hidden_sizes)
            self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
            self.fc3 = nn.Linear(hidden_sizes, output_size)
            self.label_embed = nn.Linear(1, hidden_sizes)

        def forward(self, x_start, label=None, drug_dose=None, control_feature=None):
            if label is not None:
                label_emb = self.label_embed(label)
                x_start = torch.concat([x_start, label_emb], axis=1)
            if drug_dose is not None:
                x_start = torch.concat([control_feature, drug_dose], axis=1)
            h = x_start.type(self.dtype)
            h = F.relu(self.bn1(self.fc1(h)))
            h = F.relu(self.bn2(self.fc2(h)))
            return self.fc3(h)

    class RNAAutoencoder(nn.Module):
        def __init__(self, input_size, hidden_size=512, latent_dim=60):
            super().__init__()
            self.encoder = EncoderMLPModel(input_size, hidden_size, output_size=latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
            )

        def forward(self, x):
            z = self.encoder(x)
            return z, self.decoder(z)

    return RNAAutoencoder


def load_standalone(path, input_size, hidden_size=512, latent_dim=60, device="cpu"):
    import torch

    RNAAutoencoder = build_standalone_autoencoder_class()
    model = RNAAutoencoder(input_size=input_size, hidden_size=hidden_size, latent_dim=latent_dim).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_standalone_dims(path):
    import torch

    state = torch.load(path, map_location="cpu")
    return {
        "input_size": int(state["encoder.fc1.weight"].shape[1]),
        "hidden_size": int(state["encoder.fc1.weight"].shape[0]),
        "latent_dim": int(state["encoder.fc3.weight"].shape[0]),
    }


def encode_standalone(model, matrix, device, batch_size=1024):
    import torch

    x = torch.tensor(to_numpy_matrix(matrix), dtype=torch.float32, device=device)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            z, _ = model(x[start : start + batch_size])
            outputs.append(z.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main():
    parser = common_parser("Compare standalone RNA autoencoder latent with diffusion encoder latent.")
    parser.add_argument("--standalone-path", default="checkpoints/rna_encoder.pt")
    parser.add_argument("--standalone-hidden-size", type=int, default=512)
    parser.add_argument("--standalone-latent-dim", type=int, default=60)
    args = prepare_run(parser.parse_args())
    standalone_path = Path(args.standalone_path)
    if not standalone_path.is_absolute():
        standalone_path = REPO_ROOT / standalone_path

    sampler = load_sampler(args.model_path, args.seed)
    train_raw = load_data(args.train_path)["rna"].copy()
    test_raw = load_data(args.test_path)["rna"].copy()
    train_aligned = maybe_subsample_by_group(align_rna(sampler, load_data(args.train_path)), args.max_cells_per_group, seed=args.seed)
    test_aligned = maybe_subsample_by_group(align_rna(sampler, load_data(args.test_path)), args.max_cells_per_group, seed=args.seed)

    rows = []
    diffusion_sets = {"train_aligned100": train_aligned, "test_aligned100": test_aligned}
    for name, adata in diffusion_sets.items():
        z = encode_adata(sampler, adata, batch_size=args.batch_size, mode="eval")
        rows.extend(latent_metric_rows(f"diffusion_{name}", z, adata.obs))
        label_key = "stage" if "stage" in adata.obs.columns else "Group"
        scatter_by_label(pca_coords(z, seed=args.seed), adata.obs[label_key], f"Diffusion {name} PCA", args.out_dir / f"diffusion_{name}_pca.png")
        scatter_by_label(umap_coords(z, seed=args.seed), adata.obs[label_key], f"Diffusion {name} UMAP", args.out_dir / f"diffusion_{name}_umap.png")

    if standalone_path.exists():
        device = next(sampler.model.parameters()).device
        standalone_dims = infer_standalone_dims(standalone_path)
        standalone = load_standalone(
            standalone_path,
            input_size=standalone_dims["input_size"],
            hidden_size=standalone_dims["hidden_size"],
            latent_dim=standalone_dims["latent_dim"],
            device=device,
        )
        if standalone_dims["input_size"] == train_raw.n_vars:
            standalone_sets = {"train_full": train_raw, "test_full": test_raw}
        elif standalone_dims["input_size"] == train_aligned.n_vars:
            standalone_sets = {"train_aligned100": train_aligned, "test_aligned100": test_aligned}
        else:
            standalone_sets = {}
            rows.append(
                {
                    "name": "standalone_checkpoint_dim_mismatch",
                    "label_key": "",
                    "n_cells": 0,
                    "latent_dim": standalone_dims["latent_dim"],
                    "latent_mean": np.nan,
                    "latent_std": np.nan,
                    "latent_norm_mean": np.nan,
                    "latent_norm_std": np.nan,
                    "n_labels": 0,
                    "silhouette": np.nan,
                    "centroid_distance_min": np.nan,
                    "centroid_distance_mean": np.nan,
                    "centroid_distance_max": np.nan,
                    "standalone_input_size": standalone_dims["input_size"],
                    "full_rna_n_vars": train_raw.n_vars,
                    "aligned_rna_n_vars": train_aligned.n_vars,
                }
            )
        for name, adata in standalone_sets.items():
            adata = maybe_subsample_by_group(adata, args.max_cells_per_group, seed=args.seed)
            z = encode_standalone(standalone, adata.X, device, batch_size=args.batch_size)
            rows.extend(latent_metric_rows(f"standalone_{name}", z, adata.obs))
            label_key = "stage" if "stage" in adata.obs.columns else "Group"
            scatter_by_label(pca_coords(z, seed=args.seed), adata.obs[label_key], f"Standalone {name} PCA", args.out_dir / f"standalone_{name}_pca.png")
            scatter_by_label(umap_coords(z, seed=args.seed), adata.obs[label_key], f"Standalone {name} UMAP", args.out_dir / f"standalone_{name}_umap.png")
    else:
        rows.append(
            {
                "name": "standalone_checkpoint_missing",
                "label_key": "",
                "n_cells": 0,
                "latent_dim": 0,
                "latent_mean": np.nan,
                "latent_std": np.nan,
                "latent_norm_mean": np.nan,
                "latent_norm_std": np.nan,
                "n_labels": 0,
                "silhouette": np.nan,
                "centroid_distance_min": np.nan,
                "centroid_distance_mean": np.nan,
                "centroid_distance_max": np.nan,
            }
        )

    pd.DataFrame(rows).to_csv(args.out_dir / "encoder_objective_gap_metrics.csv", index=False)
    print(f"Wrote {args.out_dir / 'encoder_objective_gap_metrics.csv'}")


if __name__ == "__main__":
    main()
