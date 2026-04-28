import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, r2_score, silhouette_score


DEFAULT_MODEL_PATH = "checkpoints/20260426_003254/best_model_0.9999.pt"
DEFAULT_TRAIN_PATH = "data/mouse/processed/mouse_train.h5mu"
DEFAULT_TEST_PATH = "data/mouse/processed/mouse_test.h5mu"
DEFAULT_OUT_DIR = "results/mouse_diff_debug"


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "Squidiff").exists() and (candidate / "sample_squidiff.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root containing Squidiff/ and sample_squidiff.py.")


REPO_ROOT = repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def import_mu():
    try:
        import muon as mu

        return mu
    except ImportError:
        import mudata as mu

        return mu


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-cells-per-group", type=int, default=None)
    return parser


def prepare_run(args):
    seed_everything(args.seed)
    args.model_path = resolve_path(args.model_path)
    args.train_path = resolve_path(args.train_path)
    args.test_path = resolve_path(args.test_path)
    args.out_dir = resolve_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_sampler(model_path, seed):
    import sample_squidiff

    return sample_squidiff.sampler(model_path=str(model_path), seed=seed)


def load_data(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".h5mu":
        return import_mu().read_h5mu(path)
    if suffix == ".h5ad":
        import scanpy as sc

        return sc.read_h5ad(path)
    raise ValueError(f"Unsupported data path: {path}")


def extract_rna_adata(data_obj):
    if hasattr(data_obj, "mod"):
        return data_obj["rna"].copy()
    return data_obj.copy()


def to_numpy_matrix(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def align_rna(sampler, data_obj):
    return sampler.align_rna_adata(extract_rna_adata(data_obj))


def subset_by_group(adata, group):
    return adata[adata.obs["Group"] == group].copy()


def maybe_subsample_by_group(adata, max_cells_per_group=None, group_key="Group", seed=42):
    if max_cells_per_group is None:
        return adata.copy()
    rng = np.random.default_rng(seed)
    selected = []
    for _, idx in adata.obs.groupby(group_key, observed=False).indices.items():
        idx = np.asarray(idx)
        if idx.size > max_cells_per_group:
            idx = rng.choice(idx, size=max_cells_per_group, replace=False)
        selected.extend(idx.tolist())
    selected = np.asarray(sorted(selected))
    return adata[selected].copy()


def tensor_from_adata(adata, device):
    import torch

    return torch.tensor(to_numpy_matrix(adata.X), dtype=torch.float32, device=device)


def model_device(sampler):
    return next(sampler.model.parameters()).device


def bn_module_summary(model):
    import torch

    rows = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            rows.append(
                {
                    "name": name,
                    "training": bool(module.training),
                    "track_running_stats": bool(module.track_running_stats),
                    "running_mean_mean": tensor_scalar_mean(module.running_mean),
                    "running_mean_std": tensor_scalar_std(module.running_mean),
                    "running_var_mean": tensor_scalar_mean(module.running_var),
                    "running_var_std": tensor_scalar_std(module.running_var),
                }
            )
    return rows


def tensor_scalar_mean(tensor):
    if tensor is None:
        return None
    return float(tensor.detach().float().mean().cpu())


def tensor_scalar_std(tensor):
    if tensor is None:
        return None
    return float(tensor.detach().float().std(unbiased=False).cpu())


def encode_tensor(sampler, tensor, batch_size=1024, mode="eval"):
    import torch

    previous_mode = sampler.model.training
    if mode == "eval":
        sampler.model.eval()
    elif mode == "train":
        sampler.model.train()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    outputs = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size]
            outputs.append(sampler.model.encoder(batch).detach().cpu().numpy())

    sampler.model.train(previous_mode)
    return np.concatenate(outputs, axis=0)


def encode_adata(sampler, adata, batch_size=1024, mode="eval"):
    return encode_tensor(sampler, tensor_from_adata(adata, model_device(sampler)), batch_size=batch_size, mode=mode)


def encode_adata_by_group(sampler, adata, batch_size=1024, mode="eval", group_key="Group"):
    chunks = []
    for group in pd.Categorical(adata.obs[group_key]).categories:
        sub = adata[adata.obs[group_key].astype(str) == str(group)].copy()
        z = encode_adata(sampler, sub, batch_size=batch_size, mode=mode)
        chunks.append(pd.DataFrame(z, index=sub.obs_names))
    out = pd.concat(chunks, axis=0).loc[adata.obs_names]
    return out.to_numpy()


def expression_summary(name, matrix):
    arr = to_numpy_matrix(matrix).astype(np.float64)
    flat = arr.ravel()
    gene_mean = arr.mean(axis=0)
    return {
        "name": name,
        "n_cells": int(arr.shape[0]),
        "n_genes": int(arr.shape[1]),
        "min": float(np.min(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q05": float(np.quantile(flat, 0.05)),
        "median": float(np.quantile(flat, 0.50)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "frac_lt_1": float(np.mean(flat < 1.0)),
        "frac_zero": float(np.mean(flat == 0.0)),
        "frac_negative": float(np.mean(flat < 0.0)),
        "gene_mean_min": float(np.min(gene_mean)),
        "gene_mean_median": float(np.median(gene_mean)),
        "gene_mean_max": float(np.max(gene_mean)),
    }


def mean_metric_rows(name, pred_matrix, truth_adata):
    pred = to_numpy_matrix(pred_matrix).astype(np.float64)
    truth = to_numpy_matrix(truth_adata.X).astype(np.float64)
    pred_mean = np.asarray(pred.mean(axis=0)).ravel()
    true_mean = np.asarray(truth.mean(axis=0)).ravel()
    pearson, pearson_p = scipy.stats.pearsonr(pred_mean, true_mean)
    return {
        "name": name,
        "r2_true_pred": float(r2_score(true_mean, pred_mean)),
        "r2_pred_true_current_bug": float(r2_score(pred_mean, true_mean)),
        "pearsonr": float(pearson),
        "pearson_p": float(pearson_p),
        "pred_mean_mean": float(pred_mean.mean()),
        "true_mean_mean": float(true_mean.mean()),
        "pred_mean_std": float(pred_mean.std()),
        "true_mean_std": float(true_mean.std()),
        "mean_bias": float((pred_mean - true_mean).mean()),
        "mae_gene_mean": float(np.mean(np.abs(pred_mean - true_mean))),
        "rmse_gene_mean": float(np.sqrt(np.mean((pred_mean - true_mean) ** 2))),
    }


def latent_metric_rows(name, z, obs, label_keys=("Group", "stage")):
    z = np.asarray(z)
    rows = []
    norm = np.linalg.norm(z, axis=1)
    base = {
        "name": name,
        "n_cells": int(z.shape[0]),
        "latent_dim": int(z.shape[1]),
        "latent_mean": float(z.mean()),
        "latent_std": float(z.std()),
        "latent_norm_mean": float(norm.mean()),
        "latent_norm_std": float(norm.std()),
    }
    for key in label_keys:
        if key not in obs.columns:
            continue
        labels = obs[key].astype(str).to_numpy()
        row = dict(base)
        row["label_key"] = key
        row["n_labels"] = int(pd.Series(labels).nunique())
        row["silhouette"] = safe_silhouette(z, labels)
        centroid_stats = centroid_distance_stats(z, labels)
        row.update(centroid_stats)
        rows.append(row)
    if not rows:
        row = dict(base)
        row["label_key"] = ""
        row["n_labels"] = 0
        row["silhouette"] = np.nan
        rows.append(row)
    return rows


def safe_silhouette(z, labels, max_cells=5000, seed=42):
    if pd.Series(labels).nunique() < 2:
        return np.nan
    if z.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(z.shape[0], size=max_cells, replace=False)
        z = z[idx]
        labels = np.asarray(labels)[idx]
    try:
        return float(silhouette_score(z, labels))
    except Exception:
        return np.nan


def centroid_distance_stats(z, labels):
    labels = np.asarray(labels)
    centroids = []
    for label in pd.Series(labels).dropna().astype(str).unique():
        mask = labels.astype(str) == str(label)
        if mask.any():
            centroids.append(z[mask].mean(axis=0))
    if len(centroids) < 2:
        return {
            "centroid_distance_min": np.nan,
            "centroid_distance_mean": np.nan,
            "centroid_distance_max": np.nan,
        }
    distances = pairwise_distances(np.vstack(centroids))
    upper = distances[np.triu_indices_from(distances, k=1)]
    return {
        "centroid_distance_min": float(upper.min()),
        "centroid_distance_mean": float(upper.mean()),
        "centroid_distance_max": float(upper.max()),
    }


def pca_coords(matrix, n_components=2, seed=42):
    return PCA(n_components=n_components, random_state=seed).fit_transform(np.asarray(matrix))


def umap_coords(matrix, seed=42, n_neighbors=15, min_dist=0.2):
    try:
        import umap

        return umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=seed,
        ).fit_transform(np.asarray(matrix))
    except ImportError:
        import scanpy as sc
        import anndata

        adata = anndata.AnnData(X=np.asarray(matrix))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata, random_state=seed, min_dist=min_dist)
        return adata.obsm["X_umap"]


def scatter_by_label(coords, labels, title, path, *, size=5, alpha=0.75):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.Categorical(pd.Series(labels).astype(str))
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set2").colors)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    for idx, category in enumerate(labels.categories):
        mask = np.asarray(labels) == category
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            color=palette[idx % len(palette)],
            label=str(category),
            edgecolors="none",
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter_pred_truth(pred_matrix, truth_adata, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_mean = to_numpy_matrix(pred_matrix).mean(axis=0).ravel()
    true_mean = to_numpy_matrix(truth_adata.X).mean(axis=0).ravel()
    lim_min = float(min(pred_mean.min(), true_mean.min()))
    lim_max = float(max(pred_mean.max(), true_mean.max()))
    fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
    ax.scatter(pred_mean, true_mean, s=10, alpha=0.8, edgecolors="none")
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Predicted gene mean")
    ax.set_ylabel("Truth gene mean")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sample_from_z(sampler, z, seed=None, use_ddim=None):
    import torch

    if seed is not None:
        seed_everything(seed)
    original_sample_fn = sampler.sample_fn
    if use_ddim is not None:
        sampler.sample_fn = sampler.diffusion.ddim_sample_loop if use_ddim else sampler.diffusion.p_sample_loop
    try:
        with torch.no_grad():
            out = sampler.pred(z_sem=z)
    finally:
        sampler.sample_fn = original_sample_fn
    return out.detach().cpu().numpy()


def current_interp_z(z_origin, direction, scale, add_noise_term=True, noise_scale=0.7, seed=42):
    rng = np.random.default_rng(seed)
    z_origin_np = to_numpy_matrix(z_origin)
    direction_np = to_numpy_matrix(direction).ravel()
    center = z_origin_np.mean(axis=0) + direction_np * scale
    if add_noise_term:
        return center + noise_scale * rng.standard_normal((z_origin_np.shape[0], z_origin_np.shape[1]))
    return np.repeat(center[None, :], z_origin_np.shape[0], axis=0)


def per_cell_interp_z(z_origin, direction, scale, add_matched_noise=False, noise_scale=0.7, seed=42):
    rng = np.random.default_rng(seed)
    z_origin_np = to_numpy_matrix(z_origin)
    direction_np = to_numpy_matrix(direction).ravel()
    out = z_origin_np + direction_np * scale
    if add_matched_noise:
        out = out + noise_scale * rng.standard_normal(out.shape)
    return out


def torch_z(array, device):
    import torch

    return torch.tensor(np.asarray(array), dtype=torch.float32, device=device)
