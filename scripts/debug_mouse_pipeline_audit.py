from pathlib import Path

import pandas as pd

from debug_mouse_common import (
    bn_module_summary,
    common_parser,
    expression_summary,
    load_data,
    load_sampler,
    prepare_run,
    REPO_ROOT,
    save_json,
    to_numpy_matrix,
    align_rna,
)


def stage_counts(adata):
    cols = [col for col in ["Group", "stage", "batch_stage", "replicate"] if col in adata.obs.columns]
    return {col: adata.obs[col].astype(str).value_counts(dropna=False).sort_index().to_dict() for col in cols}


def main():
    parser = common_parser("Audit mouse Squidiff checkpoint/data pipeline.")
    args = prepare_run(parser.parse_args())

    sampler = load_sampler(args.model_path, args.seed)
    import torch

    train_data = load_data(args.train_path)
    test_data = load_data(args.test_path)
    train_rna = train_data["rna"].copy() if hasattr(train_data, "mod") else train_data.copy()
    test_rna = test_data["rna"].copy() if hasattr(test_data, "mod") else test_data.copy()
    train_aligned = align_rna(sampler, train_data)
    test_aligned = align_rna(sampler, test_data)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    checkpoint_keys = list(checkpoint.keys())
    encoder_keys = [key for key in checkpoint_keys if key.startswith("encoder.")]

    rna_features = list(sampler.rna_features)
    train_feature_match = list(train_aligned.var_names) == rna_features
    test_feature_match = list(test_aligned.var_names) == rna_features

    payload = {
        "repo_root": str(REPO_ROOT),
        "model_path": str(args.model_path),
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "model_spec": dict(sampler.arg),
        "model_training": bool(sampler.model.training),
        "encoder_training": bool(sampler.model.encoder.training),
        "bn_modules": bn_module_summary(sampler.model),
        "checkpoint": {
            "n_keys": len(checkpoint_keys),
            "n_encoder_keys": len(encoder_keys),
            "looks_like_ema": args.model_path.name.startswith("best_model_") or args.model_path.name.startswith("model_"),
            "first_keys": checkpoint_keys[:20],
        },
        "features": {
            "saved_rna_feature_count": len(rna_features),
            "saved_rna_features_head": rna_features[:20],
            "raw_train_shape": list(train_rna.shape),
            "raw_test_shape": list(test_rna.shape),
            "aligned_train_shape": list(train_aligned.shape),
            "aligned_test_shape": list(test_aligned.shape),
            "train_feature_order_matches_checkpoint": train_feature_match,
            "test_feature_order_matches_checkpoint": test_feature_match,
            "raw_train_feature_count": int(train_rna.n_vars),
            "raw_test_feature_count": int(test_rna.n_vars),
        },
        "obs_counts": {
            "train_raw": stage_counts(train_rna),
            "test_raw": stage_counts(test_rna),
            "train_aligned": stage_counts(train_aligned),
            "test_aligned": stage_counts(test_aligned),
        },
        "expression": [
            expression_summary("train_raw_rna", train_rna.X),
            expression_summary("test_raw_rna", test_rna.X),
            expression_summary("train_aligned_rna", train_aligned.X),
            expression_summary("test_aligned_rna", test_aligned.X),
        ],
    }

    save_json(args.out_dir / "audit_summary.json", payload)
    pd.DataFrame(payload["expression"]).to_csv(args.out_dir / "audit_expression_summary.csv", index=False)
    print(f"Wrote {args.out_dir / 'audit_summary.json'}")
    print(f"Aligned train/test shapes: {train_aligned.shape} / {test_aligned.shape}")
    print(f"Model eval mode: model={not sampler.model.training}, encoder={not sampler.model.encoder.training}")
    print(f"Feature order matches checkpoint: train={train_feature_match}, test={test_feature_match}")


if __name__ == "__main__":
    main()
