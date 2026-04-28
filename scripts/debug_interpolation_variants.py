import numpy as np
import pandas as pd

from debug_mouse_common import (
    align_rna,
    common_parser,
    current_interp_z,
    encode_adata,
    expression_summary,
    latent_metric_rows,
    load_data,
    load_sampler,
    mean_metric_rows,
    maybe_subsample_by_group,
    pca_coords,
    per_cell_interp_z,
    prepare_run,
    sample_from_z,
    scatter_by_label,
    scatter_pred_truth,
    subset_by_group,
    torch_z,
    umap_coords,
)


def main():
    parser = common_parser("Compare interpolation variants without editing sample_squidiff.py.")
    args = prepare_run(parser.parse_args())

    sampler = load_sampler(args.model_path, args.seed)
    train = maybe_subsample_by_group(align_rna(sampler, load_data(args.train_path)), args.max_cells_per_group, seed=args.seed)
    test = maybe_subsample_by_group(align_rna(sampler, load_data(args.test_path)), args.max_cells_per_group, seed=args.seed)
    device = next(sampler.model.parameters()).device

    train_day0 = subset_by_group(train, 0)
    train_day3 = subset_by_group(train, 3)
    test_days = {day: subset_by_group(test, day) for day in [0, 1, 2, 3]}

    z_train_day0 = encode_adata(sampler, train_day0, batch_size=args.batch_size, mode="eval")
    z_train_day3 = encode_adata(sampler, train_day3, batch_size=args.batch_size, mode="eval")
    z_test_day0 = encode_adata(sampler, test_days[0], batch_size=args.batch_size, mode="eval")
    direction = z_train_day3.mean(axis=0) - z_train_day0.mean(axis=0)

    variant_specs = []
    for day, scale in [(1, 1 / 3), (2, 2 / 3)]:
        variant_specs.extend(
            [
                (day, "current_centroid_direction_noise", current_interp_z(z_test_day0, direction, scale, True, seed=args.seed)),
                (day, "centroid_direction_no_noise", current_interp_z(z_test_day0, direction, scale, False, seed=args.seed)),
                (day, "per_cell_direction_no_noise", per_cell_interp_z(z_test_day0, direction, scale, False, seed=args.seed)),
                (day, "per_cell_direction_matched_noise", per_cell_interp_z(z_test_day0, direction, scale, True, seed=args.seed)),
                (day, "oracle_true_day_encoded_z", encode_adata(sampler, test_days[day], batch_size=args.batch_size, mode="eval")),
            ]
        )

    metric_rows = []
    scale_rows = []
    latent_rows = []
    plot_z = []
    plot_labels = []
    for day, variant_name, z_variant in variant_specs:
        full_name = f"day{day}_{variant_name}"
        pred = sample_from_z(sampler, torch_z(z_variant, device), seed=args.seed)
        metric_rows.append(mean_metric_rows(full_name, pred, test_days[day]))
        scale_rows.append(expression_summary(full_name, pred))
        latent_rows.extend(latent_metric_rows(full_name, z_variant, test_days[day].obs))
        scatter_pred_truth(pred, test_days[day], full_name, args.out_dir / f"{full_name}_scatter.png")
        plot_z.append(z_variant)
        plot_labels.extend([full_name] * z_variant.shape[0])

    all_z = np.concatenate(plot_z, axis=0)
    labels = np.asarray(plot_labels)
    scatter_by_label(pca_coords(all_z, seed=args.seed), labels, "Interpolation variant latent PCA", args.out_dir / "interpolation_variant_latent_pca.png", size=4)
    scatter_by_label(umap_coords(all_z, seed=args.seed), labels, "Interpolation variant latent UMAP", args.out_dir / "interpolation_variant_latent_umap.png", size=4)

    pd.DataFrame(metric_rows).to_csv(args.out_dir / "interpolation_variant_metrics.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.out_dir / "interpolation_variant_scale_metrics.csv", index=False)
    pd.DataFrame(latent_rows).to_csv(args.out_dir / "interpolation_variant_latent_metrics.csv", index=False)
    print(f"Wrote {args.out_dir / 'interpolation_variant_metrics.csv'}")


if __name__ == "__main__":
    main()
