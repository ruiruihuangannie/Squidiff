import numpy as np
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
)


def main():
    parser = common_parser("Reproduce mouse predictions and measure scale/R2 order.")
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
    z_test_day3 = encode_adata(sampler, test_days[3], batch_size=args.batch_size, mode="eval")
    direction = z_train_day3.mean(axis=0) - z_train_day0.mean(axis=0)

    from debug_mouse_common import current_interp_z, torch_z

    predictions = {
        "day0_pred_from_true_day0_z": sample_from_z(sampler, torch_z(z_test_day0, device), seed=args.seed),
        "day1_current_interp": sample_from_z(sampler, torch_z(current_interp_z(z_test_day0, direction, 1 / 3, True, seed=args.seed), device), seed=args.seed),
        "day2_current_interp": sample_from_z(sampler, torch_z(current_interp_z(z_test_day0, direction, 2 / 3, True, seed=args.seed), device), seed=args.seed),
        "day3_pred_from_true_day3_z": sample_from_z(sampler, torch_z(z_test_day3, device), seed=args.seed),
    }
    truth_map = {
        "day0_pred_from_true_day0_z": test_days[0],
        "day1_current_interp": test_days[1],
        "day2_current_interp": test_days[2],
        "day3_pred_from_true_day3_z": test_days[3],
    }

    scale_rows = []
    metric_rows = []
    for name, pred in predictions.items():
        scale_rows.append(expression_summary(name, pred))
        scale_rows.append(expression_summary(name.replace("pred", "truth"), truth_map[name].X))
        metric_rows.append(mean_metric_rows(name, pred, truth_map[name]))
        scatter_pred_truth(pred, truth_map[name], name, args.out_dir / f"{name}_scatter.png")

    baseline_rows = []
    train_day_means = {day: np.asarray(subset_by_group(train, day).X.mean(axis=0)).ravel() for day in sorted(train.obs["Group"].unique())}
    test_day_means = {day: np.asarray(test_days[day].X.mean(axis=0)).ravel() for day in test_days}
    for truth_day, truth in test_days.items():
        for base_name, pred_mean in [
            ("train_same_day_mean", train_day_means.get(truth_day)),
            ("test_day0_mean", test_day_means[0]),
            ("test_day3_mean", test_day_means[3]),
        ]:
            if pred_mean is None:
                continue
            pseudo_pred = np.repeat(pred_mean[None, :], truth.n_obs, axis=0)
            row = mean_metric_rows(f"{base_name}_vs_day{truth_day}", pseudo_pred, truth)
            baseline_rows.append(row)

    pd.DataFrame(scale_rows).to_csv(args.out_dir / "prediction_scale_metrics.csv", index=False)
    pd.DataFrame(metric_rows + baseline_rows).to_csv(args.out_dir / "r2_order_comparison.csv", index=False)
    print(f"Wrote {args.out_dir / 'prediction_scale_metrics.csv'}")
    print(f"Wrote {args.out_dir / 'r2_order_comparison.csv'}")


if __name__ == "__main__":
    main()
