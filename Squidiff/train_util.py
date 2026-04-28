"""
This code is adapted from openai's guided-diffusion models and Konpat's diffae model:
https://github.com/openai/guided-diffusion
https://github.com/phizaz/diffae
"""
import copy
import functools
import glob
import json
import os
import random
from collections import defaultdict
from contextlib import nullcontext

import torch as th
from torch.optim import AdamW

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None

from . import dist_util, wandb_util
from .model_spec import save_atac_features, save_model_spec, save_rna_features
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


# Deprecated in favor of wandb metric tracking.
# def plot_loss(losses, args_train):
#     pass


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_drug_structure=False,
        comb_num=1,
        model_spec=None,
        rna_feature_names=None,
        rna_features_file=None,
        atac_feature_names=None,
        atac_features_file=None,
        val_data=None,
        use_ddim=True,
        val_recon_interval_epochs=10,
    ):
        self.accelerator = dist_util.accelerator()
        self.model = model
        self.diffusion = diffusion
        self.use_drug_structure = use_drug_structure
        self.data = data
        self.batch_size = int(batch_size)
        self.microbatch = int(microbatch) if microbatch > 0 else int(batch_size)
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = int(log_interval)
        self.save_interval = int(save_interval)
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = int(lr_anneal_steps)
        self.comb_num = comb_num
        self.model_spec = model_spec
        self.rna_feature_names = rna_feature_names
        self.rna_features_file = rna_features_file
        self.atac_feature_names = atac_feature_names
        self.atac_features_file = atac_features_file
        self.val_data = val_data
        self.use_ddim = use_ddim
        self.val_recon_interval_epochs = int(val_recon_interval_epochs)

        self.step = 0
        self.resume_step = 0
        self.epoch = 0
        self.loss_list = []
        self.best_val_loss = None
        self.model_checkpoint = resolve_model_checkpoint(self.resume_checkpoint)
        self.checkpoint_dir = resolve_checkpoint_dir(self.resume_checkpoint)

        if self.model_checkpoint:
            self.resume_step = infer_resume_step(self.model_checkpoint, self.checkpoint_dir)
            self.model.load_state_dict(
                dist_util.load_state_dict(self.model_checkpoint, map_location="cpu")
            )

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.val_data is not None:
            self.model, self.opt, self.data, self.val_data = dist_util.prepare(
                self.model, self.opt, self.data, self.val_data
            )
        else:
            self.model, self.opt, self.data = dist_util.prepare(self.model, self.opt, self.data)
        self.global_batch = self.batch_size * dist_util.num_processes()

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            base_params = self._clone_model_params()
            self.ema_params = [copy.deepcopy(base_params) for _ in self.ema_rate]

    def _load_and_sync_parameters(self):
        if self.model_checkpoint:
            self.resume_step = infer_resume_step(self.model_checkpoint, self.checkpoint_dir)
            self.model.load_state_dict(
                dist_util.load_state_dict(self.model_checkpoint, map_location="cpu")
            )

    def _load_ema_parameters(self, rate):
        ema_params = self._clone_model_params()
        ema_checkpoint = find_ema_checkpoint(self.model_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = [
                state_dict[name].detach().to(dist_util.dev())
                for name, _ in dist_util.unwrap_model(self.model).named_parameters()
            ]
        return ema_params

    def _load_optimizer_state(self):
        if not self.checkpoint_dir:
            return

        opt_checkpoint = os.path.join(self.checkpoint_dir, f"opt{self.resume_step:06d}.pt")
        if os.path.exists(opt_checkpoint):
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            sampler = getattr(self.data, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            epoch_metric_totals = defaultdict(float)
            epoch_metric_counts = defaultdict(int)

            for batch in self.data:
                if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                    break

                metrics = self.run_step(batch)
                self._accumulate_metrics(epoch_metric_totals, epoch_metric_counts, metrics)
                self.step += 1

                if self.step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", ""):
                        return

            completed_epoch = self.epoch + 1
            global_step = self.step + self.resume_step
            training_metrics = self._rename_training_metrics(
                self._finalize_metrics(epoch_metric_totals, epoch_metric_counts)
            )
            metrics_to_log = dict(training_metrics)
            if self.val_data is not None:
                should_log_val_recon = self._should_log_val_recon(completed_epoch)
                val_metrics, val_fig = self.run_validation(make_fig=should_log_val_recon)
                metrics_to_log.update(val_metrics)
                wandb_util.log(metrics_to_log, step=global_step)
                if val_fig is not None:
                    wandb_util.log_figure(
                        f"val_recon",
                        val_fig,
                        step=global_step,
                        filename=f"epoch_{completed_epoch:04d}.png",
                    )
                    plt.close(val_fig)
                current_val_loss = val_metrics.get("val_loss")
                if current_val_loss is not None and (
                    self.best_val_loss is None or current_val_loss < self.best_val_loss
                ):
                    self.best_val_loss = current_val_loss
                    self.save_best(val_metrics, epoch=completed_epoch, step=global_step)
                    wandb_util.update_summary(
                        {
                            "best_val_loss": current_val_loss,
                            "best_val_epoch": completed_epoch,
                            "best_val_step": global_step,
                        }
                    )
            elif metrics_to_log:
                wandb_util.log(metrics_to_log, step=global_step)
            self.epoch += 1

        if self.step > 0 and self.step % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        metrics = self.forward_backward(batch)
        took_step, opt_metrics = self.optimize_step()
        if took_step:
            self._update_ema()
        self._anneal_lr()
        metrics.update(opt_metrics)
        metrics["lr"] = self.opt.param_groups[0]["lr"]
        return metrics

    def forward_backward(self, batch):
        self.opt.zero_grad(set_to_none=True)
        metrics, last_loss = self._compute_batch_metrics(batch, backward=True)
        if last_loss is not None:
            self.loss_list.append(last_loss)
        return metrics

    def _build_micro_condition(self, batch, start, end):
        if self.use_drug_structure:
            return {
                "group": batch["group"][start:end],
                "drug_dose": batch["drug_dose"][start:end].to(dist_util.dev()),
                "control_feature": batch["control_feature"][start:end].to(dist_util.dev()),
                "atac_feature": None,
            }
        if "atac_feature" in batch:
            return {
                "group": batch["group"][start:end],
                "drug_dose": None,
                "control_feature": None,
                "atac_feature": batch["atac_feature"][start:end].to(dist_util.dev()),
            }
        return {
            "group": batch["group"][start:end],
            "drug_dose": None,
            "control_feature": None,
            "atac_feature": None,
        }

    def _compute_batch_metrics(self, batch, *, backward):
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)
        last_loss = None

        for i in range(0, batch["feature"].shape[0], self.microbatch):
            micro = batch["feature"][i : i + self.microbatch].to(dist_util.dev())
            micro_cond = self._build_micro_condition(batch, i, i + self.microbatch)

            last_batch = (i + self.microbatch) >= batch["feature"].shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            sync_context = nullcontext()
            if not last_batch and dist_util.num_processes() > 1:
                sync_context = self.accelerator.no_sync(self.model)

            with sync_context:
                losses = compute_losses()

                if backward and isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()

                if backward:
                    dist_util.backward(loss)

            metrics = summarize_loss_dict(
                {key: value * weights for key, value in losses.items()}
            )
            self._accumulate_metrics(
                metric_totals,
                metric_counts,
                metrics,
            )

            last_loss = dist_util.reduce_tensor(loss.detach(), reduction="mean").item()

        return self._finalize_metrics(metric_totals, metric_counts), last_loss

    def optimize_step(self):
        grad_norm, param_norm = self._compute_norms()
        self.opt.step()
        metrics = {
            "grad_norm": dist_util.reduce_tensor(
                th.tensor(grad_norm, device=dist_util.dev()), reduction="mean"
            ).item(),
            "param_norm": dist_util.reduce_tensor(
                th.tensor(param_norm, device=dist_util.dev()), reduction="mean"
            ).item(),
        }
        return True, metrics

    def _clone_model_params(self):
        return [
            param.detach().clone()
            for param in dist_util.unwrap_model(self.model).parameters()
        ]

    def _accumulate_metrics(self, metric_totals, metric_counts, metrics):
        for key, value in metrics.items():
            metric_totals[key] += float(value)
            metric_counts[key] += 1

    def _finalize_metrics(self, metric_totals, metric_counts):
        if not metric_counts:
            return {}
        return {
            key: metric_totals[key] / metric_counts[key]
            for key in metric_totals
        }

    def _rename_training_metrics(self, metrics):
        renamed_metrics = {}
        for key, value in metrics.items():
            if key == "loss":
                renamed_metrics["training_loss"] = value
            elif key == "mse":
                renamed_metrics["training_mse"] = value
            else:
                renamed_metrics[key] = value
        return renamed_metrics

    def _compute_norms(self):
        grad_norm = 0.0
        param_norm = 0.0
        for p in dist_util.unwrap_model(self.model).parameters():
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return grad_norm ** 0.5, param_norm ** 0.5

    def _state_dict_for_params(self, params):
        model_to_save = dist_util.unwrap_model(self.model)
        state_dict = model_to_save.state_dict()
        for (name, _), param in zip(model_to_save.named_parameters(), params):
            state_dict[name] = param.detach().cpu()
        return state_dict

    def _current_model_state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in dist_util.unwrap_model(self.model).state_dict().items()
        }

    def _checkpoint_path(self, filename):
        if not self.checkpoint_dir:
            raise ValueError("A checkpoint directory is required for saving.")
        return os.path.join(self.checkpoint_dir, filename)

    def _update_ema(self):
        current_params = [
            param.detach() for param in dist_util.unwrap_model(self.model).parameters()
        ]
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, current_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if dist_util.is_main_process():
            self._save_model_metadata()
            with open(self._checkpoint_path("model.pt"), "wb") as f:
                th.save(self._current_model_state_dict(), f)

            for rate, params in zip(self.ema_rate, self.ema_params):
                with open(self._checkpoint_path(f"model_{rate}.pt"), "wb") as f:
                    th.save(self._state_dict_for_params(params), f)

            opt_filename = f"opt{(self.step + self.resume_step):06d}.pt"
            with open(self._checkpoint_path(opt_filename), "wb") as f:
                th.save(self.opt.state_dict(), f)

        dist_util.wait_for_everyone()

    def _save_model_metadata(self):
        if self.model_spec is None:
            return

        save_model_spec(self.model_spec, self.checkpoint_dir)
        if self.rna_feature_names:
            save_rna_features(
                self.rna_feature_names,
                self.checkpoint_dir,
                filename=self.rna_features_file,
            )
        if self.atac_feature_names:
            save_atac_features(
                self.atac_feature_names,
                self.checkpoint_dir,
                filename=self.atac_features_file,
            )

    def _should_log_val_recon(self, completed_epoch):
        if self.val_recon_interval_epochs <= 0:
            return False
        return completed_epoch == 1 or completed_epoch % self.val_recon_interval_epochs == 0

    def run_validation(self, *, make_fig=True):
        self.model.eval()
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)
        selected_batch_idx = self._validation_batch_index() if make_fig else None
        selected_batch = None

        with th.no_grad():
            for batch_idx, batch in enumerate(self.val_data):
                metrics, _ = self._compute_batch_metrics(batch, backward=False)
                for key, value in metrics.items():
                    metric_totals[f"val_{key}"] += float(value)
                    metric_counts[f"val_{key}"] += 1
                if batch_idx == selected_batch_idx:
                    selected_batch = batch

        self.model.train()

        val_metrics = {
            key: metric_totals[key] / metric_counts[key]
            for key in metric_totals
            if metric_counts[key] > 0
        }
        val_fig = self._make_validation_scatter(selected_batch, self.epoch + 1) if selected_batch is not None else None
        return val_metrics, val_fig

    def _validation_batch_index(self):
        try:
            num_batches = len(self.val_data)
        except TypeError:
            return 0
        if num_batches <= 1:
            return 0
        return random.randrange(num_batches)

    def _encode_validation_condition(self, batch, feature_tensor):
        model_to_use = dist_util.unwrap_model(self.model)
        if not getattr(model_to_use, "use_encoder", False):
            return None

        if getattr(model_to_use, "has_atac_encoder", False):
            return model_to_use.encoder(feature_tensor, batch["atac_feature"].to(dist_util.dev()))

        if self.use_drug_structure:
            return model_to_use.encoder(
                feature_tensor,
                label=None,
                drug_dose=batch["drug_dose"].to(dist_util.dev()),
                control_feature=batch["control_feature"].to(dist_util.dev()),
            )

        if model_to_use.num_classes is None:
            return model_to_use.encoder(
                feature_tensor,
                label=None,
                drug_dose=None,
                control_feature=None,
            )

        group = batch["group"]
        if not th.is_tensor(group):
            group = th.tensor(group, dtype=th.float32)
        return model_to_use.encoder(
            feature_tensor,
            label=group.to(dist_util.dev()),
            drug_dose=None,
            control_feature=None,
        )

    def _make_validation_scatter(self, batch, epoch):
        if plt is None or batch is None:
            return None

        model_to_use = dist_util.unwrap_model(self.model)
        if not getattr(model_to_use, "use_encoder", False):
            return None

        feature_tensor = batch["feature"].to(dist_util.dev())
        with th.no_grad():
            z_sem = self._encode_validation_condition(batch, feature_tensor)
            if z_sem is None:
                return None
            sample_fn = self.diffusion.ddim_sample_loop if self.use_ddim else self.diffusion.p_sample_loop
            pred = sample_fn(
                self.model,
                shape=(feature_tensor.shape[0], feature_tensor.shape[1]),
                model_kwargs={"z_mod": z_sem},
                noise=None,
            )

        pred_mean = pred.detach().cpu().numpy().mean(axis=0)
        true_mean = feature_tensor.detach().cpu().numpy().mean(axis=0)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.scatter(pred_mean, true_mean, s=8, alpha=0.7, c="#40a8f7", edgecolors="none")
        lim_min = float(min(pred_mean.min(), true_mean.min()))
        lim_max = float(max(pred_mean.max(), true_mean.max()))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Predicted mean expression")
        ax.set_ylabel("True mean expression")
        ax.set_title(f"Validation Recon Epoch {epoch}")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return fig

    def save_best(self, metrics, *, epoch, step):
        if not self.checkpoint_dir:
            return

        if dist_util.is_main_process():
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._save_model_metadata()
            with open(self._checkpoint_path("best_model.pt"), "wb") as f:
                th.save(self._current_model_state_dict(), f)

            for rate, params in zip(self.ema_rate, self.ema_params):
                with open(self._checkpoint_path(f"best_model_{rate}.pt"), "wb") as f:
                    th.save(self._state_dict_for_params(params), f)

            best_payload = dict(metrics)
            best_payload.update({"best_epoch": epoch, "best_step": step})
            with open(self._checkpoint_path("best_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(best_payload, handle, indent=2, sort_keys=True)

        dist_util.wait_for_everyone()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def infer_resume_step(model_checkpoint, checkpoint_dir):
    parsed_step = parse_resume_step_from_filename(model_checkpoint)
    if parsed_step:
        return parsed_step

    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return 0

    opt_steps = []
    for opt_path in glob.glob(os.path.join(checkpoint_dir, "opt*.pt")):
        stem = os.path.splitext(os.path.basename(opt_path))[0]
        step_str = stem.replace("opt", "", 1)
        if step_str.isdigit():
            opt_steps.append(int(step_str))
    return max(opt_steps, default=0)


def find_ema_checkpoint(main_checkpoint, step, rate):
    del step
    if main_checkpoint is None:
        return None
    checkpoint_dir = resolve_checkpoint_dir(main_checkpoint)
    if checkpoint_dir is None:
        return None
    path = os.path.join(checkpoint_dir, f"model_{rate}.pt")
    if os.path.exists(path):
        return path
    return None


def resolve_checkpoint_dir(resume_checkpoint):
    if not resume_checkpoint:
        return None
    if resume_checkpoint.endswith(".pt"):
        return os.path.dirname(resume_checkpoint)
    return resume_checkpoint


def resolve_model_checkpoint(resume_checkpoint):
    if not resume_checkpoint:
        return None
    if resume_checkpoint.endswith(".pt"):
        return resume_checkpoint if os.path.exists(resume_checkpoint) else None
    model_path = os.path.join(resume_checkpoint, "model.pt")
    return model_path if os.path.exists(model_path) else None


def summarize_loss_dict(losses):
    metrics = {}
    for key, values in losses.items():
        metrics[key] = dist_util.reduce_tensor(
            values.mean().detach(), reduction="mean"
        ).item()
    return metrics
