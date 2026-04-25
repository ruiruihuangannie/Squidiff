"""
This code is adapted from openai's guided-diffusion models and Konpat's diffae model:
https://github.com/openai/guided-diffusion
https://github.com/phizaz/diffae
"""
import copy
import functools
import glob
import os
from collections import defaultdict
from contextlib import nullcontext

import torch as th
from torch.optim import AdamW

from . import dist_util, wandb_util
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

        self.step = 0
        self.resume_step = 0
        self.epoch = 0
        self.loss_list = []
        self.model_checkpoint = resolve_model_checkpoint(self.resume_checkpoint)
        self.checkpoint_dir = resolve_checkpoint_dir(self.resume_checkpoint)

        if self.model_checkpoint:
            self.resume_step = infer_resume_step(self.model_checkpoint, self.checkpoint_dir)
            self.model.load_state_dict(
                dist_util.load_state_dict(self.model_checkpoint, map_location="cpu")
            )

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

            for batch in self.data:
                if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                    break

                metrics = self.run_step(batch)
                global_step = self.step + self.resume_step

                if self.step % self.log_interval == 0:
                    wandb_util.log(metrics, step=global_step)

                if self.step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

            self.epoch += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        should_log = self.step % self.log_interval == 0
        metrics = self.forward_backward(batch, should_log=should_log)
        took_step, opt_metrics = self.optimize_step(should_log=should_log)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        metrics.update(opt_metrics)
        metrics.update(self.log_step())
        return metrics

    def forward_backward(self, batch, *, should_log):
        self.opt.zero_grad(set_to_none=True)
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)
        last_loss = None

        for i in range(0, batch["feature"].shape[0], self.microbatch):
            micro = batch["feature"][i : i + self.microbatch].to(dist_util.dev())
            if self.use_drug_structure:
                micro_cond = {
                    "group": batch["group"][i : i + self.microbatch],
                    "drug_dose": batch["drug_dose"][i : i + self.microbatch].to(dist_util.dev()),
                    "control_feature": batch["control_feature"][i : i + self.microbatch].to(dist_util.dev()),
                }
            else:
                micro_cond = {
                    "group": batch["group"][i : i + self.microbatch],
                    "drug_dose": None,
                    "control_feature": None,
                }

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

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                dist_util.backward(loss)

            if should_log:
                self._accumulate_metrics(
                    metric_totals,
                    metric_counts,
                    summarize_loss_dict(
                        self.diffusion,
                        t,
                        {key: value * weights for key, value in losses.items()},
                    ),
                )

            last_loss = dist_util.reduce_tensor(loss.detach(), reduction="mean").item()

        if last_loss is not None:
            self.loss_list.append(last_loss)

        return self._finalize_metrics(metric_totals, metric_counts)

    def optimize_step(self, *, should_log):
        grad_norm, param_norm = self._compute_norms()
        self.opt.step()
        metrics = {}
        if should_log:
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

    def log_step(self):
        return {
            "step": self.step + self.resume_step,
            "samples": (self.step + self.resume_step + 1) * self.global_batch,
            "lr": self.opt.param_groups[0]["lr"],
        }

    def save(self):
        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if dist_util.is_main_process():
            with open(self._checkpoint_path("model.pt"), "wb") as f:
                th.save(self._current_model_state_dict(), f)

            for rate, params in zip(self.ema_rate, self.ema_params):
                with open(self._checkpoint_path(f"model_{rate}.pt"), "wb") as f:
                    th.save(self._state_dict_for_params(params), f)

            opt_filename = f"opt{(self.step + self.resume_step):06d}.pt"
            with open(self._checkpoint_path(opt_filename), "wb") as f:
                th.save(self.opt.state_dict(), f)

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


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


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


def summarize_loss_dict(diffusion, ts, losses):
    metrics = {}
    gathered_ts = dist_util.gather_for_metrics(ts.detach()).cpu().numpy()
    for key, values in losses.items():
        metrics[key] = dist_util.reduce_tensor(
            values.mean().detach(), reduction="mean"
        ).item()

        gathered_values = dist_util.gather_for_metrics(values.detach()).cpu().numpy()
        quartile_totals = defaultdict(float)
        quartile_counts = defaultdict(int)
        for sub_t, sub_loss in zip(gathered_ts, gathered_values):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            quartile_key = f"{key}_q{quartile}"
            quartile_totals[quartile_key] += float(sub_loss)
            quartile_counts[quartile_key] += 1

        for quartile_key, total in quartile_totals.items():
            metrics[quartile_key] = total / quartile_counts[quartile_key]
    return metrics
