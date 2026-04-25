# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2026-04-19
import numpy as np
import torch
from Squidiff import dist_util
from Squidiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from sklearn.metrics import r2_score
import scipy


class sampler:
    def __init__(self,model_path,gene_size,output_dim,use_drug_structure):
        args = self.parse_args(model_path,gene_size,output_dim,use_drug_structure)
        self.accelerator = dist_util.setup_accelerator(use_fp16=args["use_fp16"])
        self.accelerator.print("load model and diffusion...")

        model, diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )

        model.load_state_dict(
            dist_util.load_state_dict(args['model_path'])
        )
        if args['use_fp16']:
            model.convert_to_fp16()
        model = dist_util.prepare(model)
        model.eval()
        self.model = model
        self.arg = args
        self.diffusion = diffusion
        self.sample_fn = (diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop)
        self.num_processes = dist_util.num_processes()
        self.process_index = dist_util.process_index()

    def _split_tensor_for_process(self, tensor):
        total = tensor.shape[0]
        per_process = (total + self.num_processes - 1) // self.num_processes
        start = self.process_index * per_process
        end = min(start + per_process, total)
        if start >= total:
            return tensor[:0], total
        return tensor[start:end], total

    def _slice_model_kwargs(self, model_kwargs, total_size):
        if model_kwargs is None:
            return None
        local_kwargs = {}
        for key, value in model_kwargs.items():
            if torch.is_tensor(value) and value.shape[0] == total_size:
                local_value, _ = self._split_tensor_for_process(value)
                local_kwargs[key] = local_value.to(dist_util.dev())
            elif torch.is_tensor(value):
                local_kwargs[key] = value.to(dist_util.dev())
            else:
                local_kwargs[key] = value
        return local_kwargs

    def _gather_tensor(self, tensor, total_size):
        if self.num_processes == 1:
            return tensor

        local_size = torch.tensor([tensor.shape[0]], device=dist_util.dev(), dtype=torch.long)
        gathered_sizes = dist_util.gather_for_metrics(local_size).cpu().tolist()
        max_size = max(gathered_sizes, default=0)

        if tensor.shape[0] < max_size:
            pad_shape = list(tensor.shape)
            pad_shape[0] = max_size - tensor.shape[0]
            padding = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)

        gathered_tensor = dist_util.gather_for_metrics(tensor)
        chunks = []
        offset = 0
        for size in gathered_sizes:
            chunks.append(gathered_tensor[offset : offset + max_size][:size])
            offset += max_size

        if not chunks:
            return tensor[:0]
        return torch.cat(chunks, dim=0)[:total_size]

    def _gather_output(self, output, total_size):
        if torch.is_tensor(output):
            return self._gather_tensor(output, total_size)
        if isinstance(output, list):
            return [self._gather_output(item, total_size) for item in output]
        if isinstance(output, dict):
            return {
                key: self._gather_output(value, total_size)
                for key, value in output.items()
            }
        return output

    def _run_distributed_batch(self, batch_tensor, run_fn, model_kwargs=None):
        local_batch, total_size = self._split_tensor_for_process(batch_tensor)
        local_batch = local_batch.to(dist_util.dev())
        local_model_kwargs = self._slice_model_kwargs(model_kwargs, total_size)

        if local_batch.shape[0] == 0:
            empty = run_fn(local_batch, local_model_kwargs, empty_batch=True)
            return self._gather_output(empty, total_size)

        local_output = run_fn(local_batch, local_model_kwargs, empty_batch=False)
        return self._gather_output(local_output, total_size)
    
    def stochastic_encode(
        self, model, x, t, model_kwargs):
        """
        ddim reverse sample
        """
        def _encode(local_x, local_kwargs, *, empty_batch):
            if empty_batch:
                empty_t = x[:0]
                return {
                    "sample": empty_t,
                    "sample_t": [],
                    "xstart_t": [],
                    "T": [],
                }

            sample = local_x
            sample_t = []
            xstart_t = []
            timesteps = []
            for i in range(t):
                timestep = torch.full((local_x.shape[0],), i, device=dist_util.dev()).long()
                with torch.no_grad():
                    out = self.diffusion.ddim_reverse_sample(
                        model,
                        sample,
                        timestep,
                        model_kwargs=local_kwargs,
                    )
                    sample = out["sample"]
                    sample_t.append(sample)
                    xstart_t.append(out["pred_xstart"])
                    timesteps.append(timestep)

            return {
                "sample": sample,
                "sample_t": sample_t,
                "xstart_t": xstart_t,
                "T": timesteps,
            }

        return self._run_distributed_batch(x, _encode, model_kwargs=model_kwargs)

    def parse_args(self,model_path,gene_size,output_dim,use_drug_structure):
        """Parse command-line arguments and update with default values."""
        # Define default arguments
        default_args = {}
        default_args.update(model_and_diffusion_defaults())
        updated_args = {
            'data_path': '',
            'schedule_sampler': 'uniform',
            'lr': 1e-4,
            'weight_decay': 0.0,
            'lr_anneal_steps': 1e5,
            'batch_size': 16,
            'microbatch': -1,
            'ema_rate': '0.9999',
            'log_interval': 1e4,
            'save_interval': 1e4,
            'resume_checkpoint': '',
            'use_fp16': False,
            'fp16_scale_growth': 1e-3,
            'gene_size': gene_size,
            'output_dim': output_dim,
            'num_layers': 3,
            'class_cond': False,
            'use_encoder': True,
            'use_ddim':True,
            'diffusion_steps': 1000,
            'logger_path': '',
            'model_path': model_path,
            'use_drug_structure':use_drug_structure,
            'comb_num':1,
            'drug_dimension':1024
        }
        default_args.update(updated_args)

        # Return the updated arguments as a dictionary
        return default_args

    def load_squidiff_model(self):
        self.accelerator.print("load model and diffusion...")
        return self.model

    def load_sample_fn(self):
        
        return self.sample_fn

    def get_diffused_data(self,model, x, t, model_kwargs):
        def _diffuse(local_x, _local_kwargs, *, empty_batch):
            if empty_batch:
                empty_t = x[:0]
                return {
                    "sample": empty_t,
                    "sample_t": [empty_t],
                    "xstart_t": [],
                    "T": [],
                }

            sample = local_x
            sample_t = [local_x]
            xstart_t = []
            timesteps = []

            for i in range(t):
                timestep = torch.full((local_x.shape[0],), i, device=dist_util.dev()).long()
                with torch.no_grad():
                    noise = torch.randn_like(sample)
                    sample = sample + noise * (i / t)
                    sample_t.append(sample)
                    xstart_t.append(sample)
                    timesteps.append(timestep)

            return {
                "sample": sample,
                "sample_t": sample_t,
                "xstart_t": xstart_t,
                "T": timesteps,
            }

        return self._run_distributed_batch(x, _diffuse, model_kwargs=model_kwargs)

    def sample_around_point(self, point, num_samples=None, scale=0.7):
        return point + scale * np.random.randn(num_samples, point.shape[0])

    def pred(self,z_sem, gene_size):
        def _pred(local_z_sem, _local_kwargs, *, empty_batch):
            if empty_batch:
                return z_sem.new_empty((0, gene_size))

            return self.sample_fn(
                self.model,
                shape=(local_z_sem.shape[0], gene_size),
                model_kwargs={"z_mod": local_z_sem},
                noise=None,
            )

        return self._run_distributed_batch(z_sem, _pred)
    
    def interp_with_direction(self, z_sem_origin = None, gene_size = None, direction = None, scale = 1, add_noise_term = True):

        z_sem_origin = z_sem_origin.detach().cpu().numpy()
        z_sem_interp_ = z_sem_origin.mean(axis=0) + direction.detach().cpu().numpy() * scale
        if add_noise_term:
            z_sem_interp_ = self.sample_around_point(z_sem_interp_, num_samples=z_sem_origin.shape[0])

        z_sem_interp_ = torch.tensor(z_sem_interp_,dtype=torch.float32).to(dist_util.dev())
        return self.pred(z_sem_interp_, gene_size)
        
    def cal_metric(self,x1,x2):
        r2 = r2_score(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        pearsonr,_ = scipy.stats.pearsonr(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        return r2, pearsonr

        
