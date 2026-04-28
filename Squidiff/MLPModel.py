from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, z_sem):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
        

class MLPBlock(TimestepBlock):
    """
    Basic MLP block with an optional timestep embedding.
    """

    def __init__(self, input_dim, output_dim, time_embed_dim=None, latent_dim = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_dense = nn.Linear(time_embed_dim, output_dim)
        if latent_dim is not None:
            self.zsem_dense = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x, emb, z_sem):
        
        h = F.silu(self.layer_norm1(self.fc1(x))) 
        if ((emb is not None)&(z_sem is None)):
            h = h + self.time_dense(emb)
        elif ((emb is not None)&(z_sem is not None)):
            h = h + self.time_dense(emb)+self.zsem_dense(z_sem)
        h = F.silu(self.layer_norm2(self.fc2(h)))
        return h
    
    


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, z_sem):
        
        for layer in self:
            
            
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, z_sem)
            else:
                x = layer(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PairedEncoderMLPModel(nn.Module):
    def __init__(
        self,
        rna_input_size,
        atac_input_size,
        latent_dim=128,
        hidden_rna=(1024, 512),
        hidden_atac=(1024, 512),
        dropout=0.2,
        use_fp16=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dtype = th.float16 if use_fp16 else th.float32
        self.rna_encoder = SimpleMLP(
            rna_input_size, latent_dim, list(hidden_rna), dropout=dropout
        )
        self.atac_encoder = SimpleMLP(
            atac_input_size, latent_dim, list(hidden_atac), dropout=dropout
        )
        self.joint_projection = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x_rna, x_atac):
        x_rna = x_rna.type(self.dtype)
        x_atac = x_atac.type(self.dtype)
        z_rna = self.rna_encoder(x_rna)
        z_atac = self.atac_encoder(x_atac)
        return self.joint_projection(th.cat([z_rna, z_atac], dim=1))

class MLPModel(nn.Module):
    """
    MLP model for single-cell RNA-seq data with timestep embedding.
    """

    def __init__(self, 
                 gene_size, 
                 num_layers, 
                 loss_type="mse",
                 hidden_sizes=2048,
                 time_pos_dim=2048,
                 num_classes = None,
                 latent_dim=60,
                 use_checkpoint=False,
                 use_fp16 = False,
                 use_scale_shift_norm =False,
                 dropout=0,
                 time_embed_dim=2048,
                 use_encoder=False,
                 use_drug_structure = False,
                 drug_dimension = 1024,
                 comb_num=1,
                 atac_input_size=None,
                 paired_latent_dim=128,
                 hidden_rna=(1024, 512),
                 hidden_atac=(1024, 512),
                 paired_dropout=0.2,
                 gmm_num_components=16,
                ):
        super().__init__()
        
        self.use_encoder = use_encoder
        self.loss_type = loss_type
        self.time_embed_dim = time_embed_dim
        self.has_atac_encoder = atac_input_size is not None
        self.latent_dim = paired_latent_dim if self.has_atac_encoder else latent_dim
        self.use_gmm_prior = loss_type == "mse-gmm"
        self.gmm_num_components = int(gmm_num_components)
        self.gmm_min_logvar = math.log(1e-4)
        if self.use_gmm_prior and not use_encoder:
            raise ValueError("loss_type='mse-gmm' requires use_encoder=True.")
        if self.gmm_num_components <= 0:
            raise ValueError("gmm_num_components must be positive.")
        self.time_embed = None
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.drug_dimension = drug_dimension
        if use_encoder:
            if self.has_atac_encoder:
                self.encoder = PairedEncoderMLPModel(
                    rna_input_size=gene_size,
                    atac_input_size=atac_input_size,
                    latent_dim=self.latent_dim,
                    hidden_rna=hidden_rna,
                    hidden_atac=hidden_atac,
                    dropout=paired_dropout,
                    use_fp16=use_fp16,
                )
            else:
                self.encoder = EncoderMLPModel(
                    gene_size,
                    self.hidden_sizes,
                    self.num_classes,
                    use_drug_structure,
                    self.drug_dimension,
                    comb_num,
                    output_size=self.latent_dim,
                    use_fp16=use_fp16,
                )
        
        if self.use_gmm_prior:
            self.gmm_means = nn.Parameter(
                th.randn(self.gmm_num_components, self.latent_dim) * 0.05
            )
            self.gmm_logvars = nn.Parameter(
                th.zeros(self.gmm_num_components, self.latent_dim)
            )
            self.gmm_logits = nn.Parameter(th.zeros(self.gmm_num_components))
        
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(time_pos_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.use_encoder: 
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim, self.latent_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        else:
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        
        self.input_layer = nn.Linear(gene_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, gene_size)
        
    def encode_condition(self, model_kwargs):
        if not self.use_encoder:
            return None
        if 'z_mod' in model_kwargs.keys():
            return model_kwargs['z_mod']
        if self.has_atac_encoder:
            atac_feature = model_kwargs.get("atac_feature")
            if atac_feature is None:
                raise ValueError("atac_feature is required for multi-omics conditioning.")
            return self.encoder(model_kwargs["x_start"], atac_feature)
        if self.num_classes is None:
            return self.encoder(
                model_kwargs['x_start'],
                label=None,
                drug_dose=model_kwargs['drug_dose'],
                control_feature=model_kwargs['control_feature'],
            )
        return self.encoder(
            model_kwargs['x_start'],
            label=model_kwargs['group'],
            drug_dose=model_kwargs['drug_dose'],
            control_feature=model_kwargs['control_feature'],
        )

    def gmm_nll(self, z_sem):
        if not self.use_gmm_prior:
            raise ValueError("gmm_nll is only available when loss_type='mse-gmm'.")

        z_sem = z_sem.float()
        means = self.gmm_means.float()
        logvars = self.gmm_logvars.float().clamp(min=self.gmm_min_logvar)
        log_weights = F.log_softmax(self.gmm_logits.float(), dim=0)

        diff = z_sem[:, None, :] - means[None, :, :]
        log_probs = -0.5 * (
            (diff ** 2) * th.exp(-logvars)[None, :, :]
            + logvars[None, :, :]
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return -th.logsumexp(log_probs + log_weights[None, :], dim=1)

    def forward(self, x, timesteps=None, **model_kwargs):
        
        
        if self.time_embed is not None and timesteps is not None:
            
            emb = self.time_embed(timestep_embedding(timesteps, self.hidden_sizes))
            
        else:
            emb = None
            
        if self.use_encoder: 
            z_sem = self.encode_condition(model_kwargs)

            h = self.input_layer(x)
            
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        else:
            z_sem = None
            h = self.input_layer(x)
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        return h

    
class EncoderMLPModel(nn.Module):

    def __init__(self, input_size, hidden_sizes, num_classes=None, use_drug_structure=False, drug_dimension=1024,comb_num=1,output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.drug_dimension = drug_dimension
    
        if num_classes is None:
            l1 = 0
        else: 
            l1 = hidden_sizes
        if use_drug_structure:
            l2 = drug_dimension
        else:
            l2 = 0
        
        self.fc1 = nn.Linear(input_size+l1+l2, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
       
        self.label_embed = nn.Linear(1, hidden_sizes)
    
    def forward(self, x_start, label=None, drug_dose=None, control_feature = None):
        
        if label is not None:
            label_emb = self.label_embed(label)
            x_start = th.concat([x_start,label_emb],axis=1)
        
        if drug_dose is not None:
            x_start = th.concat([control_feature,drug_dose],axis=1)
            
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h
    

    
class EncoderMLPModel2(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=None, output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel2, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.dtype = th.float16 if use_fp16 else th.float32
        
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.label_embed = nn.Linear(1, hidden_sizes)  

    def forward(self, x_start, label=None):
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
    

        if label is not None:
            label = label.type(self.dtype) 
            label_emb = self.label_embed(label)
            
            h = h + label_emb  # Add label embedding as a residual connection


        h = self.dropout_layer(h)
        h = self.fc3(h)
       
        return h
