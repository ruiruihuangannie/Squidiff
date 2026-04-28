<img src="squidiff_logo.png" width="80" /> **squidiff: Predicting cellular development and responses to perturbations using a diffusion model**
---
Squidiff is a diffusion model-based generative framework designed to predict transcriptomic changes across diverse cell types in response to a wide range of environmental changes.

<img src=squidiff_fig.png width="1000" />

### Installation
`pip install Squidiff`

### Model Input:
- `.h5ad`: RNA-only input
- `.h5mu`: MuData multiomics input accessed via `mdata['rna']` and `mdata['atac']`
- `control_data_path` remains a separate RNA `.h5ad` when drug-structure conditioning is used

### Features 
- Predicting single-cell transcriptomics upon drug treatments 
- Predicting cell differentiation 
- Predicting gene perturbation

### Training Squidiff
```
python -m Squidiff.train_squidiff --config config/rna.yaml
python -m Squidiff.train_squidiff --config config/rna-atac.yaml
```
For incorporating drug structure in training, see the example: 
```
python -m Squidiff.train_squidiff --config path/to/config.yaml --data_path datasets/sci_plex_train_random_split_0.h5ad
```
### Sample Squidiff
```python
sampler = sample_squidiff.sampler(
    model_path='simu_results/model.pt'
)

test_adata_scrna = sc.read_h5ad('datasets/sc_simu_test.h5ad')
test_adata_scrna = sampler.align_rna_adata(test_adata_scrna)
z_sem_scrna = sampler.model.encoder(torch.tensor(test_adata_scrna.X).to('cuda'))

scrnas_pred = sampler.pred(z_sem_scrna)
```
`gene_size` is fixed by the checkpoint metadata. Do not pass the current dataset
dimension into `sampler.pred()`; align evaluation RNA features with
`sampler.align_rna_adata()` first.

For multi-GPU sampling, launch your own inference script with `accelerate launch --multi_gpu ...`; the batch-based sampler methods such as `pred()` will shard work across processes and gather the outputs back.

### Demo
Please forward to https://github.com/siyuh/Squidiff_reproducibility for data preparation, model usage, and downstream analysis.

### How to cite Squidiff

Please cite:
```
He, S., Zhu, Y., Tavakol, D.N. et al. Squidiff: predicting cellular development and responses to perturbations using a diffusion model. Nat Methods (2025). https://doi.org/10.1038/s41592-025-02877-y
```
```
Predicting cellular responses with conditional diffusion models. Nat Methods (2025). https://doi.org/10.1038/s41592-025-02878-x
```
## Contact
In case you have questions, please contact:
- Siyu He - siyuhe@stanford.edu
- via Github Issues
