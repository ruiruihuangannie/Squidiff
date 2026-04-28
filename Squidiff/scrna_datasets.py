from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

try:
    import muon as mu
except ImportError:  # pragma: no cover - optional dependency at runtime
    mu = None


@dataclass
class ResolvedDataSpec:
    rna_dim: int
    atac_dim: int | None
    rna_feature_names: list[str]
    atac_feature_names: list[str] | None

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    adopted from PRnet @Author: Xiaoning Qi.
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

class AnnDataDataset(Dataset):
    def __init__(
        self,
        adata,
        control_adata=None,
        use_drug_structure=False,
        comb_num=1,
    ):
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
                self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.feature_dim = self.features.shape[1]
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
                
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.use_drug_structure:
            return {
                'feature':self.features[idx],
                'drug_dose':self.encode_drug_doses[idx],
                'group': self.encoded_obs_tensor[idx],
                'control_feature':self.control_features[idx],
            }
        else:
            return {'feature':self.features[idx], 'group': self.encoded_obs_tensor[idx]}


def _obs_group_values(adata):
    if "Group" in adata.obs.columns:
        return adata.obs["Group"].copy().values
    return np.zeros((adata.n_obs, 1), dtype=np.float32)


def _row_to_dense(matrix, idx):
    row = matrix[idx]
    if sparse.issparse(row):
        row = row.toarray().squeeze(0)
    return np.asarray(row, dtype=np.float32)


class PairedAnnDataDataset(Dataset):
    def __init__(self, rna_adata, atac_adata):
        common = rna_adata.obs_names.intersection(atac_adata.obs_names)
        if len(common) == 0:
            raise ValueError("RNA and ATAC AnnData objects do not share any cells.")

        self.rna_adata = rna_adata[common].copy()
        self.atac_adata = atac_adata[common].copy()
        if self.rna_adata.n_obs != self.atac_adata.n_obs:
            raise ValueError("Paired RNA and ATAC data must have the same number of cells.")
        if not np.all(self.rna_adata.obs_names == self.atac_adata.obs_names):
            raise ValueError("RNA and ATAC cell ordering does not match after intersection.")

        self.group = _obs_group_values(self.rna_adata)
        self.feature_dim = self.rna_adata.n_vars
        self.atac_feature_dim = self.atac_adata.n_vars

    def __len__(self):
        return self.rna_adata.n_obs

    def __getitem__(self, idx):
        return {
            "feature": torch.from_numpy(_row_to_dense(self.rna_adata.X, idx)),
            "atac_feature": torch.from_numpy(_row_to_dense(self.atac_adata.X, idx)),
            "group": self.group[idx],
        }


def _select_top_hvg(adata, gene_size):
    if gene_size is None:
        return adata

    gene_size = int(gene_size)
    if gene_size <= 0:
        raise ValueError("gene_size must be positive when provided.")
    if gene_size > adata.n_vars:
        raise ValueError(
            f"Requested gene_size={gene_size}, but RNA input only has {adata.n_vars} genes."
        )

    adata = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=gene_size, flavor="seurat")
    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def _subset_to_rna_features(adata, reference_var_names):
    missing = [name for name in reference_var_names if name not in adata.var_names]
    if missing:
        raise ValueError(
            "Control RNA data is missing genes required by the processed training RNA matrix."
        )
    return adata[:, reference_var_names].copy()


def _subset_to_features(adata, reference_var_names, *, modality_name):
    missing = [name for name in reference_var_names if name not in adata.var_names]
    if missing:
        raise ValueError(
            f"{modality_name} data is missing features required by the processed training matrix."
        )
    return adata[:, reference_var_names].copy()


def _load_data_object(data_path):
    suffix = Path(data_path).suffix.lower()
    if suffix == ".h5ad":
        adata = sc.read_h5ad(data_path)
        if _looks_like_legacy_multiomics_h5ad(adata):
            raise ValueError(
                "Legacy concatenated multiomics .h5ad input is no longer supported. "
                "Please convert this dataset to a MuData .h5mu file with 'rna' and "
                "'atac' modalities."
            )
        return suffix, adata
    if suffix == ".h5mu":
        if mu is None:
            raise ImportError(
                "Loading .h5mu files requires the optional 'muon' dependency."
            )
        return suffix, mu.read_h5mu(data_path)
    raise ValueError(
        f"Unsupported dataset file type '{suffix}'. Expected .h5ad for RNA-only "
        "or .h5mu for multiomics."
    )


def _looks_like_legacy_multiomics_h5ad(adata):
    if "modality_dims" in adata.uns:
        return True
    if "modality" in adata.var.columns:
        modality_values = set(adata.var["modality"].astype(str))
        if {"rna", "atac"}.issubset(modality_values):
            return True
    preview = [str(name) for name in adata.var_names[:20]]
    has_rna_prefix = any(name.startswith("rna::") for name in preview)
    has_atac_prefix = any(name.startswith("atac::") for name in preview)
    return has_rna_prefix or has_atac_prefix


def _extract_rna_adata(data_obj):
    data_type, loaded = data_obj
    if data_type == ".h5ad":
        return loaded.copy()
    if "rna" not in loaded.mod:
        raise ValueError("MuData input is missing the required 'rna' modality.")
    return loaded["rna"].copy()


def _extract_atac_adata(data_obj):
    data_type, loaded = data_obj
    if data_type != ".h5mu":
        raise ValueError(
            "rna_only=False requires a multi-omics .h5mu input with 'rna' and 'atac' modalities."
        )
    if "atac" not in loaded.mod:
        raise ValueError("MuData input is missing the required 'atac' modality.")
    return loaded["atac"].copy()


def _load_rna_adata(data_obj, gene_size, rna_feature_names=None):
    rna_adata = _extract_rna_adata(data_obj)
    if rna_feature_names is not None:
        return _subset_to_features(rna_adata, rna_feature_names, modality_name="RNA")
    return _select_top_hvg(rna_adata, gene_size)


def _load_atac_adata(data_obj, atac_feature_names=None):
    atac_adata = _extract_atac_adata(data_obj)
    if atac_feature_names is not None:
        return _subset_to_features(atac_adata, atac_feature_names, modality_name="ATAC")
    return atac_adata


def _build_multiomics_dataset(data_obj, gene_size, rna_feature_names=None, atac_feature_names=None):
    rna_adata = _load_rna_adata(data_obj, gene_size, rna_feature_names=rna_feature_names)
    atac_adata = _load_atac_adata(data_obj, atac_feature_names=atac_feature_names)
    dataset = PairedAnnDataDataset(rna_adata, atac_adata)
    return dataset, ResolvedDataSpec(
        rna_dim=dataset.feature_dim,
        atac_dim=dataset.atac_feature_dim,
        rna_feature_names=rna_adata.var_names.to_list(),
        atac_feature_names=atac_adata.var_names.to_list(),
    )


def _build_singleomics_dataset(
    data_obj,
    control_data_dir,
    use_drug_structure,
    comb_num,
    gene_size,
    rna_feature_names=None,
):
    train_adata = _load_rna_adata(data_obj, gene_size, rna_feature_names=rna_feature_names)
    if use_drug_structure:
        control_adata = sc.read_h5ad(control_data_dir)
        control_adata = _subset_to_rna_features(control_adata, train_adata.var_names)
    else:
        control_adata = None
    dataset = AnnDataDataset(
        train_adata,
        control_adata,
        use_drug_structure,
        comb_num,
    )
    return dataset, ResolvedDataSpec(
        rna_dim=dataset.feature_dim,
        atac_dim=None,
        rna_feature_names=train_adata.var_names.to_list(),
        atac_feature_names=None,
    )


def _build_dataloader(dataset, batch_size, *, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def prepared_data(
    data_path=None,
    control_data_dir=None,
    batch_size=64,
    use_drug_structure=False,
    comb_num=1,
    rna_only=None,
    gene_size=None,
    rna_feature_names=None,
    atac_feature_names=None,
    shuffle=True,
):
    if data_path is None:
        raise ValueError("data_path is required.")
    if rna_only is None:
        raise ValueError("rna_only is required.")
    data_obj = _load_data_object(data_path)

    if not rna_only and data_obj[0] != ".h5mu":
        raise ValueError(
            "rna_only=False requires a .h5mu multiomics input. Plain .h5ad files are RNA-only."
        )

    if not rna_only:
        dataset, resolved_spec = _build_multiomics_dataset(
            data_obj,
            gene_size,
            rna_feature_names=rna_feature_names,
            atac_feature_names=atac_feature_names,
        )
    else:
        dataset, resolved_spec = _build_singleomics_dataset(
            data_obj,
            control_data_dir,
            use_drug_structure,
            comb_num,
            gene_size,
            rna_feature_names=rna_feature_names,
        )

    dataloader = _build_dataloader(dataset, batch_size, shuffle=shuffle)
    return dataloader, resolved_spec
