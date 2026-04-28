import json
import os

from .script_util import model_and_diffusion_defaults


MODEL_SPEC_FILENAME = "model_spec.json"
RNA_FEATURES_FILENAME = "rna_features.txt"
ATAC_FEATURES_FILENAME = "atac_features.txt"
_EXTRA_SPEC_KEYS = (
    "use_ddim",
    "rna_only",
    "rna_features_file",
    "atac_features_file",
)


def build_model_spec(args):
    spec = {
        key: args.get(key)
        for key in (*model_and_diffusion_defaults().keys(), *_EXTRA_SPEC_KEYS)
    }
    spec["spec_version"] = 1
    return spec


def save_model_spec(spec, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, MODEL_SPEC_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)


def _checkpoint_dir(path):
    if path.endswith(".pt"):
        return os.path.dirname(path)
    return path


def load_model_spec(path):
    checkpoint_dir = _checkpoint_dir(path)
    spec_path = os.path.join(checkpoint_dir, MODEL_SPEC_FILENAME)
    if not os.path.exists(spec_path):
        raise ValueError(
            "Checkpoint metadata is missing. This checkpoint predates model_spec "
            "persistence and cannot be safely loaded."
        )
    with open(spec_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_rna_features(feature_names, checkpoint_dir, filename=RNA_FEATURES_FILENAME):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        for feature_name in feature_names:
            handle.write(f"{feature_name}\n")
    return path


def load_rna_features(path, filename=None):
    checkpoint_dir = _checkpoint_dir(path)
    feature_filename = filename or RNA_FEATURES_FILENAME
    feature_path = os.path.join(checkpoint_dir, feature_filename)
    if not os.path.exists(feature_path):
        raise ValueError(
            "RNA feature metadata is missing. This checkpoint predates RNA feature "
            "persistence and cannot be safely aligned for evaluation."
        )
    with open(feature_path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def save_atac_features(feature_names, checkpoint_dir, filename=ATAC_FEATURES_FILENAME):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        for feature_name in feature_names:
            handle.write(f"{feature_name}\n")
    return path


def load_atac_features(path, filename=None):
    checkpoint_dir = _checkpoint_dir(path)
    feature_filename = filename or ATAC_FEATURES_FILENAME
    feature_path = os.path.join(checkpoint_dir, feature_filename)
    if not os.path.exists(feature_path):
        raise ValueError(
            "ATAC feature metadata is missing. This multiomics checkpoint cannot "
            "be safely aligned for evaluation."
        )
    with open(feature_path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]
