import argparse
import inspect

from . import diffusion
from .respace import SpacedDiffusion, space_timesteps
from .MLPModel import MLPModel, EncoderMLPModel

NUM_CLASSES = 4


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        loss_type="mse",
        alpha=0.1,
        predict_xstart=False,
        rescale_timesteps=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        gene_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for training.
    """
    res = dict(
        num_layers=3,
        gene_size=None,
        num_channels=128,
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_encoder=False,
        use_drug_structure=False,
        drug_dimension=1024,
        comb_num=1,
        atac_input_size=None,
        paired_latent_dim=128,
        hidden_rna="1024,512",
        hidden_atac="1024,512",
        paired_dropout=0.2,
        gmm_num_components=16,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    gene_size,
    num_layers,
    class_cond,
    learn_sigma,
    num_channels,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    loss_type,
    alpha,
    predict_xstart,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
    use_fp16,
    use_encoder,
    use_drug_structure,
    drug_dimension,
    comb_num,
    atac_input_size,
    paired_latent_dim,
    hidden_rna,
    hidden_atac,
    paired_dropout,
    gmm_num_components,
):
    model = create_model(
        gene_size,
        num_layers,
        loss_type=loss_type,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_fp16=use_fp16,
        use_encoder=use_encoder,
        use_drug_structure = use_drug_structure,
        drug_dimension = drug_dimension,
        comb_num=comb_num,
        atac_input_size=atac_input_size,
        paired_latent_dim=paired_latent_dim,
        hidden_rna=hidden_rna,
        hidden_atac=hidden_atac,
        paired_dropout=paired_dropout,
        gmm_num_components=gmm_num_components,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        loss_type=loss_type,
        alpha=alpha,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing=timestep_respacing,
        use_encoder=use_encoder
    )
    return model, diffusion


def create_model(
    gene_size,
    num_layers,
    loss_type="mse",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=False,
    dropout=0,
    use_fp16=False,
    use_encoder=False,
    use_drug_structure = False,
    drug_dimension = 1024,
    comb_num=1,
    atac_input_size=None,
    paired_latent_dim=128,
    hidden_rna="1024,512",
    hidden_atac="1024,512",
    paired_dropout=0.2,
    gmm_num_components=16,
):
    if isinstance(hidden_rna, str):
        hidden_rna = tuple(int(part) for part in hidden_rna.split(",") if part)
    if isinstance(hidden_atac, str):
        hidden_atac = tuple(int(part) for part in hidden_atac.split(",") if part)

    return MLPModel(
        gene_size  = gene_size,
        num_layers = num_layers,
        loss_type=loss_type,
        dropout=dropout,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        use_encoder = use_encoder,
        use_drug_structure = use_drug_structure,
        drug_dimension = drug_dimension,
        comb_num=comb_num,
        atac_input_size=atac_input_size,
        paired_latent_dim=paired_latent_dim,
        hidden_rna=hidden_rna,
        hidden_atac=hidden_atac,
        paired_dropout=paired_dropout,
        gmm_num_components=gmm_num_components,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    loss_type="mse",
    alpha=0.1,
    predict_xstart=False,
    rescale_timesteps=False,
    timestep_respacing="",
    use_encoder = False
):
    print('diffusion num of steps = ',steps)
    betas = diffusion.get_named_beta_schedule(noise_schedule, steps)
    use_gmm_loss = False
    if loss_type == "mse":
        diffusion_loss_type = diffusion.LossType.MSE
        use_kl_loss = False
    elif loss_type == "mse-kl":
        diffusion_loss_type = diffusion.LossType.MSE
        use_kl_loss = True
    elif loss_type == "mse-gmm":
        diffusion_loss_type = diffusion.LossType.MSE
        use_kl_loss = False
        use_gmm_loss = True
    elif loss_type == "kl":
        diffusion_loss_type = diffusion.LossType.KL
        use_kl_loss = False
    elif loss_type == "rescaled-mse":
        diffusion_loss_type = diffusion.LossType.RESCALED_MSE
        use_kl_loss = False
    else:
        raise ValueError(
            "loss_type must be one of: kl, mse, mse-kl, mse-gmm, rescaled-mse."
        )
    if not timestep_respacing:
        timestep_respacing = [steps]
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            diffusion.ModelMeanType.EPSILON if not predict_xstart else diffusion.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                diffusion.ModelVarType.FIXED_LARGE
                if not sigma_small
                else diffusion.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else diffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=diffusion_loss_type,
        rescale_timesteps=rescale_timesteps,
        use_encoder=use_encoder,
        use_kl_loss=use_kl_loss,
        kl_weight=alpha,
        use_gmm_loss=use_gmm_loss,
        gmm_weight=alpha,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: args[k] for k in keys if k in args}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
