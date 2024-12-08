import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, DDPMScheduler


def create_model(sd_version="CompVis/stable-diffusion-v1-4") -> UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(sd_version, subfolder="unet")
    unet.enable_xformers_memory_efficient_attention()
    return unet


def load_model_weights(model, weights_path):
    custom_diffusion_state_dict = torch.load(f"{weights_path}/pytorch_custom_diffusion_weights.pt", map_location='cpu')
    model.load_state_dict(custom_diffusion_state_dict, strict=False)
    return model


def get_train_noise_scheduler(sd_version="CompVis/stable-diffusion-v1-4"):
    return DDPMScheduler.from_pretrained(sd_version, subfolder="scheduler")
