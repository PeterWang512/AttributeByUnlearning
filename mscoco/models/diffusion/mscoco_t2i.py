import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, DDPMScheduler


def create_model(sample_size=128, n_channels=4) -> UNet2DConditionModel:
    sample_size = sample_size // 8
    model = UNet2DConditionModel(
        sample_size=sample_size,  # the target image resolution
        in_channels=n_channels,  # the number of input channels, 3 for RGB images
        out_channels=n_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128,
            256,
            256,
            256,
        ),  # the number of output channels for each UNet block
        cross_attention_dim=1024,  # NOTE: 1024 for V2,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        attention_head_dim=8,
    )

    return model


def load_model_weights(model, weights_path):
    if str(weights_path).endswith('.safetensors'):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=model.device)
    model.load_state_dict(state_dict)
    return model


def get_train_noise_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)
