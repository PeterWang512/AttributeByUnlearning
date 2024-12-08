import re
import torch
import torch.optim as optim

from utils import find_and_import_module, check_substrings


def get_model(model_name, model_path=None, **kwargs):
    # Load model dynamically
    module = find_and_import_module("models/diffusion", model_name)

    # Load diffusion model from filename
    model = module.create_model(**kwargs)
    if model_path is not None:
        module.load_model_weights(model, model_path)

    # Get noise scheduler
    noise_scheduler = module.get_train_noise_scheduler()

    return model, noise_scheduler


def get_optimizer(lr, model, optimizer_name, param_pattern_list=[]):
    # Select parameters to optimize based on regex_pattern
    params_to_optimize = []
    param_names_to_optimize = []
    for name, param in model.named_parameters():
        if check_substrings(name, param_pattern_list):
            params_to_optimize.append(param)
            param_names_to_optimize.append(name)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Select optimizer based on optimizer_name
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(params_to_optimize, lr=lr, weight_decay=0)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=0)
    elif optimizer_name is None:
        optimizer = None
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    return optimizer, param_names_to_optimize


def print_param_info(model):
    train_param_count = 0
    all_param_count = 0
    train_param_names = []
    all_param_names = []

    # Count trainable and all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_param_names.append(name)
            train_param_count += param.numel()
        all_param_names.append(name)
        all_param_count += param.numel()

    print(
        "####################\n"
        "# Trainable params #\n"
        "####################\n",
        "\n".join(train_param_names)
    )

    print(f"All parameter count: {all_param_count}")
    print(f"Trainable parameter count: {train_param_count}")
