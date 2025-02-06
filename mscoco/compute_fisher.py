import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.func import vmap
from diffusers import DDPMScheduler

from loader import get_dataset
from models import get_model


def compute_loss(model,
                 noise_scheduler,
                 weights,
                 buffers,
                 latent,
                 label,
                 timestep):
    latent = latent.unsqueeze(0)
    noise = torch.randn(latent.shape).to(latent.device)
    noisy_latent = noise_scheduler.add_noise(latent, noise, timestep)

    # MS COCO has 5 labels per image
    i = np.random.randint(len(label))
    kwargs = {
        "encoder_hidden_states": label[i].unsqueeze(0),
        "return_dict": False,
    }

    noise_pred = torch.func.functional_call(
        model, (weights, buffers), args=(noisy_latent, timestep), kwargs=kwargs
    )[0]
    return F.mse_loss(noise_pred, noise)


# defines a function that takes in data, model, and noise scheduler and outputs the gradients using vmap
def grads_loss(latents, labels, model, grads_fn, noise_scheduler, func_weights, func_buffers):
    # Sample a random timestep for each image
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz, 1), device=latents.device
    ).long()

    # compute the gradients
    gs = vmap(grads_fn,
              in_dims=(None, None, None, None, 0, 0, 0),
              randomness='different')(model,
                                      noise_scheduler,
                                      func_weights,
                                      func_buffers,
                                      latents,
                                      labels,
                                      timesteps)

    return gs


# iterate over the dataset and compute the gradients for each image to get the fisher information
def estimate_fisher(model, noise_scheduler, data_loader, num_epochs=5):
    func_weights = dict(model.named_parameters())
    func_buffers = dict(model.named_buffers())
    grads_fn = torch.func.grad(compute_loss, has_aux=False, argnums=2)

    sample_iter = 0
    fisher_accum = None
    accum_count = 0
    print("Gathering fisher information...")
    for epoch in range(num_epochs):
        pbar = tqdm(data_loader)
        for batch in pbar:
            latents, states = batch
            current_bs = len(latents)
            sample_iter += current_bs

            latents = latents.cuda()
            states = states.cuda()

            # compute the gradients
            grad = grads_loss(latents, states, model, grads_fn, noise_scheduler, func_weights, func_buffers)
            with torch.no_grad():
                fisher_avg = {key: grad[key].double().square().mean(dim=0).detach() for key in grad.keys()}
                if fisher_accum is None:
                    fisher_accum = fisher_avg
                else:
                    for key in fisher_accum.keys():
                        fisher_accum[key] += fisher_avg[key]
            accum_count += 1
            pbar.set_description(f"Epoch {epoch + 1} / {num_epochs}")
        pbar.close()

    # average the fisher information
    with torch.no_grad():
        fisher_diagonals = {key: (fisher_accum[key] / accum_count) for key in fisher_accum.keys()}

    print("Fisher info estimation done...")
    return fisher_diagonals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='output path')
    parser.add_argument('--task', type=str, default='mscoco_t2i', help='task name')
    parser.add_argument('--dataroot', type=str, default='data/mscoco', help='data root')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_path', type=str, default='data/mscoco/model.bin', help='model path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    # to prevent:
    # "RuntimeError: Batching rule not implemented for
    # aten::_chunk_grad_outputs_efficient_attention. We could not generate a
    # fallback."
    # when running with functorch, we need to disable memory-efficient SDP

    # uncomment when running fisher matrix
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # load model and noise scheduler
    model, noise_scheduler = get_model(args.task, model_path=args.model_path)
    model.to(args.device).eval()

    # load data loader
    dataset = get_dataset(args.task, dataroot=args.dataroot, split='train', mode='no_flip_and_flip')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # collect fisher information
    fisher_matrix = estimate_fisher(model, noise_scheduler, data_loader, num_epochs=5)

    # save fisher information
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(fisher_matrix, args.output_path)
