import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loader import get_dataset
from models import get_model


def collect_loss(model, data_loader, noise_scheduler, batch_size=8000, time_samples=20, num_captions=5, avg_timesteps=True, avg_captions=True, init_random_seed=0, device='cuda'):
    """
    Collect DDPM loss for each training sample.
    IMPORTANT: Fix batch_size, time_samples, and init_random_seed to retain same random noise for each train point.
    :param model: DDPM model
    :param data_loader: data loader
    :param noise_scheduler: noise scheduler
    :param batch_size: batch size
    :param time_samples: number to uniform subsample the timesteps
    :param num_captions: number of captions per image
    :param avg_timesteps: whether to average over timesteps
    :param avg_captions: whether to average over captions
    :param random_seed: random seed
    :return: loss for each training sample (shape: [num_train, time_samples] or [num_train])
    """
    assert batch_size % time_samples == 0
    small_bs = batch_size // time_samples // num_captions
    assert data_loader.batch_size == small_bs

    # precompute timesteps needed for forward pass
    total_timesteps = noise_scheduler.num_train_timesteps
    stride = total_timesteps // time_samples
    timesteps = torch.arange(0, total_timesteps, stride, device=device)
    timesteps = timesteps.unsqueeze(0).expand(small_bs * num_captions, -1).reshape(-1)

    losses = []
    batch_id = 0
    for batch in tqdm(data_loader):
        latents, conds = batch
        current_bs = latents.shape[0]
        latents = latents.to(device).repeat_interleave(num_captions * time_samples, dim=0)
        conds = conds.to(device).view(-1, *conds.shape[2:]).repeat_interleave(time_samples, dim=0)

        noise_seed = init_random_seed + batch_id
        generator = torch.Generator(device=device)
        generator.manual_seed(noise_seed)

        with torch.no_grad():
            noise = torch.randn(latents.shape, device=device, generator=generator)
            noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps[:latents.shape[0]])

            noise_pred = model(noisy_latent, timesteps[:latents.shape[0]], conds, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=[1, 2, 3])
            loss = loss.reshape(current_bs, num_captions, time_samples)
            if avg_timesteps:
                loss = loss.mean(axis=2)
            if avg_captions:
                loss = loss.mean(axis=1)
            loss = loss.cpu().numpy()
        losses.append(loss)

        batch_id += 1

    losses = np.concatenate(losses, axis=0)
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='output path')
    parser.add_argument('--task', type=str, default='mscoco_t2i', help='task name')
    parser.add_argument('--dataroot', type=str, default='data/mscoco', help='data root')
    parser.add_argument('--batch_size', type=int, default=8000, help='batch size')
    parser.add_argument('--model_path', type=str, default='data/mscoco/model.bin', help='model path')
    parser.add_argument('--time_samples', type=int, default=20, help='number of time samples to average over')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    # load model and noise scheduler
    model, noise_scheduler = get_model(args.task, model_path=args.model_path)
    model.to(args.device).eval()

    # load data loader
    dataset = get_dataset(args.task, dataroot=args.dataroot, split='train', mode='no_flip_and_flip')
    small_bs = args.batch_size // dataset.num_captions // args.time_samples
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=small_bs,
        shuffle=False
    )

    # collect loss
    with torch.no_grad():
        loss_no_flip_and_flip = collect_loss(
            model,
            data_loader,
            noise_scheduler,
            batch_size=args.batch_size,
            time_samples=args.time_samples,
            num_captions=dataset.num_captions,
            avg_timesteps=True,
            avg_captions=True,
            init_random_seed=0,
            device=args.device
        )

    # save loss
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, loss_no_flip_and_flip)
