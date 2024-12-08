import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loader import get_dataset
from models import get_model


def collect_loss(model, data_loader, noise_scheduler, batch_size=200, time_samples=10, avg_timesteps=False, init_random_seed=0, device='cuda'):
    """
    Collect DDPM loss for each training sample.
    IMPORTANT: Fix batch_size, time_samples, and init_random_seed to retain same random noise for each train point.
    :param model: DDPM model
    :param data_loader: data loader
    :param noise_scheduler: noise scheduler
    :param batch_size: batch size
    :param time_samples: number to uniform subsample the timesteps
    :param avg_timesteps: whether to average over timesteps
    :param random_seed: random seed
    :return: loss for each training sample (shape: [num_train, time_samples] or [num_train])
    """
    assert batch_size % time_samples == 0
    small_bs = batch_size // time_samples
    assert data_loader.batch_size == small_bs

    # precompute timesteps needed for forward pass
    total_timesteps = noise_scheduler.num_train_timesteps
    stride = total_timesteps // time_samples
    timesteps = torch.arange(0, total_timesteps, stride, device=device)
    timesteps = timesteps.unsqueeze(0).expand(small_bs, -1).reshape(-1)

    losses = []
    batch_id = 0
    for batch in tqdm(data_loader):
        latents, hidden_encoder_states = batch
        latents = latents.to(device)
        current_bs = latents.shape[0]
        latents = latents.unsqueeze(1).expand(-1, time_samples, -1, -1, -1).reshape(-1, *latents.shape[-3:])

        hidden_encoder_states = hidden_encoder_states.to(device)
        hidden_encoder_states = hidden_encoder_states.unsqueeze(1).expand(-1, time_samples, -1, -1).reshape(-1, *hidden_encoder_states.shape[-2:])

        noise_seed = init_random_seed + batch_id
        generator = torch.Generator(device='cuda')
        generator.manual_seed(noise_seed)

        with torch.no_grad():
            noise = torch.randn(latents.shape, device='cuda', generator=generator)
            noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps[:latents.shape[0]])
        
            noise_pred = model(noisy_latent, timesteps[:latents.shape[0]], hidden_encoder_states, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=[1, 2, 3])
            loss = loss.reshape(current_bs, time_samples)
            if avg_timesteps:
                loss = loss.mean(axis=1)
            loss = loss.cpu().numpy()
        losses.append(loss)

        batch_id += 1

    losses = np.concatenate(losses, axis=0)
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='results', help='result directory')
    parser.add_argument('--task_json', type=str, required=True, help='a json file containing the test cases')
    parser.add_argument('--sd_version', type=str, default='CompVis/stable-diffusion-v1-4', help='sd version')
    parser.add_argument('--dataroot', type=str, default='data/abc', help='data root')
    parser.add_argument('--laion_subset_size', type=int, default=100_000, help='laion subset size. Our experiments are conducted under a subset for runtime consideration.')
    parser.add_argument('--vae_batch_size', type=int, default=20, help='batch size for vae to extract latents')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--time_samples', type=int, default=10, help='number of time samples to average over')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    # load tasks
    with open(args.task_json, 'r') as f:
        tasks = json.load(f)

    # calculate small batch size
    small_bs = args.batch_size // args.time_samples

    # get laion dataloader
    laion_dataset = get_dataset('laion', dataroot=args.dataroot, subset_size=args.laion_subset_size, mode='no_flip_and_flip')
    loader_laion = torch.utils.data.DataLoader(
        laion_dataset,
        batch_size=small_bs,
        shuffle=False,
        num_workers=0
    )

    seen_models = set()
    for t in tasks:
        # skip if model is already processed
        if t['model_path'] in seen_models:
            continue
        seen_models.add(t['model_path'])

        # get exemplar dataloader
        exemplar_dataset = get_dataset(
            'exemplar',
            sd_version=args.sd_version,
            custom_diffusion_model_path=f"{args.dataroot}/{t['model_path']}",
            test_case=t['test_case'],
            test_case_ind=t['test_case_ind'],
            train_prompt=t['train_prompt'],
            vae_batch_size=args.vae_batch_size,
            vae_device=args.device,
            dataroot=args.dataroot,
            mode='no_flip_and_flip'
        )
        loader_exemplar = torch.utils.data.DataLoader(
            exemplar_dataset,
            batch_size=small_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        torch.cuda.empty_cache()

        # load model and noise scheduler
        model, noise_scheduler = get_model(
            'custom_diffusion',
            model_path=f"{args.dataroot}/{t['model_path']}",
            sd_version=args.sd_version
        )
        model.to(args.device).eval()

        # collect loss
        with torch.no_grad():
            exemplar_loss = collect_loss(
                model,
                loader_exemplar,
                noise_scheduler,
                batch_size=args.batch_size,
                time_samples=args.time_samples,
                avg_timesteps=True,
                init_random_seed=0,
                device=args.device
            )
            laion_loss = collect_loss(
                model,
                loader_laion,
                noise_scheduler,
                batch_size=args.batch_size,
                time_samples=args.time_samples,
                avg_timesteps=True,
                init_random_seed=0,
                device=args.device
            )

        # save loss
        model_folder = t['model_path'].replace('models/', '')
        save_dir = f'{args.result_dir}/orig_loss/{model_folder}'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/exemplar_loss.npy', exemplar_loss)
        np.save(f'{save_dir}/laion_loss.npy', laion_loss)
