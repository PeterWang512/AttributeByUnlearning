import os
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from loader import get_dataset
from models import get_model, print_param_info, get_optimizer
from compute_training_loss import collect_loss
from utils import COCOVis


def get_param_pattern_list(weight_selection):
    if weight_selection == 'cross-attn-kv':
        param_pattern_list = [
            ['attn2'],
            ['to_k', 'to_v'],
        ]
    elif weight_selection == 'cross-attn':
        param_pattern_list = [
            ['attn2'],
        ]
    elif weight_selection == 'attn':
        param_pattern_list = [
            ['attn'],
        ]
    elif weight_selection == 'all':
        param_pattern_list = []
    else:
        raise ValueError(f'Invalid weight selection: {weight_selection}')
    return param_pattern_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True, help='directory to save results')
    parser.add_argument('--sample_latent_path', type=str, required=True, help='path to sample latents')
    parser.add_argument('--sample_text_path', type=str, required=True, help='path to sample text')
    parser.add_argument('--sample_idx', type=int, required=True, help='sample index')
    parser.add_argument('--pretrain_loss_path', type=str, required=True, help='path to pretrain loss')

    parser.add_argument('--task', type=str, default='mscoco_t2i', help='task name')
    parser.add_argument('--dataroot', type=str, default='data/mscoco', help='data root')
    parser.add_argument('--model_path', type=str, default='data/mscoco/model.bin', help='model path')
    parser.add_argument('--fisher_path', type=str, default='data/mscoco/fisher_info.pt', help='path to fisher information')
    parser.add_argument('--weight_selection', type=str, default='cross-attn-kv', help='weight selection')

    parser.add_argument('--unlearn_lr', type=float, default=0.01, help='learning rate for unlearning')
    parser.add_argument('--unlearn_steps', type=int, default=1, help='number of unlearning steps')
    parser.add_argument('--unlearn_batch_size', type=int, default=80, help='batch size for unlearning')
    parser.add_argument('--unlearn_grad_accum_steps', type=int, default=625, help='gradient accumulation steps for unlearning')
    parser.add_argument('--loss_batch_size', type=int, default=8000, help='batch size for loss calculation, use the same as the pretrain batch size for best performance')
    parser.add_argument('--loss_time_samples', type=int, default=20, help='number of time samples to average over, use the same as the pretrain time samples for best performance')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    # load model and noise scheduler
    model, noise_scheduler = get_model(args.task, model_path=args.model_path)
    model.to(args.device)

    # get parameter to update
    param_pattern_list = get_param_pattern_list(args.weight_selection)
    optimizer, param_names_to_optimize = get_optimizer(lr=0, model=model, optimizer_name='SGD', param_pattern_list=param_pattern_list)
    print_param_info(model)

    # load data loader
    dataset = get_dataset(args.task, dataroot=args.dataroot, split='train', mode='no_flip_and_flip')
    small_bs = args.loss_batch_size // dataset.num_captions // args.loss_time_samples
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=small_bs,
        shuffle=False
    )
    len_dataset = dataset.orig_length

    # load fisher information
    fisher_info = torch.load(args.fisher_path, map_location=args.device)

    # load sample latents and text for unlearning
    latents = np.load(args.sample_latent_path)[[args.sample_idx]]
    latents = torch.from_numpy(latents).to(args.device).float().expand(args.unlearn_batch_size, -1, -1, -1)
    conds = np.load(args.sample_text_path)[[args.sample_idx], 0]
    conds = torch.from_numpy(conds).to(args.device).float().expand(args.unlearn_batch_size, -1, -1)

    # run unlearning
    print('Running unlearning...')
    for optim_step in range(args.unlearn_steps):
        optimizer.zero_grad()
        pbar = tqdm(range(args.unlearn_grad_accum_steps), desc=f'Unlearning step {optim_step + 1}/{args.unlearn_steps}')
        for step in pbar:
            # Sample noise to add to the images
            noise = torch.randn_like(latents)

            # Add noise to the latents
            bs = latents.shape[0]
            timesteps = torch.randperm(noise_scheduler.num_train_timesteps)[:bs].to(latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_output = model(noisy_latents, timesteps, conds, return_dict=False)[0]
            loss = F.mse_loss(model_output, noise)
            loss.backward()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in param_names_to_optimize:
                    update_p = args.unlearn_lr * p.grad.double() / (fisher_info[n] * len_dataset) / args.unlearn_grad_accum_steps
                    p.add_(update_p)
    optimizer.zero_grad()
    print('Unlearning done.')

    # collect loss
    print('Collecting loss...')
    with torch.no_grad():
        loss_no_flip_and_flip = collect_loss(
            model,
            data_loader,
            noise_scheduler,
            batch_size=args.loss_batch_size,
            time_samples=args.loss_time_samples,
            num_captions=dataset.num_captions,
            avg_timesteps=True,
            avg_captions=True,
            init_random_seed=0,
            device=args.device
        )
    print('Loss collected.')

    # influence calculation
    print('Calculating influence and visualize results...')
    influence = loss_no_flip_and_flip - np.load(args.pretrain_loss_path)
    assert influence.shape == (len_dataset * 2,)
    influence = np.maximum(influence[:len_dataset], influence[len_dataset:])

    # save results
    os.makedirs(args.result_dir, exist_ok=True)
    results = {
        'influence': influence,
        'rank': np.argsort(-influence),
    }
    with open(os.path.join(args.result_dir, f'influence_{args.sample_idx}.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # visualize top 10 influential images
    plt.figure(figsize=(20, 2))
    plt.subplot(1, 11, 1)
    plt.imshow(plt.imread(f'{args.dataroot}/sample/{args.sample_idx}.png'))
    plt.title('Query')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.01)

    coco_vis = COCOVis(path=args.dataroot, split='train')
    top_10_indices = results['rank'][:10]
    for i, idx in enumerate(top_10_indices):
        plt.subplot(1, 11, i + 2)
        plt.imshow(np.asarray(coco_vis[idx][0]))
        plt.title(f'infl: {results["influence"][idx]:.2e}')
        plt.axis('off')

    plt.savefig(os.path.join(args.result_dir, f'visualization_{args.sample_idx}.jpg'))
    plt.close()
    print('Results saved.')
