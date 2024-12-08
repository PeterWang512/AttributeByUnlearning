import os
import copy
import json
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import DiffusionPipeline

from loader import get_dataset
from models import get_model, print_param_info, get_optimizer
from compute_training_loss import collect_loss
from utils import LAIONVis, ExemplarVis


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


def get_synth_latent_text_embed(pipeline_ref, custom_diffusion_path, image_path, caption, batch_size, device='cuda'):
    pipeline = copy.deepcopy(pipeline_ref).to(device)
    pipeline.load_textual_inversion(custom_diffusion_path, weight_name="<new1>.bin")

    # get image transformation
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    synth_image = image_transforms(Image.open(image_path)).unsqueeze(0).to(device)
    sample_latent = pipeline.vae.encode(synth_image).latent_dist.sample()[0].detach()
    latents = sample_latent.to(device).unsqueeze(0).expand(batch_size, -1, -1, -1)

    with torch.no_grad():
        tokens = pipeline.tokenizer([caption],
                            max_length=pipeline.tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")['input_ids'].cuda()
        encoder_hidden_states = pipeline.text_encoder(tokens)[0]
        encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)

    pipeline.to('cpu')
    del pipeline
    return latents, encoder_hidden_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='results', help='directory to save results')
    parser.add_argument('--task_json', type=str, required=True, help='a json file containing the test cases')

    parser.add_argument('--sd_version', type=str, default='CompVis/stable-diffusion-v1-4', help='sd version')
    parser.add_argument('--dataroot', type=str, default='data/abc', help='data root')
    parser.add_argument('--laion_subset_size', type=int, default=100_000, help='laion subset size. Our experiments are conducted under a subset for runtime consideration.')
    parser.add_argument('--fisher_path', type=str, default='data/abc/sd_fisher.pt', help='path to fisher information')
    parser.add_argument('--weight_selection', type=str, default='cross-attn-kv', help='weight selection')

    parser.add_argument('--unlearn_lr', type=float, default=0.1, help='learning rate for unlearning')
    parser.add_argument('--unlearn_steps', type=int, default=100, help='number of unlearning steps')
    parser.add_argument('--unlearn_batch_size', type=int, default=16, help='batch size for unlearning')
    parser.add_argument('--unlearn_grad_accum_steps', type=int, default=10, help='gradient accumulation steps for unlearning')
    parser.add_argument('--loss_batch_size', type=int, default=200, help='batch size for loss calculation, use the same as the pretrain batch size for best performance')
    parser.add_argument('--loss_time_samples', type=int, default=10, help='number of time samples to average over, use the same as the pretrain time samples for best performance')
    parser.add_argument('--vae_batch_size', type=int, default=20, help='batch size for vae to extract latents')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    # load sd pipeline
    pipeline_sd = DiffusionPipeline.from_pretrained(args.sd_version)

    # load tasks
    with open(args.task_json, 'r') as f:
        tasks = json.load(f)

    # calculate small batch size
    small_bs = args.loss_batch_size // args.loss_time_samples

    # get laion dataloader
    laion_dataset = get_dataset('laion', dataroot=args.dataroot, subset_size=args.laion_subset_size, mode='no_flip_and_flip')
    loader_laion = torch.utils.data.DataLoader(
        laion_dataset,
        batch_size=small_bs,
        shuffle=False,
        num_workers=0
    )
    len_dataset = 400_000_000    # hardcoded for LAION-400M

    # load fisher information
    fisher_info = torch.load(args.fisher_path, map_location=args.device)

    for t in tasks:
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
        model.to(args.device)

        # get parameter to update
        param_pattern_list = get_param_pattern_list(args.weight_selection)
        optimizer, param_names_to_optimize = get_optimizer(lr=0, model=model, optimizer_name='SGD', param_pattern_list=param_pattern_list)
        print_param_info(model)

        latents, conds = get_synth_latent_text_embed(
            pipeline_sd,
            f"{args.dataroot}/{t['model_path']}",
            f"{args.dataroot}/{t['synth_image_path']}",
            t['prompt'],
            args.unlearn_batch_size,
            device=args.device
        )

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
        with torch.no_grad():
            exemplar_loss = collect_loss(
                model,
                loader_exemplar,
                noise_scheduler,
                batch_size=args.loss_batch_size,
                time_samples=args.loss_time_samples,
                avg_timesteps=True,
                init_random_seed=0,
                device=args.device
            )
            laion_loss = collect_loss(
                model,
                loader_laion,
                noise_scheduler,
                batch_size=args.loss_batch_size,
                time_samples=args.loss_time_samples,
                avg_timesteps=True,
                init_random_seed=0,
                device=args.device
            )

        # save loss
        synth_folder = t['synth_image_path'].replace('synth/', '')
        save_path = f'{args.result_dir}/influence/{synth_folder}'
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'exemplar_loss.npy'), exemplar_loss)
        np.save(os.path.join(save_path, 'laion_loss.npy'), laion_loss)

        # get influence
        model_folder = t['model_path'].replace('models/', '')
        orig_exemplar_loss = np.load(f'{args.result_dir}/orig_loss/{model_folder}/exemplar_loss.npy')
        orig_laion_loss = np.load(f'{args.result_dir}/orig_loss/{model_folder}/laion_loss.npy')

        laion_influence = laion_loss - orig_laion_loss
        laion_len = len(laion_influence)
        laion_influence = np.maximum(laion_influence[:laion_len // 2], laion_influence[laion_len // 2:])
        exemplar_influence = exemplar_loss - orig_exemplar_loss
        exemplar_len = len(exemplar_influence)
        exemplar_influence = np.maximum(exemplar_influence[:exemplar_len // 2], exemplar_influence[exemplar_len // 2:])

        influence = np.concatenate([exemplar_influence, laion_influence])
        rank = np.argsort(-influence)

        # save results
        results = {
            'influence': influence,
            'rank': rank,
        }
        with open(os.path.join(save_path, 'influence.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # visualize top 10 influential images
        plt.figure(figsize=(20, 2))
        plt.subplot(1, 11, 1)
        plt.imshow(plt.imread(f"{args.dataroot}/{t['synth_image_path']}"))
        plt.title('Query')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01)

        laion_vis = LAIONVis(path=f'{args.dataroot}/laion_subset')
        exemplar_vis = ExemplarVis(exemplar_paths=exemplar_dataset.exemplar_paths)
        top_10_indices = results['rank'][:10]
        for i, idx in enumerate(top_10_indices):
            plt.subplot(1, 11, i + 2)

            # determine whether this idx is from exemplar or laion
            if idx < len(exemplar_dataset.exemplar_paths):
                img = exemplar_vis[idx]
                plt.imshow(np.asarray(img))
                # add red border with thickness 3 if it is from exemplar
                plt.gca().add_patch(plt.Rectangle((0, 0), img.size[0], img.size[1], fill=False, edgecolor='red', linewidth=3))
            else:
                img = laion_vis[idx - len(exemplar_dataset.exemplar_paths)][0]
                plt.imshow(np.asarray(img))

            plt.title(f'infl: {results["influence"][idx]:.2e}')
            plt.axis('off')

        plt.savefig(os.path.join(save_path, f'visualization.jpg'))
        plt.close()
        print('Results saved.')
