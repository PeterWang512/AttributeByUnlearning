import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loader import get_dataset
from models import get_model
from utils import T2IWrapper


torch.set_grad_enabled(False)
torch.backends.cudnn.deterministic = True  # for deterministic behavior


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True, help='output path')
    parser.add_argument('--task', type=str, default='mscoco_t2i', help='task name')
    parser.add_argument('--captions', type=str, default='data/mscoco/sample_caption.txt')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--model_path', type=str, default='data/mscoco/model.bin', help='model path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    # load model and noise scheduler
    model, noise_scheduler = get_model(args.task, model_path=args.model_path)
    model.to(args.device).eval()

    t2i_wrapper = T2IWrapper(args.task, model, args.device)

    # load captions
    with open(args.captions, 'r') as f:
        captions = f.read().splitlines()

    text_embeddings = []
    for start in range(0, len(captions), args.batch_size):
        end = min(start + args.batch_size, len(captions))
        captions_batch = captions[start:end]
        embeddings = t2i_wrapper.get_prompt_embed(captions_batch, args.device)
        text_embeddings.append(embeddings.cpu().numpy())
    text_embeddings = np.concatenate(text_embeddings, axis=0)[:, None, :, :]
    text_embeddings = torch.from_numpy(text_embeddings).to(args.device)

    # fixing the seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    generator = torch.Generator(device='cpu').manual_seed(0)

    latents = []
    all_images = []
    for start in tqdm(range(0, len(captions), args.batch_size)):
        end = min(start + args.batch_size, len(captions))
        embed_batch = text_embeddings[start:end][:, 0]
        # if embed_batch is less than batch_size, pad it
        # this is to make sure the generated samples are consistent with the ones used in the paper
        if embed_batch.shape[0] < args.batch_size:
            embed_batch = F.pad(embed_batch, (0, 0, 0, 0, 0, args.batch_size - embed_batch.shape[0]), value=0)

        images, x_0_hats, x_ts = t2i_wrapper.generate_from_embed(embed_batch, generator=generator)

        latents.append(x_ts[0][:end-start])
        all_images.extend(images[:end-start])
        torch.cuda.empty_cache()


    latents = torch.cat(latents, dim=0)        # (sample_size, 4, 16, 16)
    all_images = all_images                           

    # save results
    os.makedirs(f'{args.output_folder}/images', exist_ok=True)
    np.save(f'{args.output_folder}/latents.npy', latents.cpu().numpy())
    np.save(f'{args.output_folder}/text_embeddings.npy', text_embeddings.cpu().numpy())
    for i, image in enumerate(all_images):
        image.save(f'{args.output_folder}/images/{i}.png')
