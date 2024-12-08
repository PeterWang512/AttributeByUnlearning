import os
import random
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot='data/mscoco', split='train', mode='no_flip'):
        self.dataroot = dataroot
        self.latents = np.load(os.path.join(dataroot, f'coco17_{split}_latents.npy'), mmap_mode='r')
        self.hidden_states = np.load(os.path.join(dataroot, f'coco17_{split}_text_embeddings.npy'), mmap_mode='r')
        self.num_captions = self.hidden_states.shape[1]

        self.mode = mode
        self.orig_length = len(self.hidden_states)
        if mode == 'no_flip':
            self.length = self.orig_length
            self.latents = self.latents[:self.length]
        elif mode == 'flip':
            self.length = self.orig_length
            self.latents = self.latents[self.length:]
        elif mode == 'no_flip_and_flip':
            self.length = self.orig_length * 2
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        latents_tensor = torch.from_numpy(self.latents[idx].copy()).float()

        hidx = idx % len(self.hidden_states)
        hidden_states_tensor = torch.from_numpy(self.hidden_states[hidx].copy()).float()

        return latents_tensor, hidden_states_tensor
