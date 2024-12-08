import os
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot='data/abc', subset_size=None, mode='no_flip'):
        self.dataroot = dataroot
        self.subset_size = subset_size

        self.latents = np.load(os.path.join(dataroot, f'laion_latents.npy'), mmap_mode='r')
        self.hidden_states = np.load(os.path.join(dataroot, f'laion_text_embeddings.npy'), mmap_mode='r')
        self.num_captions = self.hidden_states.shape[1]        

        self.mode = mode
        self.orig_length = len(self.hidden_states)
        if mode == 'no_flip':
            self.length = self.orig_length if subset_size is None else subset_size
            self.latents = self.latents[:self.length]
        elif mode == 'flip':
            self.length = self.orig_length if subset_size is None else subset_size
            self.latents = self.latents[self.orig_length : self.orig_length + self.length]
        elif mode == 'no_flip_and_flip':
            self.length = self.orig_length * 2 if subset_size is None else subset_size * 2
            if subset_size is not None:
                latents_no_flip = self.latents[:subset_size]
                latents_flip = self.latents[self.orig_length : self.orig_length + subset_size]
                self.latents = np.concatenate([latents_no_flip, latents_flip], axis=0)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if subset_size is not None:
            self.hidden_states = self.hidden_states[:subset_size]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        latents_tensor = torch.from_numpy(self.latents[idx].copy()).float()

        hidx = idx % len(self.hidden_states)
        hidden_states_tensor = torch.from_numpy(self.hidden_states[hidx].copy()).float()

        return latents_tensor, hidden_states_tensor
