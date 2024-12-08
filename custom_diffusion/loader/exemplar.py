import json
import torch
from PIL import Image
from torchvision import transforms
from diffusers import DiffusionPipeline


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 sd_version,
                 custom_diffusion_model_path,
                 test_case,
                 test_case_ind,
                 train_prompt,
                 vae_batch_size=20,
                 vae_device='cuda',
                 dataroot='data/abc',
                 mode='no_flip_and_flip'
                 ):
        super().__init__()
        self.sd_version = sd_version
        self.model_path = custom_diffusion_model_path
        self.test_case = test_case
        self.test_case_ind = test_case_ind
        self.train_prompt = train_prompt
        self.dataroot = dataroot
        self.mode = mode

        self.exemplar_paths = self._get_exemplar_paths()
        self.latents, self.hidden_states = self._prepare_exemplar_dataset(self.exemplar_paths, batch_size=vae_batch_size, device=vae_device)
        self.length = len(self.latents)
        self.hidden_states_size = self.hidden_states.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.latents[idx], self.hidden_states[idx % self.hidden_states_size]

    def _get_exemplar_paths(self):
        # load a dictionary of exemplars paths
        with open(f'{self.dataroot}/json/{self.test_case}.json', 'r') as f:
            exemplar_paths = json.load(f)[self.test_case_ind]['exemplar']
        return [s.replace('dataset', self.dataroot) for s in exemplar_paths]

    @torch.no_grad()
    def _prepare_exemplar_dataset(self, exemplar_paths, batch_size, device):
        # load pipeline
        pipeline = DiffusionPipeline.from_pretrained(self.sd_version).to(device)
        pipeline.load_textual_inversion(self.model_path, weight_name="<new1>.bin")
        pipeline.safety_checker = None
        vae = pipeline.vae
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        # get image transformation
        size = 512
        if self.mode == 'no_flip_and_flip' or self.mode == 'no_flip':
            image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            images_noflip = [image_transforms(Image.open(exemplar_path).convert('RGB')) for exemplar_path in exemplar_paths]

        if self.mode == 'no_flip_and_flip' or self.mode == 'flip':
            image_transform_flip = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            images_flip = [image_transform_flip(Image.open(exemplar_path).convert('RGB')) for exemplar_path in exemplar_paths]

        if self.mode == 'no_flip_and_flip':
            images = torch.stack(images_noflip + images_flip).to(device)
        elif self.mode == 'no_flip':
            images = torch.stack(images_noflip).to(device)
        elif self.mode == 'flip':
            images = torch.stack(images_flip).to(device)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        num_exemplars = images.size(0)
        latents = []
        for start in range(0, num_exemplars, batch_size):
            end = min(start + batch_size, num_exemplars)
            batch_images = images[start:end]

            lats = vae.encode(batch_images).latent_dist.sample()
            lats = lats * vae.config.scaling_factor
            latents.append(lats.cpu())

        latents = torch.cat(latents, dim=0)

        tokens = tokenizer([self.train_prompt],
                            max_length=tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")['input_ids'].to(device)
        encoder_hidden_states = text_encoder(tokens)[0].cpu()
        encoder_hidden_states = encoder_hidden_states.expand(latents.size(0), -1, -1)

        del pipeline
        return latents, encoder_hidden_states
