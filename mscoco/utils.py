import os
import re
import importlib
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from pycocotools.coco import COCO
from typing import Union, Optional, List, Callable, Dict, Any
from diffusers import DDPMScheduler, StableDiffusionPipeline


def convert_images_to_grid(
    images: List[Image.Image],
    ncols: int = 8,
    grid_sz: int = 64
) -> Image.Image:
    """
    Convert a list of images to a grid.

    Args:
        images (List[Image.Image]): List of images to convert.
        k (int): Number of images to include in the grid.
        ncols (int): Number of columns in the grid.
        grid_sz (int): Size of each grid cell.

    Returns:
        Image.Image: The grid image.
    """
    # Create a grid of subplots
    k = len(images)
    nrows = (k - 1) // ncols + 1
    grid = Image.new('RGB', (ncols * grid_sz, nrows * grid_sz))

    # Iterate over the top k images
    for i, image in enumerate(images):
        # Resize the image to fit the grid size
        resized_image = image.resize((grid_sz, grid_sz))

        # Calculate the position of the image in the grid
        x = (i % ncols) * grid_sz
        y = (i // ncols) * grid_sz

        # Paste the resized image onto the grid
        grid.paste(resized_image, (x, y))

    return grid


def find_and_import_module(folder_name, module_name):
    found = False
    for module_path in os.listdir(folder_name):
        module_path = module_path.split(".")[0]
        if module_path.lower() == module_name.lower():
            found = True
            break

    if not found:
        raise ValueError(f"Cannot find module {module_name} in {folder_name}.")

    # Import the module dynamically
    parent_module = folder_name.replace("/", ".")
    full_module_name = f"{parent_module}.{module_path}"
    module = importlib.import_module(full_module_name)
    return module


def check_substrings(text, list_of_substrings):
    # Start building the regex pattern with lookaheads for each sublist
    pattern = ""
    for substrings in list_of_substrings:
        # Create a pattern segment for the current list of substrings
        substrings_pattern = f"({'|'.join(map(re.escape, substrings))})"
        # Add a lookahead for this pattern segment
        pattern += f"(?=.*{substrings_pattern})"

    # Use re.search to check if the text matches the full pattern
    if re.search(pattern, text):
        return True
    else:
        return False


class COCOVis:
    def __init__(self, path="data/mscoco", split="train"):
            dataType = f"{split}2017"
            annFile = os.path.join(path, "annotations", f"captions_{dataType}.json")
            self.imgdir = os.path.join(path, dataType)
            self.coco = COCO(annFile)
            self.img_ids = list(self.coco.imgs.keys())
            self.captions = self.coco.imgToAnns

    def __getitem__(self, idx):
        # get image and caption
        i = self.img_ids[idx]
        img_path = os.path.join(self.imgdir, self.coco.loadImgs(i)[0]['file_name'])
        img = Image.open(img_path)
        img = img.convert('RGB')

        # center crop
        w, h = img.size
        if w > h:
            img = img.crop(((w - h) // 2, 0, (w + h) // 2, h))
        elif h > w:
            img = img.crop((0, (h - w) // 2, w, (h + w) // 2))

        captions = [x["caption"] for x in self.captions[i]]
        return img, captions


class T2IWrapper:
    def __init__(self, task, unet, device):
        if task == "mscoco_t2i":
            # load pipeline to generate images
            noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
            sd_model_card = "stabilityai/stable-diffusion-2"
            self.pipe = StableDiffusionPipeline.from_pretrained(sd_model_card,
                                                        unet=unet,
                                                        scheduler=noise_scheduler,
                                                        safety_checker=None).to(device)
            noise_scheduler.config.steps_offset = 0     # For DDPM steps_offset is still 0, only do 1 for others like DDIMScheduler
            noise_scheduler.config.clip_sample = True   # Same here, clip_sample is set to True for DDPM
        else:
            raise ValueError(f"Invalid task: {task}")

        self.device = device

    def generate_from_prompt(self, prompt, num_images_per_prompt=1, num_inference_steps=1000, generator=None):
        # run the pipeline
        images, x_0_hats, x_ts = run_t2i_pipeline(self.pipe,
                                                  prompt=prompt,
                                                  num_images_per_prompt=num_images_per_prompt,
                                                  num_inference_steps=num_inference_steps,
                                                  generator=generator)
        return images, x_0_hats, x_ts

    def generate_from_embed(self, embed, num_images_per_prompt=1, num_inference_steps=1000, generator=None):
        # run the pipeline
        images, x_0_hats, x_ts = run_t2i_pipeline(self.pipe,
                                                  prompt_embeds=embed,
                                                  num_images_per_prompt=num_images_per_prompt,
                                                  num_inference_steps=num_inference_steps,
                                                  generator=generator)
        return images, x_0_hats, x_ts

    def encode_text(self, prompt):
        tokenizer, text_encoder = self.pipe.tokenizer, self.pipe.text_encoder
        tokens = tokenizer(prompt,
                           max_length=tokenizer.model_max_length,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")['input_ids']

        tokens = tokens.to(self.device)
        states = text_encoder(tokens)[0].cpu().numpy()
        return states

    def decode_latents(self, latents, vae_batch_size=20, generator=None):
        image = []
        vae = self.pipe.vae
        for start in range(0, latents.shape[0], vae_batch_size):
            end = min(start + vae_batch_size, latents.shape[0])
            im = vae.decode(latents[start:end] / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image.append(im)
        image = torch.cat(image, dim=0)
        return image

    def get_image_latent(self, image, device):
        return self.pipe.encode_image(image, device, 1)

    def get_prompt_embed(self, prompt, device):
        return self.pipe.encode_prompt(prompt, device, 1, False)[0]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def run_t2i_pipeline(
    self,
    prompt: Union[str, List[str]] = None,
    vae_batch_size: int = 20,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
            using zero terminal SNR.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
            otherwise a `tuple` is returned where the first element is a list with the generated images and the
            second element is a list of `bool`s indicating whether the corresponding generated image contains
            "not-safe-for-work" (nsfw) content.
    """

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    # print(timesteps, num_inference_steps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    x_0_hats = []
    inputs = [latents.detach().clone().cpu()]

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
            latents = out.prev_sample
            x_0_hat = out.pred_original_sample

            x_0_hats.append(x_0_hat.detach().clone().cpu())
            inputs.append(latents.detach().clone().cpu())

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    image = []
    for start in range(0, latents.shape[0], vae_batch_size):
        end = min(start + vae_batch_size, latents.shape[0])
        im = self.vae.decode(latents[start:end] / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image.append(im)
    image = torch.cat(image, dim=0)

    do_denormalize = [True] * image.shape[0]
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    self.maybe_free_model_hooks()

    return image, x_0_hats[::-1], inputs[::-1]
