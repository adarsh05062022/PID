"""Loading and running Stable Diffusion 3, kept out of `core/utils.py`.

Protocol constants match SAFREE's `sdv3/sdv3.py` (28 steps, guidance 7.0) so
generations are comparable with that baseline.

Needs diffusers >= 0.29 for `StableDiffusion3Pipeline`; the env the SD/SANA
paths run in has 0.25 and will raise on import. Use an env that has it
(`munba3_sd3` on this machine).
"""
import torch

MODEL_ID = 'stabilityai/stable-diffusion-3-medium-diffusers'
NUM_STEPS = 28
GUIDANCE_SCALE = 7.0
IMG_SIZE = 1024
NUM_BLOCKS = 24          # SD3-medium: 24 JointTransformerBlocks


def load_sd3(device, model_id: str = MODEL_ID, dtype=torch.float16, drop_t5: bool = False):
    """Load SD3 onto one device.

    drop_t5 frees the ~10GB T5 encoder. It CHANGES what the model generates,
    so steering vectors and the images they are applied to must agree on it --
    do not estimate with T5 and generate without it.
    """
    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError as exc:
        raise ImportError(
            'StableDiffusion3Pipeline requires diffusers >= 0.29; '
            'the default env here has 0.25. Run SD3 under an env that has it.'
        ) from exc

    kwargs = {'torch_dtype': dtype}
    if drop_t5:
        kwargs.update(text_encoder_3=None, tokenizer_3=None)
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_sd3(pipe, prompt: str, seed: int, device, num_images: int = 1,
            num_steps: int = NUM_STEPS, guidance_scale: float = GUIDANCE_SCALE,
            img_size: int = IMG_SIZE):
    """Generate with the protocol SAFREE's sdv3.py uses."""
    with torch.no_grad():
        return pipe(
            prompt=prompt,
            negative_prompt='',
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=img_size,
            width=img_size,
            num_images_per_prompt=num_images,
            generator=torch.Generator(device='cpu').manual_seed(seed),
        ).images
