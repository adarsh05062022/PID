"""Stable Diffusion 3 (MMDiT) support, kept entirely inside this package.

Nothing outside `core/sd3/` and the two `scripts/diffusion/*_sd3.py` entry
points was modified to add SD3, so deleting this directory and those two
scripts removes the feature completely and leaves the SD/SANA paths
byte-identical.

Why SD3 could not just be another `DiffusionModelType`: SD3 has no
cross-attention. `core/diffusion_steering.py` hooks `BasicTransformerBlock.attn2`
-- the module that writes text information into image tokens -- and MMDiT has
no such module. Text and image tokens are concatenated and run through *joint
self-attention* instead, and that joint attention returns a TUPLE
`(image_stream, text_stream)` rather than a tensor, which the shared hook in
`core/diffusion_steering.py` cannot consume. See `hooks.py`.
"""
from .hooks import SD3HookManager, sd3_register_vector_controls_with_hooks
from .pipeline import GUIDANCE_SCALE, IMG_SIZE, MODEL_ID, NUM_BLOCKS, NUM_STEPS, load_sd3, run_sd3

__all__ = [
    'SD3HookManager', 'sd3_register_vector_controls_with_hooks',
    'load_sd3', 'run_sd3',
    'MODEL_ID', 'NUM_STEPS', 'GUIDANCE_SCALE', 'IMG_SIZE', 'NUM_BLOCKS',
]
