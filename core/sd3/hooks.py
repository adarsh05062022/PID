"""Hook registration for SD3's MMDiT, the one thing SD3 genuinely needs.

WHERE THE STEERING GOES
-----------------------
`core/diffusion_steering.py` hooks `BasicTransformerBlock.attn2` -- the
cross-attention whose output carries text information into image tokens.
SD3 has no cross-attention at all: each `JointTransformerBlock` concatenates
text and image tokens and runs ONE joint self-attention over both streams,

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,              # image tokens
        encoder_hidden_states=norm_encoder_hidden_states,  # text tokens
    )

then splits the result back apart. `attn_output` is the image half AFTER it
has attended over the text -- structurally the same quantity `attn2`'s output
is in the UNet, and that is what this steers. `context_attn_output` (the text
stream) is passed through untouched: steering it would edit the prompt
representation every later block reads, which is a different intervention
than CASteer's.

WHY THE SHARED HOOK CANNOT BE REUSED
------------------------------------
`HookManager._create_attn_output_hook` does `output[..., None, :]`, which
assumes `output` is a tensor. Joint attention returns a TUPLE, so that hook
raises on SD3. The hook here unpacks the tuple, steers element 0, and
rebuilds it. (A block with `attn2` -- SD3.5's dual-attention layers, absent
in SD3-medium -- returns a bare tensor, which is handled too.)

BLOCK COUNT
-----------
`VectorControl` infers the diffusion step by counting block visits against
`num_attn_layers`, so that count must equal the number of hooked modules
visited per forward pass -- 24 for SD3-medium, one `attn` per
`JointTransformerBlock`. A wrong count does not raise, it silently
desynchronises every per-step lookup.
"""
from ..controller import VectorControl

PLACE = 'joint'          # already in the controllers' VALID_PLACES


def _create_joint_attn_hook(control: VectorControl, place_in_unet: str):
    """Steer the image-token half of a joint-attention output."""
    def hook_fn(module, inputs, output):
        if not control.active:
            return output

        is_tuple = isinstance(output, tuple)
        image_stream = output[0] if is_tuple else output

        # [B, tokens, dim] -> [B, tokens, 1, dim]: the 4D layout the
        # controllers expect, with a single head of full width, exactly as
        # the SD hook presents attn2's output.
        steered = control(image_stream[..., None, :], place_in_unet)[..., 0, :]
        # Controllers return .half(); SD3 may run in another dtype.
        steered = steered.to(image_stream.dtype)

        if is_tuple:
            return (steered,) + tuple(output[1:])
        return steered

    return hook_fn


class SD3HookManager:
    """Mirror of `core.diffusion_steering.HookManager`'s public surface."""

    def __init__(self, hooks, controls, block_count):
        self.hooks = hooks
        self.controls = list(controls)
        self.block_count = block_count

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset_controls(self):
        for control in self.controls:
            control.reset()


def sd3_register_vector_controls_with_hooks(transformer, *controls: VectorControl) -> SD3HookManager:
    """Hook every JointTransformerBlock's joint attention on an SD3 transformer.

    `transformer` is the pipeline's `SD3Transformer2DModel` (pipe.transformer).
    """
    blocks = getattr(transformer, 'transformer_blocks', None)
    if not blocks:
        raise ValueError(
            f'{type(transformer).__name__} has no transformer_blocks; '
            'expected an SD3Transformer2DModel (pipe.transformer).'
        )

    hooks = []
    block_count = 0
    for block in blocks:
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        block_count += 1
        for control in controls:
            hooks.append(attn.register_forward_hook(_create_joint_attn_hook(control, PLACE)))

    if block_count == 0:
        raise ValueError('No joint-attention modules found to hook')

    for control in controls:
        control.num_attn_layers = block_count

    return SD3HookManager(hooks, controls, block_count)
