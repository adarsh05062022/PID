import argparse
import json
import os
import math
import typing as tp

import pandas as pd
from diffusers import DiffusionPipeline

from core.adaptive_kg_pid_steering import AdaptiveKgPIDSteering
from core.controller import CrossAttentionOutputSteering, VectorControl
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.pid_steering_step_block import PIDSteering
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model

# Steering laws applied at the cross-attention outputs. All three read the
# same steering-vector files and are hooked into the same blocks; they differ
# only in how the scalar multiplying the steering direction is computed.
#   'casteer'     -- beta * max(<ca_X, ca_out>, 0)                (Eq. 6)
#   'pid'         -- Kp*e + Ki*I(b,t) + Kg*I_global(t) + Kd*D     (flat Kg)
#   'adaptive_kg' -- as 'pid', with Kg's share reweighted per block
PID_CONTROLLERS = {
    'pid': PIDSteering,
    'adaptive_kg': AdaptiveKgPIDSteering,
}

SAVE_OPTIONS = {
    'PNG': {},
    'JPEG': {
        'subsampling': '4:4:4',
        'quality': 95,
    },
}

EXTENSIONS = {
    'PNG': 'png',
    'JPEG': 'jpg',
}


def load_prompts(args):
    """Load evaluation prompts: a custom CSV, ImageNet templates, or COCO captions.

    Returns (prompts, num_images_per_prompt, seeds); seeds is a per-prompt
    list only for --prompts_csv with --seed_column, else None (every prompt
    then starts at --seed).
    """
    if args.prompts_csv:
        df = pd.read_csv(args.prompts_csv)
        keep = [isinstance(p, str) and bool(p.strip()) for p in df['prompt']]
        prompts = [p for p, k in zip(df['prompt'], keep) if k]
        seeds = None
        if args.seed_column:
            # Benchmarks such as SAFREE's carry their own per-prompt seed
            # (evaluation_seed); the row order is kept so <prompt_idx> still
            # indexes the CSV.
            seeds = [int(s) for s, k in zip(df[args.seed_column], keep) if k]
        return prompts, args.num_images_per_prompt, seeds
    if args.generate_concept != 'coco':
        template_path = args.template_path or 'exp/datasets/eval/imagenet/template.json'
        with open(template_path) as f:
            templates = json.load(f)
        return [t.format(args.generate_concept) for t in templates], args.num_images_per_prompt, None
    else:
        df = pd.read_csv('exp/datasets/eval/coco/coco_30k.csv')
        prompts = [p for p in df['prompt'] if 'horse' not in p.lower()]
        if args.max_samples:
            prompts = prompts[:args.max_samples]
        return prompts, 1, None


def hook_model(pipeline: DiffusionPipeline, device: tp.Any, args: argparse.Namespace) -> VectorControl:
    if args.command is None:
        return None

    if args.command == 'erase':
        source_concept = unpickle(args.concept_path)
        target_concept = None
    else:
        source_concept = unpickle(args.source_concept_path)
        target_concept = unpickle(args.target_concept_path)

    if args.controller == 'casteer':
        vector_control = CrossAttentionOutputSteering(
            target_concepts=[target_concept],
            source_concepts=[source_concept],
            strength=args.steering_strength,
            device=device,
            intermediate_clipping=args.intermediate_clipping,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
        )
    else:
        # Same vectors, same blocks, same first-step-vector convention as
        # above -- only the scalar applied to the direction differs. At
        # Ki=Kg=Kd=0 with --intermediate_clipping this is CASteer Eq. 6 with
        # beta = Kp, so the PID runs stay comparable to the CASteer ones.
        vector_control = PID_CONTROLLERS[args.controller](
            target_concepts=[target_concept],
            source_concepts=[source_concept],
            kp_gain=args.kp,
            ki_gain=args.ki,
            kg_gain=args.kg,
            kd_gain=args.kd,
            device=device,
            clip=args.intermediate_clipping,
            integral_clamp=args.integral_clamp,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
            selected_blocks=None,   # every hooked block, as CASteer steers
            diag=args.diag,
        )

    model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        vector_control,
        model_type=DiffusionModelType.from_model(args.model_name),
    )
    return vector_control


def main(args: argparse.Namespace):
    if args.controller != 'casteer':
        # Kp is the PID analogue of beta, so --steering_strength stands in for
        # --kp when only one of the two is given.
        if args.kp is None:
            args.kp = args.steering_strength
        if args.kp is None:
            raise ValueError(f'--kp (or --steering_strength) must be specified for --controller {args.controller}')
        if args.command is None:
            raise ValueError(f'--controller {args.controller} needs a steering action (erase or translate)')
        if not args.intermediate_clipping:
            # Without the clamp a negative PID output adds the steering
            # direction back, i.e. injects the concept. CASteer's Eq. 6 is
            # never run that way, and neither should the PID law be.
            print('WARNING: --controller {} without --intermediate_clipping leaves the PID output '
                  'unclamped, so the I/D terms can inject the concept instead of erasing it'
                  .format(args.controller))
    elif args.steering_strength is None and args.command is not None:
        raise ValueError(f'--steering_strength (float) must be specified for steering')

    if args.command is None and args.steering_strength is not None:
        raise ValueError(f'--steering_strength is provided but no steering action (erase or translate) specified')

    if args.generate_concept is None and args.prompts_csv is None:
        raise ValueError('pass --generate_concept or --prompts_csv')
    if args.seed_column and not args.prompts_csv:
        raise ValueError('--seed_column needs --prompts_csv')

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    if args.scheduler == 'dpm-multistep':
        # SAFREE's sampler (DPMSolverMultistepScheduler.from_pretrained on the
        # model's own scheduler config); the model's default is kept otherwise.
        from diffusers import DPMSolverMultistepScheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    if args.vae_slicing:
        # Decode the batch one image at a time. Cuts the peak memory of a
        # 10-image SD-1.4 batch from >10 GB to ~4 GB; the images differ from a
        # batched decode only by +-1/255 rounding on ~1-2% of pixels.
        pipeline.enable_vae_slicing()
    device = get_device()

    vector_control = hook_model(pipeline, device, args)

    prompts, num_images_per_prompt, prompt_seeds = load_prompts(args)
    skipped = generated = 0

    # --diag only exists on the PID controllers: one CSV per generated batch
    # with the per-block, per-step P/I/G/D trace behind that batch's images.
    records_dir = None
    # Whole-run term magnitudes, accumulated here rather than read off the
    # controller's own diag_summary(): reset() clears its internal stats, and
    # reset() runs after every image, so by the end of the run they are empty.
    diag_totals = {k: 0.0 for k in ('n', 'e', 'p', 'i', 'g', 'd')}
    if args.diag and args.controller != 'casteer':
        records_dir = os.path.join(args.output_dir, 'pid_records')
        os.makedirs(records_dir, exist_ok=True)

    if args.controller == 'casteer':
        setting = f'strength {args.steering_strength}'
    else:
        setting = f'{args.controller} kp={args.kp} ki={args.ki} kg={args.kg} kd={args.kd}'
    print(f'Generating images for concept {args.generate_concept} with {setting}')
    shard_start = args.shard_start if args.shard_start is not None else 0
    shard_end = args.shard_end if args.shard_end is not None else len(prompts)
    if shard_start or shard_end != len(prompts):
        print(f'Sharding: only prompt_idx in [{shard_start}, {shard_end}) '
              f'({shard_end - shard_start} of {len(prompts)} prompts)')
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < shard_start or prompt_idx >= shard_end:
            continue
        # --prompts_csv prompts are free text and can exceed the filesystem's
        # ~255-byte filename limit, so they cannot be used as a directory name
        # the way a short template/caption is; fall back to the index instead.
        prompt_dir = str(prompt_idx) if args.prompts_csv else prompt
        base_seed = prompt_seeds[prompt_idx] if prompt_seeds is not None else args.seed
        num_batches = math.ceil(num_images_per_prompt / args.batch_size)
        for batch_id in range(0, num_batches):
            seed = base_seed + batch_id
            num_images = min(args.batch_size, num_images_per_prompt - batch_id * args.batch_size)

            output_paths = [f'{args.output_dir}/{prompt_dir}/{seed}-{idx}.{EXTENSIONS[args.file_format]}' for idx in range(num_images)]
            if all(os.path.exists(path) for path in output_paths):
                skipped += num_images
                continue
            generated += num_images
            images = run_image_model(
                model_type=args.model_name,
                pipe=pipeline,
                prompt=prompt,
                seed=seed,
                device=device,
                num_images=num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                image_size=args.image_size,
            )
            if records_dir is not None:
                # Drain before reset() so this batch's trace does not bleed
                # into the next one's.
                records = vector_control.drain_records()
                if records:
                    pd.DataFrame(records).to_csv(
                        f'{records_dir}/{prompt_idx:05d}-{seed}.csv', index=False)
                    diag_totals['n'] += len(records)
                    for key, field in (('e', 'mean_abs_e'), ('p', 'mean_abs_p_out'),
                                       ('i', 'mean_abs_i_out'), ('g', 'mean_abs_g_out'),
                                       ('d', 'mean_abs_d_out')):
                        diag_totals[key] += sum(r[field] for r in records)
            if vector_control is not None:
                vector_control.reset()
            os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
            if args.prompts_csv:
                with open(f'{args.output_dir}/{prompt_dir}/prompt.txt', 'w') as f:
                    f.write(prompt + '\n')
            for path, image in zip(output_paths, images):
                image.save(path, format=args.file_format, **SAVE_OPTIONS[args.file_format])

    print(f'Skipped {skipped} images, generated {generated} images')
    if diag_totals['n']:
        n = diag_totals['n']
        print(f"PID term magnitudes over the whole run: "
              f"mean|e|={diag_totals['e']/n:.4f}  "
              f"mean|Kp*e|={diag_totals['p']/n:.4f}  "
              f"mean|Ki*I|={diag_totals['i']/n:.4f}  "
              f"mean|Kg*Ig|={diag_totals['g']/n:.4f}  "
              f"mean|Kd*D|={diag_totals['d']/n:.4f}  "
              f"(over {int(n)} steered block visits)")
        print(f'Per-step PID records written to {records_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main_parser = parser.add_argument_group('Common arguments')

    # Generation params
    main_parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                             help='Diffusion model name used for generation')
    main_parser.add_argument('--generate_concept', type=str, default=None,
                             help='Concept for which to generate images (label only when --prompts_csv is given)')
    main_parser.add_argument('--prompts_csv', type=str, default=None,
                             help="CSV with a 'prompt' column to generate from, overriding "
                                  '--generate_concept\'s built-in ImageNet/COCO prompt sources')
    main_parser.add_argument('--output_dir', type=str, required=True, help='Directory where generated images should be written')
    main_parser.add_argument('--num_images_per_prompt', type=int, default=10, help='Number of images to generate for each prompt')
    main_parser.add_argument('--batch_size', type=int, default=1, help='Batch size used for image generation')
    main_parser.add_argument('--seed', type=int, default=0, help='Starting seed for each prompt')
    main_parser.add_argument('--file_format', type=str, choices=['PNG', 'JPEG'], default='PNG', help='File format for generated images')
    main_parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from the dataset')
    main_parser.add_argument('--shard_start', type=int, default=None,
                             help='Only generate prompt_idx >= this (global index into --prompts_csv, '
                                  'before filtering); pairs with --shard_end to split one CSV across '
                                  'GPUs while keeping directory names/skip-logic globally consistent')
    main_parser.add_argument('--shard_end', type=int, default=None,
                             help='Only generate prompt_idx < this (exclusive)')
    main_parser.add_argument('--template_path', type=str, default=None, help='Path to template JSON for evaluation prompts (default: imagenet template)')
    main_parser.add_argument('--vae_slicing', action='store_true',
                             help='Decode the VAE one image at a time (much lower peak memory for batch_size > 1)')
    # Sampling-protocol overrides. All default to "leave the model's usual
    # settings alone"; a benchmark that fixes its own sampler (SAFREE's SDXL
    # rows: --scheduler dpm-multistep --num_inference_steps 50
    # --guidance_scale 7.5 --image_size 512 --seed_column seed) sets them.
    main_parser.add_argument('--seed_column', type=str, default=None,
                             help='Column of --prompts_csv holding a per-prompt seed (overrides --seed)')
    main_parser.add_argument('--num_inference_steps', type=int, default=None,
                             help='Denoising steps (default: the per-model count in core.utils)')
    main_parser.add_argument('--guidance_scale', type=float, default=None,
                             help="Classifier-free guidance scale (default: the pipeline's own)")
    main_parser.add_argument('--image_size', type=int, default=None,
                             help="Square output resolution (default: the pipeline's own)")
    main_parser.add_argument('--scheduler', choices=['default', 'dpm-multistep'], default='default',
                             help="'dpm-multistep' swaps in DPMSolverMultistepScheduler on the model's scheduler config")

    # Steering params
    main_parser.add_argument('--steering_strength', type=float, default=None, help='Steering strength beta (default for erasure: 2.0)')
    main_parser.add_argument('--intermediate_clipping', action='store_true',
                             help='Apply intermediate clipping (Eq. 6). For the PID controllers this is the '
                                  'same non-negativity clamp, applied to the combined PID output')
    main_parser.add_argument('--use_all_diffusion_steps', action='store_true', help='Use per-step steering vectors instead of single-step')

    # Controller choice. All controllers use the same steering vectors, the
    # same hooks and the same blocks; only the scalar applied to the steering
    # direction differs.
    pid_parser = parser.add_argument_group(
        'PID controller arguments (--controller pid / adaptive_kg)')
    main_parser.add_argument('--controller', choices=['casteer', *PID_CONTROLLERS], default='casteer',
                             help="'casteer' = core.controller.CrossAttentionOutputSteering (Eq. 6); "
                                  "'pid' = core.pid_steering_step_block.PIDSteering (per-block + cross-block "
                                  "integrators, flat Kg); 'adaptive_kg' = "
                                  'core.adaptive_kg_pid_steering.AdaptiveKgPIDSteering (Kg reweighted per block)')
    pid_parser.add_argument('--kp', type=float, default=None,
                            help='Proportional gain, the PID analogue of beta (default: --steering_strength)')
    pid_parser.add_argument('--ki', type=float, default=0.0,
                            help='Per-block integral gain, over the ~50 denoising steps (start near Kp/50)')
    pid_parser.add_argument('--kg', type=float, default=0.0,
                            help='Cross-block integral gain, on the same scale as --ki')
    pid_parser.add_argument('--kd', type=float, default=0.0, help='Derivative gain')
    pid_parser.add_argument('--integral_clamp', type=float, default=None,
                            help='Anti-windup cap applied to both integrators')
    pid_parser.add_argument('--diag', action='store_true',
                            help='Report mean P/I/G/D magnitudes and write the per-step trace '
                                 'to {output_dir}/pid_records')

    subparsers = parser.add_subparsers(dest='command')

    # Params for concept erasure
    erase_parser = subparsers.add_parser('erase')
    erase_parser.add_argument('--concept_path', type=str, required=True,
                              help='Path to concept vectors to erase')

    # Params for concept switching
    translate_parser = subparsers.add_parser('translate')
    translate_parser.add_argument('--source_concept_path', type=str, required=True,
                                  help='Path to source concept vectors')
    translate_parser.add_argument('--target_concept_path', type=str, required=True,
                                  help='Path to target concept vectors')

    args = parser.parse_args()

    main(args)
