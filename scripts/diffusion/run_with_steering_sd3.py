"""
Generate with Stable Diffusion 3 under CASteer / PID steering.

The SD3 counterpart of `run_with_steering.py`, with the same `--controller`
choices and the same gains; only the model and the hook site differ (see
`core/sd3/hooks.py`). Kept separate so deleting `core/sd3/` and the two
`*_sd3.py` scripts removes SD3 support entirely.

    python scripts/diffusion/run_with_steering_sd3.py \
        --prompts_csv datasets/unsafe_plus_safe.csv \
        --output_dir ./results/sd3/unsafe_plus_safe/adaptive_kg \
        --controller adaptive_kg --kp 2.0 --ki 0.01 --kg 0.01 --kd 0 \
        --intermediate_clipping --diag \
        erase --concept_path ./results/sd3/steering_vectors/nudity.pt

Needs an env with diffusers >= 0.29 (StableDiffusion3Pipeline).
"""
import argparse
import json
import math
import os

import pandas as pd

from core.adaptive_kg_pid_steering import AdaptiveKgPIDSteering
from core.controller import CrossAttentionOutputSteering
from core.pickle import unpickle
from core.pid_steering_step_block import PIDSteering
from core.sd3 import GUIDANCE_SCALE, IMG_SIZE, MODEL_ID, NUM_BLOCKS, NUM_STEPS, load_sd3, run_sd3, \
    sd3_register_vector_controls_with_hooks

PID_CONTROLLERS = {
    'pid': PIDSteering,
    'adaptive_kg': AdaptiveKgPIDSteering,
}

SAVE_OPTIONS = {'PNG': {}, 'JPEG': {'subsampling': '4:4:4', 'quality': 95}}
EXTENSIONS = {'PNG': 'png', 'JPEG': 'jpg'}


def load_prompts(args):
    if args.prompts_csv:
        df = pd.read_csv(args.prompts_csv)
        return [p for p in df['prompt'] if isinstance(p, str) and p.strip()]
    template_path = args.template_path or 'exp/datasets/eval/imagenet/template.json'
    with open(template_path) as f:
        templates = json.load(f)
    return [t.format(args.generate_concept) for t in templates]


def hook_model(pipe, args):
    if args.command is None:
        return None

    if args.command == 'erase':
        source_concept, target_concept = unpickle(args.concept_path), None
    else:
        source_concept = unpickle(args.source_concept_path)
        target_concept = unpickle(args.target_concept_path)

    import torch
    device = torch.device(args.device)

    if args.controller == 'casteer':
        vector_control = CrossAttentionOutputSteering(
            source_concepts=[source_concept],
            target_concepts=[target_concept],
            strength=args.steering_strength,
            device=device,
            intermediate_clipping=args.intermediate_clipping,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
        )
    else:
        vector_control = PID_CONTROLLERS[args.controller](
            source_concepts=[source_concept],
            target_concepts=[target_concept],
            kp_gain=args.kp, ki_gain=args.ki, kg_gain=args.kg, kd_gain=args.kd,
            device=device,
            clip=args.intermediate_clipping,
            integral_clamp=args.integral_clamp,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
            selected_blocks=None,
            diag=args.diag,
        )

    manager = sd3_register_vector_controls_with_hooks(pipe.transformer, vector_control)
    if manager.block_count != NUM_BLOCKS:
        print(f'note: hooked {manager.block_count} joint-attention blocks, '
              f'expected {NUM_BLOCKS} for SD3-medium')
    return vector_control


def main(args):
    if args.controller != 'casteer':
        if args.kp is None:
            args.kp = args.steering_strength
        if args.kp is None:
            raise ValueError(f'--kp (or --steering_strength) must be specified for --controller {args.controller}')
        if args.command is None:
            raise ValueError(f'--controller {args.controller} needs a steering action (erase or translate)')
        if not args.intermediate_clipping:
            print(f'WARNING: --controller {args.controller} without --intermediate_clipping leaves the PID '
                  'output unclamped, so the I/D terms can inject the concept instead of erasing it')
    elif args.steering_strength is None and args.command is not None:
        raise ValueError('--steering_strength (float) must be specified for steering')

    if args.generate_concept is None and args.prompts_csv is None:
        raise ValueError('pass --generate_concept or --prompts_csv')

    pipe = load_sd3(args.device, model_id=args.model_id, drop_t5=args.drop_t5)
    vector_control = hook_model(pipe, args)

    prompts = load_prompts(args)
    skipped = generated = 0

    records_dir = None
    diag_totals = {k: 0.0 for k in ('n', 'e', 'p', 'i', 'g', 'd')}
    if args.diag and args.controller != 'casteer':
        records_dir = os.path.join(args.output_dir, 'pid_records')
        os.makedirs(records_dir, exist_ok=True)

    setting = (f'strength {args.steering_strength}' if args.controller == 'casteer'
               else f'{args.controller} kp={args.kp} ki={args.ki} kg={args.kg} kd={args.kd}')
    print(f'Generating SD3 images for concept {args.generate_concept} with {setting}')

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = str(prompt_idx) if args.prompts_csv else prompt
        num_batches = math.ceil(args.num_images_per_prompt / args.batch_size)
        for batch_id in range(num_batches):
            seed = args.seed + batch_id
            num_images = min(args.batch_size, args.num_images_per_prompt - batch_id * args.batch_size)

            output_paths = [f'{args.output_dir}/{prompt_dir}/{seed}-{idx}.{EXTENSIONS[args.file_format]}'
                            for idx in range(num_images)]
            if all(os.path.exists(path) for path in output_paths):
                skipped += num_images
                continue
            generated += num_images

            images = run_sd3(pipe, prompt, seed=seed, device=args.device, num_images=num_images,
                             num_steps=args.num_steps, guidance_scale=args.guidance_scale,
                             img_size=args.img_size)

            if records_dir is not None:
                records = vector_control.drain_records()
                if records:
                    pd.DataFrame(records).to_csv(f'{records_dir}/{prompt_idx:05d}-{seed}.csv', index=False)
                    diag_totals['n'] += len(records)
                    for key, field in (('e', 'mean_abs_e'), ('p', 'mean_abs_p_out'), ('i', 'mean_abs_i_out'),
                                       ('g', 'mean_abs_g_out'), ('d', 'mean_abs_d_out')):
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
              f"mean|e|={diag_totals['e']/n:.4f}  mean|Kp*e|={diag_totals['p']/n:.4f}  "
              f"mean|Ki*I|={diag_totals['i']/n:.4f}  mean|Kg*Ig|={diag_totals['g']/n:.4f}  "
              f"mean|Kd*D|={diag_totals['d']/n:.4f}  (over {int(n)} steered block visits)")
        print(f'Per-step PID records written to {records_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = parser.add_argument_group('Common arguments')
    m.add_argument('--generate_concept', type=str, default=None)
    m.add_argument('--prompts_csv', type=str, default=None, help="CSV with a 'prompt' column")
    m.add_argument('--template_path', type=str, default=None)
    m.add_argument('--output_dir', type=str, required=True)
    m.add_argument('--num_images_per_prompt', type=int, default=1)
    m.add_argument('--batch_size', type=int, default=1)
    m.add_argument('--seed', type=int, default=42)
    m.add_argument('--file_format', choices=['PNG', 'JPEG'], default='PNG')
    m.add_argument('--device', default='cuda:0')
    m.add_argument('--model_id', default=MODEL_ID)
    m.add_argument('--num_steps', type=int, default=NUM_STEPS)
    m.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE)
    m.add_argument('--img_size', type=int, default=IMG_SIZE)
    m.add_argument('--drop_t5', action='store_true')

    m.add_argument('--steering_strength', type=float, default=None)
    m.add_argument('--intermediate_clipping', action='store_true')
    m.add_argument('--use_all_diffusion_steps', action='store_true')
    m.add_argument('--controller', choices=['casteer', *PID_CONTROLLERS], default='casteer')

    pid = parser.add_argument_group('PID controller arguments')
    pid.add_argument('--kp', type=float, default=None, help='default: --steering_strength')
    pid.add_argument('--ki', type=float, default=0.0, help='start near Kp/num_steps (28 for SD3)')
    pid.add_argument('--kg', type=float, default=0.0)
    pid.add_argument('--kd', type=float, default=0.0)
    pid.add_argument('--integral_clamp', type=float, default=None)
    pid.add_argument('--diag', action='store_true')

    sub = parser.add_subparsers(dest='command')
    e = sub.add_parser('erase')
    e.add_argument('--concept_path', type=str, required=True)
    t = sub.add_parser('translate')
    t.add_argument('--source_concept_path', type=str, required=True)
    t.add_argument('--target_concept_path', type=str, required=True)

    main(parser.parse_args())
