"""
Artist-style erasure images, generated under the artist-removal protocol.

`scripts/diffusion/run_with_steering.py` writes `{prompt}/{seed}-{idx}.png` with
one fixed seed per run, which the artist protocol cannot score. This script runs
the SAME controllers over the SAME steering vectors and the same hooks, but
pinned to the protocol's generation settings so `scripts/style/eval_artist.py`
can read its output directly:

  * fp16 SD-v1.4 with `DPMSolverMultistepScheduler.from_pretrained(model_id,
    subfolder='scheduler')` and the safety checker off -- the scheduler is what
    differs from CASteer's own runner, which keeps the default PNDM;
  * the per-row `evaluation_seed` from the prompt CSV, not one seed per run;
  * 50 steps, guidance 7.5, 512px;
  * `{case_number}.png` written into `{save_dir}/all/`.

Every arm of the comparison -- including the unsteered baseline that LPIPS is
measured against (`--no_steer`) -- goes through this one code path, so nothing
but the steering law differs between the directories being compared.

    # the method
    python scripts/style/generate_artist.py \
        --artist "Van Gogh" \
        --concept_path "results/sd14/steering_vectors/Van Gogh.pt" \
        --controller adaptive_kg --kp 2.0 --ki 0.01 --kg 0.01 --kd 0.0 \
        --intermediate_clipping \
        --save_dir results/sd14/style/casteer_vectors/adaptive_kg_vangogh --gpus 1,3,6

    # the LPIPS reference -- vector-set-independent, so it stays outside
    # casteer_vectors/teca_vectors at the shared top level
    python scripts/style/generate_artist.py \
        --artist "Van Gogh" --no_steer \
        --save_dir results/sd14/style/baseline_vangogh --gpus 1

`scripts/style/run_artist.sh vangogh <gpu_a> <gpu_b> <gpu_c> <gpu_d>` runs all
four arms; see `scripts/style/README.md` for the protocol.
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.multiprocessing as mp
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.adaptive_kg_pid_steering import AdaptiveKgPIDSteering
from core.controller import CrossAttentionOutputSteering
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.pid_steering_step_block import PIDSteering
from core.style_data import ARTIST_DATASETS, load_artist_dataset

PID_CONTROLLERS = {
    'pid': PIDSteering,
    'adaptive_kg': AdaptiveKgPIDSteering,
}


def load_pipeline(model_id: str, device: torch.device) -> StableDiffusionPipeline:
    """SD-v1.4 as the artist protocol builds it: DPMSolver, fp16, no safety checker."""
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, safety_checker=None,
        torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_control(args, device):
    """Instantiate the steering law. Same vectors and same hooks for all three;
    only the scalar multiplying the steering direction differs."""
    source_concept = unpickle(args.concept_path)

    if args.controller == 'casteer':
        return CrossAttentionOutputSteering(
            source_concepts=[source_concept],
            target_concepts=[None],
            strength=args.steering_strength,
            device=device,
            intermediate_clipping=args.intermediate_clipping,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
        )
    return PID_CONTROLLERS[args.controller](
        source_concepts=[source_concept],
        target_concepts=[None],
        kp_gain=args.kp,
        ki_gain=args.ki,
        kg_gain=args.kg,
        kd_gain=args.kd,
        device=device,
        clip=args.intermediate_clipping,
        integral_clamp=args.integral_clamp,
        use_first_diffusion_step=not args.use_all_diffusion_steps,
        selected_blocks=None,
        diag=args.diag,
    )


def worker(device_id: int, rows: list, args: argparse.Namespace) -> None:
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)

    image_dir = os.path.join(args.save_dir, 'all')
    os.makedirs(image_dir, exist_ok=True)

    pipe = load_pipeline(args.model_id, device)

    control = None
    records_dir = None
    if args.steer:
        control = build_control(args, device)
        diffusion_register_vector_controls_with_hooks(
            pipe.unet, control, model_type=DiffusionModelType.from_model('sd14'))
        if args.diag and args.controller != 'casteer':
            records_dir = os.path.join(args.save_dir, 'pid_records')
            os.makedirs(records_dir, exist_ok=True)

    for case_num, prompt, seed in tqdm(rows, desc=f'cuda:{device_id}',
                                       mininterval=5.0, miniters=5):
        out_path = os.path.join(image_dir, f'{case_num}.png')
        if os.path.exists(out_path):
            continue

        generator = torch.Generator(device=device).manual_seed(int(seed))
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=args.img_size,
                width=args.img_size,
                generator=generator,
            ).images[0]

        if control is not None:
            # Drain before reset() so this image's trace does not bleed into
            # the next one's, as run_with_steering.py does per batch.
            if records_dir is not None:
                records = control.drain_records()
                if records:
                    pd.DataFrame(records).to_csv(
                        os.path.join(records_dir, f'{case_num}.csv'), index=False)
            control.reset()

        image.save(out_path)

    print(f'[cuda:{device_id}] done ({len(rows)} rows)')


def write_config(args, n_rows: int) -> None:
    """Record what produced this directory, next to the images it produced."""
    os.makedirs(args.save_dir, exist_ok=True)
    config = {
        'artist': args.artist,
        'prompts_csv': args.prompts_csv,
        'n_prompts': n_rows,
        'model_id': args.model_id,
        'scheduler': 'DPMSolverMultistepScheduler',
        'num_steps': args.num_steps,
        'guidance_scale': args.guidance_scale,
        'img_size': args.img_size,
        'seed': 'per-row evaluation_seed',
        'steer': args.steer,
    }
    if args.steer:
        config.update({
            'controller': args.controller,
            'concept_path': args.concept_path,
            'intermediate_clipping': args.intermediate_clipping,
            'use_first_diffusion_step': not args.use_all_diffusion_steps,
        })
        if args.controller == 'casteer':
            config['steering_strength'] = args.steering_strength
        else:
            config.update({'kp': args.kp, 'ki': args.ki, 'kg': args.kg, 'kd': args.kd,
                           'integral_clamp': args.integral_clamp})
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def main(args):
    if args.steer and args.controller != 'casteer':
        # Kp is the PID analogue of beta, so --steering_strength stands in when
        # only one of the two is given (as in run_with_steering.py).
        if args.kp is None:
            args.kp = args.steering_strength
        if not args.intermediate_clipping:
            print('WARNING: --controller {} without --intermediate_clipping leaves the PID output '
                  'unclamped, so the I/D terms can inject the concept instead of erasing it'
                  .format(args.controller))
    if args.steer and args.concept_path is None:
        raise ValueError('--concept_path is required unless --no_steer is passed')

    df = load_artist_dataset(args.artist, args.prompts_csv)
    args.prompts_csv = args.prompts_csv or ARTIST_DATASETS[args.artist]
    rows = [(int(r['case_number']), str(r['prompt']), int(r['evaluation_seed']))
            for _, r in df.iterrows() if isinstance(r['prompt'], str)]

    setting = ('baseline (no steering)' if not args.steer else
               f'strength {args.steering_strength}' if args.controller == 'casteer' else
               f'{args.controller} kp={args.kp} ki={args.ki} kg={args.kg} kd={args.kd}')
    print(f'{args.artist}: {len(rows)} prompts, {setting} -> {args.save_dir}/all')

    write_config(args, len(rows))

    gpus = [int(g) for g in args.gpus.split(',') if g.strip() != '']
    if len(gpus) == 1:
        worker(gpus[0], rows, args)
        return

    # Round-robin so every worker gets a mix of erased/unerased prompts (the
    # CSV is grouped by artist, so a contiguous split would not).
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank, device_id in enumerate(gpus):
        chunk = rows[rank::len(gpus)]
        print(f'  cuda:{device_id} -> {len(chunk)} prompts')
        p = mp.Process(target=worker, args=(device_id, chunk, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    failed = [p.exitcode for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f'{len(failed)} worker(s) failed: {failed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artist', choices=list(ARTIST_DATASETS), required=True,
                        help='Selects the prompt set and the erased/unerased split')
    parser.add_argument('--prompts_csv', default=None,
                        help="Override the artist's default CSV (needs case_number/prompt/evaluation_seed)")
    parser.add_argument('--save_dir', required=True,
                        help='Images are written to {save_dir}/all/{case_number}.png')
    parser.add_argument('--gpus', default='0', help='Comma-separated CUDA device ids')

    parser.add_argument('--controller', choices=['casteer', *PID_CONTROLLERS], default='adaptive_kg',
                        help="'casteer' = Eq. 6; 'pid' = flat Kg; 'adaptive_kg' = Kg reweighted per block")
    parser.add_argument('--concept_path', default=None, help='Steering vectors .pt to erase')
    parser.add_argument('--steering_strength', type=float, default=2.0,
                        help='Beta for --controller casteer, and the --kp default for the others')
    parser.add_argument('--kp', type=float, default=None)
    parser.add_argument('--ki', type=float, default=0.0)
    parser.add_argument('--kg', type=float, default=0.0)
    parser.add_argument('--kd', type=float, default=0.0)
    parser.add_argument('--integral_clamp', type=float, default=None)
    parser.add_argument('--intermediate_clipping', action='store_true',
                        help='Non-negativity clamp (Eq. 6); for the PID laws it clamps the combined output')
    parser.add_argument('--use_all_diffusion_steps', action='store_true',
                        help="Per-step steering vectors instead of the first step's")
    parser.add_argument('--diag', action='store_true',
                        help='Write the per-block/per-step trace to {save_dir}/pid_records')
    parser.add_argument('--no_steer', dest='steer', action='store_false',
                        help='Generate the unsteered baseline through this same path')

    # Protocol constants. Changing any of them invalidates LPIPS against a
    # baseline directory generated before the change.
    parser.add_argument('--model_id', default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--img_size', type=int, default=512)

    main(parser.parse_args())
