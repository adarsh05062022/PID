"""
Compute CASteer steering vectors for Stable Diffusion 3 (MMDiT).

Same method as `estimate_steering_vectors.py` -- normalize(mean(A_pos) -
mean(A_neg)) over the same paired prompts -- but hooked at SD3's joint
attention instead of a UNet's cross-attention, since MMDiT has none. See
`core/sd3/hooks.py` for what is steered and why.

Kept as a separate entry point rather than a branch in the original so that
deleting `core/sd3/` and the two `*_sd3.py` scripts removes SD3 support
entirely.

    python scripts/diffusion/estimate_steering_vectors_sd3.py \
        --concept nudity --mode human-related --num_prompts 40 \
        --output_dir ./results/sd3/steering_vectors

Needs an env with diffusers >= 0.29 (StableDiffusion3Pipeline).
"""
import argparse
import os

import torch
import tqdm

from core.construct_prompts import get_prompts_concrete, get_prompts_human_related, get_prompts_style
from core.sd3 import IMG_SIZE, MODEL_ID, NUM_BLOCKS, NUM_STEPS, load_sd3, run_sd3, \
    sd3_register_vector_controls_with_hooks
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode


def collect_means(pipe, prompts, device, num_steps, img_size):
    """Run SD3 on a list of prompts and return mean joint-attention outputs per block."""
    stats = CrossAttentionOutputStatsCollector(
        token_aggregation_mode=TokenAggregationMode.AVERAGE,
        normalize=False,
    )
    hook_manager = sd3_register_vector_controls_with_hooks(pipe.transformer, stats)
    if hook_manager.block_count != NUM_BLOCKS:
        print(f'note: hooked {hook_manager.block_count} blocks, expected {NUM_BLOCKS} for SD3-medium')

    for prompt in tqdm.tqdm(prompts, desc='Collecting activations'):
        _ = run_sd3(pipe, prompt, seed=0, device=device, num_steps=num_steps, img_size=img_size)
        stats.reset()

    means = stats.means
    hook_manager.remove_hooks()
    return means


def compute_steering_vectors(pos_means, neg_means):
    """normalize(pos_mean - neg_mean) per block -- identical to the UNet script's."""
    steering_vectors = {}
    for step in pos_means:
        steering_vectors[step] = {}
        for place in pos_means[step]:
            steering_vectors[step][place] = []
            for layer_idx in range(len(pos_means[step][place])):
                sv = pos_means[step][place][layer_idx] - neg_means[step][place][layer_idx]
                sv = sv / sv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                steering_vectors[step][place].append(sv)
    return steering_vectors


def main(args):
    output_path = os.path.join(args.output_dir, f'{args.concept}.pt')
    if os.path.exists(output_path):
        print(f'File {output_path} already exists. Skipping.')
        return
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'concrete':
        prompts_pos, prompts_neg = get_prompts_concrete(
            num=args.num_prompts, concept_pos=args.concept, concept_neg=args.concept_neg)
    elif args.mode == 'style':
        prompts_pos, prompts_neg = get_prompts_style(
            num=args.num_prompts, concept_pos=args.concept, concept_neg=args.concept_neg)
    elif args.mode == 'human-related':
        prompts_pos, prompts_neg = get_prompts_human_related(
            concept_pos=args.concept, concept_neg=args.concept_neg)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    # human-related ignores --num_prompts (its prompt list is fixed); SD3 at
    # 28 steps is ~10x slower per image than the sdxl-turbo estimation path,
    # so allow trimming it here.
    if args.max_pairs:
        prompts_pos = prompts_pos[:args.max_pairs]
        prompts_neg = prompts_neg[:args.max_pairs]
    print(f"Generated {len(prompts_pos)} prompt pairs for concept '{args.concept}' (mode: {args.mode})")

    pipe = load_sd3(args.device, model_id=args.model_id, drop_t5=args.drop_t5)

    print('Collecting positive prompt activations...')
    pos_means = collect_means(pipe, prompts_pos, args.device, args.num_steps, args.img_size)
    print('Collecting negative prompt activations...')
    neg_means = collect_means(pipe, prompts_neg, args.device, args.num_steps, args.img_size)

    torch.save(compute_steering_vectors(pos_means, neg_means), output_path)
    print(f'Saved steering vectors to {output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--concept', type=str, required=True)
    p.add_argument('--concept_neg', type=str, default=None)
    p.add_argument('--mode', choices=['concrete', 'style', 'human-related'], required=True)
    p.add_argument('--num_prompts', type=int, default=50)
    p.add_argument('--max_pairs', type=int, default=0, help='0 = all; trims any mode (SD3 is slow)')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model_id', default=MODEL_ID)
    p.add_argument('--num_steps', type=int, default=NUM_STEPS)
    p.add_argument('--img_size', type=int, default=IMG_SIZE)
    p.add_argument('--drop_t5', action='store_true', help='free the T5 encoder (changes generations)')
    main(p.parse_args())
