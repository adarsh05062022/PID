"""
Artist-style steering vectors contrasted against *other artists*, SD-v1.4.

`estimate_steering_vectors.py --mode style` writes the pair as

    pos   "{ImageNet class}, Van Gogh style"
    neg   "{ImageNet class}"

so the negative has no style at all. Write an activation as
`A(class) + g + d_artist`, where `g` is the generic "this is a styled painting"
component every artist shares; the difference is then `g + d_vangogh`, and
steering it away removes `g` too -- which is Picasso's and Rembrandt's style as
much as Van Gogh's. That is the collateral damage `Acc_u` / `LPIPS_u` measure in
`scripts/style/README.md`.

This script keeps the positive half identical and replaces the negative half
with the same ImageNet classes rendered in *other* artists' styles, pooled into
one mean:

    pos   "{ImageNet class}, Van Gogh style"              (num_prompts)
    neg   "{ImageNet class}, Pablo Picasso style"         (num_prompts x negatives)
          "{ImageNet class}, Andy Warhol style"
          "{ImageNet class}, Caravaggio style"
          "{ImageNet class}, Rembrandt style"

    v = normalize(mu_pos - mu_neg)  ~  d_vangogh - mean_a(d_a)

`g` cancels, leaving what makes Van Gogh *this* artist rather than a painter.
Note this only cancels it in the mean -- `<v, d_picasso>` is not forced to zero
for any individual artist.

The negatives pool correctly because every artist contributes the same class
list, and because `CrossAttentionOutputStatsCollector` sums over every prompt
and divides by the total count: pos and neg never have to be the same length,
and there is no per-pair correspondence to preserve.

Kept as a separate entry point, on the same reasoning as
`estimate_steering_vectors_sd3.py`: deleting this one file removes the
contrastive-negative build entirely, and `--mode style` keeps producing exactly
what it produced before.

    # the eval set's own retain artists -- optimizes Acc_u/LPIPS_u directly
    PYTHONPATH=. python scripts/diffusion/estimate_steering_vectors_style.py \
        --concept "Van Gogh" --negatives retain --num_prompts 50 --gpu 0 \
        --output_dir ./results/sd14/steering_vectors_retain

    # artists absent from both eval CSVs -- any gain there is generalization
    PYTHONPATH=. python scripts/diffusion/estimate_steering_vectors_style.py \
        --concept "Van Gogh" --negatives heldout --num_prompts 50 --gpu 0 \
        --output_dir ./results/sd14/steering_vectors_heldout

`--gpu` is not optional in practice. `init_pipeline_for_image_model` loads with
`device_map='balanced'`, which shards the UNet across every visible GPU and then
dies in the first residual add with "Expected all tensors to be on the same
device". Pinning one device is what `scripts/style/run_artist.sh` does too.

Both are worth building. `retain` gives the stronger number but is fit against
the very artists the preservation metrics are computed over, so on its own it
cannot tell a real gain apart from having been shown the answer.

Output is byte-compatible with `estimate_steering_vectors.py`, so
`generate_artist.py --concept_path` and every controller read it unchanged.
Cost is (1 + len(negatives)) x num_prompts images -- 250 for the defaults, about
15 minutes on one GPU. To check the vector moved the way it should before
spending that on a full 100-image eval, compare the cross-artist cosine against
the old vector with `scripts/style/compare_steering_vectors.py`.
"""
import argparse
import os
import sys

# CUDA_VISIBLE_DEVICES has to be set before torch reads the device list, and
# torch caches it during the imports below -- so --gpu is pulled off argv by
# hand here rather than waiting for parse_args().
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]

import torch
import tqdm

from core.construct_prompts import get_prompts_style
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.style_data import is_erased_prompt, load_artist_dataset
from core.utils import get_device, init_pipeline_for_image_model, run_image_model
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode

MODEL_NAME = 'sd14'

# Artists absent from both eval CSVs, each set chosen to sit in the same
# register as the artist it stands in for -- a painter's "generic style"
# centroid is not estimated by a set of digital illustrators, or the reverse.
HELDOUT = {
    'Van Gogh': ['Claude Monet', 'Salvador Dali', 'Katsushika Hokusai',
                 'Gustav Klimt', 'J.M.W. Turner'],
    'Kelly McKernan': ['Greg Rutkowski', 'Artgerm', 'Loish',
                       'James Jean', 'Alphonse Mucha'],
}


def resolve_negatives(concept, spec):
    """'retain' -> the eval set's other four artists, 'heldout' -> the list above."""
    if spec == ['retain']:
        artists = load_artist_dataset(concept)['artist'].astype(str).unique()
        return sorted(a for a in artists if not is_erased_prompt({'artist': a}, concept))
    if spec == ['heldout']:
        if concept not in HELDOUT:
            raise SystemExit(f'No held-out list for {concept!r}; pass artist names explicitly.')
        return HELDOUT[concept]
    return spec


def build_prompts(concept, negatives, num):
    """Positive half unchanged; negative half is the same classes in other styles."""
    prompts_pos, _ = get_prompts_style(num=num, concept_pos=concept)
    prompts_neg = []
    for artist in negatives:
        styled, _ = get_prompts_style(num=num, concept_pos=artist)
        prompts_neg.extend(styled)
    return prompts_pos, prompts_neg


def collect_means(pipeline, prompts, device):
    stats = CrossAttentionOutputStatsCollector(
        token_aggregation_mode=TokenAggregationMode.AVERAGE,
        normalize=False,
    )
    hook_manager = diffusion_register_vector_controls_with_hooks(
        pipeline.unet,
        stats,
        model_type=DiffusionModelType.from_model(MODEL_NAME),
    )

    for prompt in tqdm.tqdm(prompts, desc='Collecting activations'):
        _ = run_image_model(
            model_type=MODEL_NAME,
            pipe=pipeline,
            prompt=prompt,
            seed=0,
            device=device,
        )
        stats.reset()

    means = stats.means
    hook_manager.remove_hooks()
    return means


def compute_steering_vectors(pos_means, neg_means):
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

    negatives = resolve_negatives(args.concept, args.negatives)
    prompts_pos, prompts_neg = build_prompts(args.concept, negatives, args.num_prompts)

    print(f'target    {args.concept}')
    print(f'negatives {", ".join(negatives)}')
    print(f'prompts   {len(prompts_pos)} positive, {len(prompts_neg)} negative')
    print(f'          e.g. {prompts_pos[0]!r}  vs  {prompts_neg[0]!r}')

    pipeline = init_pipeline_for_image_model(model=MODEL_NAME)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    print('\nCollecting positive prompt activations...')
    pos_means = collect_means(pipeline, prompts_pos, device)
    print('Collecting negative prompt activations...')
    neg_means = collect_means(pipeline, prompts_neg, device)

    torch.save(compute_steering_vectors(pos_means, neg_means), output_path)
    print(f'Saved steering vectors to {output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--concept', required=True, help='Artist to erase, e.g. "Van Gogh"')
    p.add_argument('--negatives', nargs='+', default=['retain'],
                   help="'retain' (the eval set's other artists), 'heldout', or explicit names")
    p.add_argument('--num_prompts', type=int, default=50,
                   help='ImageNet classes per style; the negative set is this times len(negatives)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--gpu', default=None,
                   help='Single GPU index to pin; see the note above on device_map')
    main(p.parse_args())
