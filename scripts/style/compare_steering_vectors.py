"""
Compare two families of cross-attention steering vectors for the same concept.

Both families claim to hold "the direction of artist X" in SD-v1.4's 16
cross-attention blocks, and both are built as normalize(mean(CA_pos) -
mean(CA_neg)) over 50 prompt pairs. They differ in how the pairs are written and
how the activations are pooled:

    CASteer/PID   scripts/diffusion/estimate_steering_vectors.py --mode style
                  pairs are "{ImageNet class}, {artist} style" vs "{ImageNet class}"
                  fp16 pipeline, default PNDM scheduler, seed 0 for every prompt

    TECA/SAFREE   SAFREE/style/style_erasure_TECA_final.py::build_steering_vectors
                  pairs come from a hand-written per-concept prompt list
                  fp32 pipeline, DPMSolver scheduler, seed varies per prompt

Both are applied the same way at inference: the step-0 vector is used for every
denoising step, so step 0 is the one that matters and the rest are diagnostic.

The question this answers is not "are the files the same" -- they are built by
different code and never will be. It is whether they point the same way. The
control that makes that answerable is the cross-artist comparison: if
same-artist/cross-family similarity is no higher than cross-artist/cross-family
similarity, the two families are not encoding the same concept, whatever the
absolute numbers look like.

    python scripts/style/compare_steering_vectors.py
    python scripts/style/compare_steering_vectors.py --step 0 --json out.json
"""
import argparse
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SAFREE_STYLE = os.path.join(os.path.dirname(_REPO_ROOT), 'SAFREE', 'style')

# (label, artist, family, path)
DEFAULT_SETS = [
    ('CASteer/PID  Van Gogh', 'Van Gogh', 'casteer',
     os.path.join(_REPO_ROOT, 'results', 'sd14', 'steering_vectors', 'Van Gogh.pt')),
    ('CASteer/PID  Kelly',    'Kelly',    'casteer',
     os.path.join(_REPO_ROOT, 'results', 'sd14', 'steering_vectors', 'Kelly McKernan.pt')),
    ('TECA/SAFREE  Van Gogh', 'Van Gogh', 'teca',
     os.path.join(_SAFREE_STYLE, 'sv_vangogh_final_beta1.pt')),
    ('TECA/SAFREE  Kelly',    'Kelly',    'teca',
     os.path.join(_SAFREE_STYLE, 'sv_kellymckernan_final_beta1.pt')),
]

# The 16 steered blocks in visitation order, as the hooks walk the UNet.
BLOCK_ORDER = [('down', i) for i in range(6)] + [('mid', 0)] + [('up', i) for i in range(9)]


def load(path: str) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def vec(sv: dict, step: int, place: str, idx: int) -> torch.Tensor:
    """One block's direction as a flat unit vector.

    CASteer stores [1, dim] (it keeps a head axis), TECA stores [dim]; flatten
    both and renormalize so a cosine is just a dot product.
    """
    t = sv[step][place][idx].detach().float().flatten()
    return t / t.norm().clamp(min=1e-8)


def block_cosines(a: dict, b: dict, step: int) -> list:
    out = []
    for place, idx in BLOCK_ORDER:
        try:
            out.append(float(torch.dot(vec(a, step, place, idx), vec(b, step, place, idx))))
        except (KeyError, IndexError):
            out.append(float('nan'))
    return out


def summarize(cos: list) -> dict:
    t = torch.tensor([c for c in cos if c == c])
    return {'mean': float(t.mean()), 'min': float(t.min()), 'max': float(t.max()),
            'median': float(t.median())}


def main(args):
    sets = []
    for label, artist, family, path in DEFAULT_SETS:
        if not os.path.exists(path):
            print(f'MISSING: {path}')
            continue
        sets.append((label, artist, family, load(path)))

    if len(sets) < 2:
        raise SystemExit('need at least two vector sets to compare')

    step = args.step
    print(f'\nAll comparisons at step {step} — the step both families actually apply.')
    print(f'Cosine similarity between unit block directions; 0.0 = orthogonal, 1.0 = identical.\n')

    # ---- full pairwise matrix, mean over the 16 blocks -------------------
    labels = [s[0] for s in sets]
    width = max(len(l) for l in labels) + 2
    print('Mean cosine over all 16 blocks')
    print(' ' * width + ''.join(f'{l.split()[-1][:9]:>11}' for l in labels))
    matrix = {}
    for la, aa, fa, sa in sets:
        row = []
        for lb, ab, fb, sb in sets:
            m = summarize(block_cosines(sa, sb, step))['mean']
            row.append(m)
            matrix[f'{la} | {lb}'] = round(m, 4)
        print(f'{la:<{width}}' + ''.join(f'{v:>11.3f}' for v in row))

    # ---- the comparison that was asked for -------------------------------
    print('\n\nSame artist, different family — do the two methods agree?')
    print('-' * 78)
    results = {}
    for artist in ('Van Gogh', 'Kelly'):
        a = next((s for s in sets if s[1] == artist and s[2] == 'casteer'), None)
        b = next((s for s in sets if s[1] == artist and s[2] == 'teca'), None)
        if not (a and b):
            continue
        cos = block_cosines(a[3], b[3], step)
        st = summarize(cos)
        results[artist] = {'summary': st, 'per_block': dict(
            zip([f'{p}{i}' for p, i in BLOCK_ORDER], [round(c, 4) for c in cos]))}
        print(f'\n{artist}:  mean {st["mean"]:+.3f}   median {st["median"]:+.3f}   '
              f'range [{st["min"]:+.3f}, {st["max"]:+.3f}]')
        print(f'{"":>10}' + ''.join(f'{p}{i:<2}'.rjust(8) for p, i in BLOCK_ORDER))
        print(f'{"cos":>10}' + ''.join(f'{c:>8.3f}' for c in cos))

    # ---- the control -----------------------------------------------------
    print('\n\nControl — cross-artist similarity (how concept-specific is each family?)')
    print('-' * 78)
    for family, name in (('casteer', 'CASteer/PID'), ('teca', 'TECA/SAFREE')):
        a = next((s for s in sets if s[1] == 'Van Gogh' and s[2] == family), None)
        b = next((s for s in sets if s[1] == 'Kelly' and s[2] == family), None)
        if a and b:
            st = summarize(block_cosines(a[3], b[3], step))
            results[f'{family}_cross_artist'] = st
            print(f'{name:<14} Van Gogh vs Kelly:  mean {st["mean"]:+.3f}  '
                  f'range [{st["min"]:+.3f}, {st["max"]:+.3f}]')

    # ---- per-block artist specificity ------------------------------------
    # Where the cross-artist cosine is near zero, that block encodes WHICH
    # artist; where it is high, the block encodes generic "painterly style"
    # that both artists share. This is the structure a per-block controller
    # like adaptive_kg is operating over, so it is worth reading directly.
    cs = {a: next((s[3] for s in sets if s[1] == a and s[2] == 'casteer'), None)
          for a in ('Van Gogh', 'Kelly')}
    tc = {a: next((s[3] for s in sets if s[1] == a and s[2] == 'teca'), None)
          for a in ('Van Gogh', 'Kelly')}
    if all(cs.values()) and all(tc.values()):
        print('\n\nPer-block artist specificity (cross-artist cosine, per family)')
        print('low = block encodes WHICH artist   high = block encodes generic style')
        print('-' * 78)
        print(f'{"block":<8}{"CASteer/PID":>13}{"TECA/SAFREE":>13}{"dim":>8}')
        spec = {}
        for place, i in BLOCK_ORDER:
            a = float(torch.dot(vec(cs['Van Gogh'], step, place, i),
                                vec(cs['Kelly'], step, place, i)))
            b = float(torch.dot(vec(tc['Van Gogh'], step, place, i),
                                vec(tc['Kelly'], step, place, i)))
            d = cs['Van Gogh'][step][place][i].flatten().shape[0]
            spec[f'{place}{i}'] = {'casteer': round(a, 4), 'teca': round(b, 4), 'dim': d}
            print(f'{place}{i:<7}{a:>13.3f}{b:>13.3f}{d:>8}')
        results['per_block_artist_specificity'] = spec

    # ---- step drift ------------------------------------------------------
    print('\n\nStep drift — how far each family\'s direction moves from its own step 0')
    print('(both apply step 0 to all 50 steps, so this is how much that assumption costs)')
    print('-' * 78)
    probe_steps = [1, 5, 10, 25, 49]
    print(f'{"":<24}' + ''.join(f'{"step "+str(s):>11}' for s in probe_steps))
    for label, artist, family, sv in sets:
        row = []
        for s in probe_steps:
            if s not in sv:
                row.append(float('nan'))
                continue
            cos = [float(torch.dot(vec(sv, 0, place, i), vec(sv, s, place, i)))
                   for place, i in BLOCK_ORDER]
            row.append(summarize(cos)['mean'])
        results[f'{label}_drift'] = {f'step{s}': round(v, 4) for s, v in zip(probe_steps, row)}
        print(f'{label:<24}' + ''.join(f'{v:>11.3f}' for v in row))

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'step': step, 'matrix': matrix, 'results': results}, f, indent=4)
        print(f'\nSaved -> {args.json}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--step', type=int, default=0,
                   help='Denoising step to compare (default 0 — the step both families apply)')
    p.add_argument('--json', default=None, help='Write the full numbers to this path')
    main(p.parse_args())
