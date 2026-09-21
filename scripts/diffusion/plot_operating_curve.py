"""
Kp-sweep operating curve from analysis/kp_sweep_summary.csv: for each arm, the
mean applied u on the target-concept prompts (x) against the mean applied u on
the preserved prompts (y), one point per Kp. Also fits u ~ a*Kp + b per arm and
prints the Kp offset at which ours matches CASteer on the target group, with
the preserved-side u each pays at that matched point.

    python scripts/diffusion/plot_operating_curve.py \
        --summary results/sd14/vangogh_kp_sweep/analysis/kp_sweep_summary.csv \
        --target forget --preserve retain \
        --output results/sd14/vangogh_kp_sweep/analysis/u_operating_curve.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--target', required=True, help='group name of the concept being erased')
    p.add_argument('--preserve', required=True, help='group name of the prompts to leave alone')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    s = pd.read_csv(args.summary)
    piv = s.pivot_table(index='kp', columns=['method', 'group'], values='mean_u_out')
    kps = piv.index.values
    n = s.groupby('group')['n_prompts'].first()

    print(piv[[('casteer', args.target), ('ours', args.target),
               ('casteer', args.preserve), ('ours', args.preserve)]].round(3).to_string(), '\n')

    fits = {}
    for g in (args.target, args.preserve):
        a_c, b_c = np.polyfit(kps, piv[('casteer', g)].values, 1)
        a_o, b_o = np.polyfit(kps, piv[('ours', g)].values, 1)
        fits[g] = (a_c, b_c, a_o, b_o)
        print(f'{g:9s}: CASteer u ~ {a_c:.3f}*Kp {b_c:+.3f}   ours u ~ {a_o:.3f}*Kp {b_o:+.3f}'
              f'   -> ours@Kp ~= CASteer@(Kp + {(b_o - b_c) / a_c:.2f})')

    # For each ours point, the Kp at which CASteer's fitted target-u equals it.
    a_c, b_c, _, _ = fits[args.target]
    a_cp, b_cp, _, _ = fits[args.preserve]
    print(f'\nMatched on {args.target} u (ours @Kp vs CASteer @ fitted Kp*), {args.preserve}-side u each pays:')
    for kp in kps:
        ou, op = piv.loc[kp, ('ours', args.target)], piv.loc[kp, ('ours', args.preserve)]
        kp_star = (ou - b_c) / a_c
        cp_star = a_cp * kp_star + b_cp
        print(f'  ours Kp={kp}: {args.target} {ou:.3f}, {args.preserve} {op:.3f}'
              f'   |   CASteer Kp*={kp_star:.2f}: {args.preserve} {cp_star:.3f}'
              f'   -> {args.preserve} u {100 * (op / cp_star - 1):+.1f}%')

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for m, col, mk, label in (('casteer', '#7f7f7f', 's', 'CASteer (Kp only)'),
                              ('ours', '#d62728', 'o', 'Ours (ki=kg=0.01, kd=0.02)')):
        x, y = piv[(m, args.target)], piv[(m, args.preserve)]
        ax.plot(x, y, '-', color=col, marker=mk, label=label)
        for kp, xi, yi in zip(kps, x, y):
            ax.annotate(f'Kp={kp}', (xi, yi), textcoords='offset points', xytext=(6, -10), fontsize=8, color=col)
    ax.set_xlabel(f'mean applied u on {args.target.upper()} prompts ({n[args.target]})')
    ax.set_ylabel(f'mean applied u on {args.preserve.upper()} prompts ({n[args.preserve]})')
    ax.set_title(f'Operating curve: {args.preserve}-side correction paid for a given {args.target} correction')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
