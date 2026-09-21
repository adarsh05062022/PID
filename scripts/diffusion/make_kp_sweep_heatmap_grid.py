"""
One figure for the whole Kp sweep: rows = Kp, columns = CASteer | Ours | diff,
for the unsafe and safe prompt groups side by side. Each cell is the
(block x diffusion_step) grid of a --diag column (default mean_u_out, the
applied correction), averaged over the group's prompts -- the same panels
compare_step_block_error.py writes per Kp, stacked so the sweep can be read
top to bottom with one shared color scale per column.

    python scripts/diffusion/make_kp_sweep_heatmap_grid.py \
        --sweep_dir results/sd14/unsafe_plus_safe_kp_sweep \
        --groups "unsafe:0-11,safe:12-16" \
        --metric mean_u_out --output results/sd14/unsafe_plus_safe_kp_sweep/analysis/grid_u_kp_sweep.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_step_block_error import assign_groups, load_records, parse_groups, pivot_by_step_block

KPS = ('0.5', '1.0', '1.5', '2.0')
LABELS = {
    'mean_u_out': 'applied correction u (post-clip)',
    'mean_u_raw': 'raw u = P+I+G+D (pre-clip)',
    'mean_abs_e': 'mean |e|',
    'mean_e_signed': 'mean signed e',
}


def arm_dirs(sweep_dir: str, kp: str) -> tuple[str, str]:
    casteer = os.path.join(sweep_dir, f'casteer_kp{kp}', 'pid_records')
    ours = os.path.join(sweep_dir, f'adaptive_kg_kp{kp}_ki0.01_kg0.01_kd0.02', 'pid_records')
    return casteer, ours


def _draw(ax, grid, cmap, vmin, vmax, title):
    im = ax.imshow(grid.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[grid.columns.min() - 0.5, grid.columns.max() + 0.5, len(grid.index) - 0.5, -0.5])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=6)
    ax.set_title(title, fontsize=10)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir', required=True)
    p.add_argument('--groups', default='unsafe:0-11,safe:12-16')
    p.add_argument('--metric', default='mean_u_out', choices=list(LABELS))
    p.add_argument('--output', required=True)
    args = p.parse_args()

    groups = parse_groups(args.groups)
    group_names = list(groups)
    signed = args.metric in ('mean_u_raw', 'mean_e_signed')

    # grids[kp][group] = (casteer_grid, ours_grid)
    grids = {}
    for kp in KPS:
        c_dir, o_dir = arm_dirs(args.sweep_dir, kp)
        c_df = assign_groups(load_records(c_dir), groups)
        o_df = assign_groups(load_records(o_dir), groups)
        grids[kp] = {}
        for g in group_names:
            grids[kp][g] = (pivot_by_step_block(c_df[c_df['group'] == g], args.metric),
                            pivot_by_step_block(o_df[o_df['group'] == g], args.metric))

    # One color scale per group, shared across Kp rows so the sweep is comparable.
    scale = {}
    for g in group_names:
        vals = np.concatenate([np.concatenate([c.values.ravel(), o.values.ravel()])
                               for c, o in (grids[kp][g] for kp in KPS)])
        diffs = np.concatenate([(o - c).values.ravel() for c, o in (grids[kp][g] for kp in KPS)])
        amax = float(np.nanmax(np.abs(vals)))
        scale[g] = {
            'cmap': 'RdBu_r' if signed else 'viridis',
            'vmin': -amax if signed else 0.0,
            'vmax': amax if signed else float(np.nanmax(vals)),
            'dmax': float(np.nanmax(np.abs(diffs))),
        }

    ncols = 3 * len(group_names)
    fig, axes = plt.subplots(len(KPS), ncols, figsize=(5.2 * ncols, 3.0 * len(KPS)), sharex=True)
    label = LABELS[args.metric]

    for r, kp in enumerate(KPS):
        for gi, g in enumerate(group_names):
            c, o = grids[kp][g]
            s = scale[g]
            n = groups[g]
            base = 3 * gi
            im_abs = _draw(axes[r, base], c, s['cmap'], s['vmin'], s['vmax'],
                           f'[{g}, {len(n)} prompts] CASteer Kp={kp}')
            _draw(axes[r, base + 1], o, s['cmap'], s['vmin'], s['vmax'],
                  f'[{g}] Ours Kp={kp} (ki=kg=0.01, kd=0.02)')
            im_diff = _draw(axes[r, base + 2], o - c, 'RdBu_r', -s['dmax'], s['dmax'],
                            f'[{g}] Ours - CASteer, Kp={kp}')
            if r == len(KPS) - 1:
                for k in range(3):
                    axes[r, base + k].set_xlabel('diffusion step')
            if gi == 0:
                axes[r, 0].set_ylabel('block')
        # colorbars once per group, on the last row
        if r == len(KPS) - 1:
            for gi, g in enumerate(group_names):
                fig.colorbar(im_abs if gi == len(group_names) - 1 else
                             axes[r, 3 * gi].images[0], ax=axes[:, 3 * gi:3 * gi + 2].ravel().tolist(),
                             label=f'{label} [{g}]', shrink=0.6, pad=0.01)
                fig.colorbar(axes[r, 3 * gi + 2].images[0], ax=axes[:, 3 * gi + 2].ravel().tolist(),
                             label=f'diff {label} [{g}]', shrink=0.6, pad=0.01)

    fig.suptitle(f'Kp sweep -- {label}, per (block, diffusion step), mean over prompts', fontsize=13)
    fig.savefig(args.output, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
