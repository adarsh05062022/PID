"""
Per-(diffusion_step, block) comparison of the controller's applied correction:
normal CASteer vs. our PID/adaptive_kg controller.

Both controllers act on the same signal at every visited block,

    e(b,t)      = <ca_out(b,t), b_hat(b)>                    # measured, setpoint 0
    u_raw(b,t)  = Kp*e + Ki*I(b,t) + Kg*I_global(t)[*w(b,t)] + Kd*D(b,t)
    u(b,t)      = max(u_raw(b,t), 0)                          # actuator limit (clip)
    ca_out_new  = ca_out - u(b,t) * b_hat(b)

so u(b,t) -- not e(b,t) -- is the intervention the controller actually makes.
The --diag trace logs both: `mean_u_out` is u after the clip (what is
subtracted from the activation) and `mean_u_raw` is the pre-clip sum, which
can be negative where the I/G/D terms would have pushed the concept back IN
and the clip blocked them. At Ki=Kg=Kd=0 the PID/adaptive_kg controllers are
bit-exact to CASteer's Eq. 6 (u = max(Kp*e, 0)), so a Kp-only run's
pid_records IS CASteer's correction trace -- point `--baseline_dir` at one.

Primary output is the u(b,t) grid; e(b,t) is written alongside as context.
Each quantity is compared per cell from each run's own --diag trace (not by
comparing images between runs).

The prompt set mixes regimes (unsafe_plus_safe: I2P/adversarial prompts where
the concept IS present and COCO captions where it is NOT), and the controller
should behave oppositely on the two -- so pass --groups to get a separate
grid per regime plus a `per_prompt.csv` and `*_by_group.png` trajectory
plot, instead of one blended average.

    python scripts/diffusion/compare_step_block_error.py \
        --baseline_dir results/sd14/unsafe_plus_safe/signed/casteer/pid_records \
        --baseline_label "CASteer (Kp=2)" \
        --method_dir results/sd14/unsafe_plus_safe/signed/pid_kd002/pid_records \
        --method_label "Our method (kd=0.02)" \
        --groups "unsafe:0-11,safe:12-16" \
        --output_dir results/sd14/unsafe_plus_safe/error_by_step_block/pid_kd002
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOCK_ORDER = (
    [f'down-{i}' for i in range(6)]
    + ['mid-0']
    + [f'up-{i}' for i in range(9)]
)

# (column, file prefix, axis label, can be negative)
METRICS = [
    ('mean_u_out', 'correction', 'applied correction u (post-clip)', False),
    ('mean_u_raw', 'correction_raw', 'raw u = P+I+G+D (pre-clip)', True),
    ('mean_abs_e', 'error', 'mean |e|', False),
    ('mean_e_signed', 'signed_error', 'mean signed e', True),
]


def load_records(records_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(records_dir, '*.csv')))
    if not paths:
        raise FileNotFoundError(f'no pid_records CSVs found in {records_dir}')
    run_dir = os.path.dirname(os.path.abspath(records_dir))
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        name = os.path.splitext(os.path.basename(path))[0]
        df['image'] = name
        # pid_records/00003-42.csv <-> <run_dir>/3/prompt.txt
        index = int(name.split('-')[0])
        df['image_index'] = index
        prompt_path = os.path.join(run_dir, str(index), 'prompt.txt')
        df['prompt'] = open(prompt_path).read().strip() if os.path.exists(prompt_path) else ''
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['block'] = df['place_in_unet'] + '-' + df['block_index'].astype(str)
    return df


def parse_groups(spec: str | None) -> dict[str, set[int]]:
    """'unsafe:0-11,safe:12-16' -> {'unsafe': {0..11}, 'safe': {12..16}}."""
    if not spec:
        return {}
    groups = {}
    for part in spec.split(','):
        name, rng = part.split(':')
        lo, hi = (int(x) for x in rng.split('-')) if '-' in rng else (int(rng), int(rng))
        groups[name.strip()] = set(range(lo, hi + 1))
    return groups


def assign_groups(df: pd.DataFrame, groups: dict[str, set[int]]) -> pd.DataFrame:
    df = df.copy()
    df['group'] = 'all'
    for name, indices in groups.items():
        df.loc[df['image_index'].isin(indices), 'group'] = name
    return df


def pivot_by_step_block(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean of value_col over images, indexed [block x diffusion_step]."""
    grid = df.groupby(['block', 'diffusion_step'])[value_col].mean().unstack('diffusion_step')
    order = [b for b in BLOCK_ORDER if b in grid.index]
    return grid.reindex(order)


def build_long_table(baseline_df, method_df, value_col: str) -> pd.DataFrame:
    b = baseline_df.groupby(['diffusion_step', 'block'])[value_col].mean().rename('baseline')
    m = method_df.groupby(['diffusion_step', 'block'])[value_col].mean().rename('method')
    joined = pd.concat([b, m], axis=1).reset_index()
    joined['diff'] = joined['method'] - joined['baseline']
    joined['pct_change'] = 100 * joined['diff'] / joined['baseline'].abs().clip(lower=1e-8)
    joined['block'] = pd.Categorical(joined['block'], categories=BLOCK_ORDER, ordered=True)
    return joined.sort_values(['block', 'diffusion_step'])


def _imshow_grid(ax, grid, cmap, vmin, vmax, title, ylabel='block'):
    im = ax.imshow(grid.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[grid.columns.min() - 0.5, grid.columns.max() + 0.5,
                            len(grid.index) - 0.5, -0.5])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return im


def plot_heatmaps(baseline_grid, method_grid, baseline_label, method_label,
                   value_label, output_path, signed=False):
    if signed:
        amax = max(np.nanmax(np.abs(baseline_grid.values)), np.nanmax(np.abs(method_grid.values)))
        cmap, vmin, vmax = 'RdBu_r', -amax, amax
    else:
        cmap, vmin = 'viridis', 0.0
        vmax = max(np.nanmax(baseline_grid.values), np.nanmax(method_grid.values))
    diff_grid = method_grid - baseline_grid
    dmax = np.nanmax(np.abs(diff_grid.values))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    im = _imshow_grid(axes[0], baseline_grid, cmap, vmin, vmax, f'{baseline_label} -- {value_label}')
    fig.colorbar(im, ax=axes[0], label=value_label)
    im = _imshow_grid(axes[1], method_grid, cmap, vmin, vmax, f'{method_label} -- {value_label}')
    fig.colorbar(im, ax=axes[1], label=value_label)
    im = _imshow_grid(axes[2], diff_grid, 'RdBu_r', -dmax, dmax,
                      f'{method_label} - {baseline_label}  (diff)')
    axes[2].set_xlabel('diffusion step')
    fig.colorbar(im, ax=axes[2], label=f'diff {value_label}')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_marginals(long_df, value_label, baseline_label, method_label, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    by_step = long_df.groupby('diffusion_step')[['baseline', 'method']].mean().reset_index()
    ax.plot(by_step['diffusion_step'], by_step['baseline'], label=baseline_label,
            color='#7f7f7f', linestyle='--')
    ax.plot(by_step['diffusion_step'], by_step['method'], label=method_label, color='#d62728')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlabel('diffusion step')
    ax.set_ylabel(f'{value_label} (mean over blocks & images)')
    ax.set_title(f'{value_label} over the denoising trajectory')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    by_block = long_df.groupby('block', observed=True)[['baseline', 'method']].mean().reindex(BLOCK_ORDER).dropna()
    x = np.arange(len(by_block))
    width = 0.4
    ax.bar(x - width / 2, by_block['baseline'], width, label=baseline_label, color='#7f7f7f')
    ax.bar(x + width / 2, by_block['method'], width, label=method_label, color='#d62728')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(by_block.index, rotation=90, fontsize=7)
    ax.set_ylabel(f'{value_label} (mean over steps & images)')
    ax.set_title(f'{value_label} by block')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def clip_stats(df: pd.DataFrame) -> dict:
    """How much of the pre-clip correction the actuator limit removed.

    u_out = max(u_raw, 0) per token, so mean_u_out - mean_u_raw is the mean
    magnitude of the negative ("re-inject the concept") part that was blocked.
    """
    clipped_mass = df['mean_u_out'] - df['mean_u_raw']
    return {
        'frac_visits_negative_raw': float((df['mean_u_raw'] < 0).mean()),
        'mean_clipped_mass': float(clipped_mass.mean()),
        'clipped_mass_by_block': clipped_mass.groupby(df['block']).mean().reindex(BLOCK_ORDER).dropna(),
    }


def metric_section(long_df, value_label, heading, prefix) -> list[str]:
    overall_baseline = long_df['baseline'].mean()
    overall_method = long_df['method'].mean()
    up_row = long_df.loc[long_df['diff'].idxmax()]
    down_row = long_df.loc[long_df['diff'].idxmin()]
    by_block = long_df.groupby('block', observed=True)['diff'].mean().sort_values(ascending=False)
    by_step = long_df.groupby('diffusion_step')['diff'].mean()

    rel = ''
    if abs(overall_baseline) > 1e-8:
        rel = f' ({100 * (overall_method - overall_baseline) / abs(overall_baseline):+.1f}%)'

    lines = [
        f'## {heading}',
        '',
        f'- Overall mean {value_label}: baseline = {overall_baseline:.4f}, method = {overall_method:.4f}{rel}',
        f'- Cell where method is HIGHEST above baseline: block={up_row["block"]}, '
        f'step={int(up_row["diffusion_step"])} '
        f'(baseline={up_row["baseline"]:.4f} -> method={up_row["method"]:.4f})',
        f'- Cell where method is LOWEST below baseline: block={down_row["block"]}, '
        f'step={int(down_row["diffusion_step"])} '
        f'(baseline={down_row["baseline"]:.4f} -> method={down_row["method"]:.4f})',
        '',
        'Mean diff (method - baseline) by block, largest first:',
        '',
    ]
    lines += [f'- {block}: {diff:+.4f}' for block, diff in by_block.items()]
    lines += [
        '',
        'Mean diff (method - baseline) by diffusion step:',
        '',
        f'- Early steps (0-9): {by_step.loc[by_step.index <= 9].mean():+.4f}',
        f'- Mid steps (10-39): {by_step.loc[(by_step.index > 9) & (by_step.index <= 39)].mean():+.4f}',
        f'- Late steps (40+): {by_step.loc[by_step.index > 39].mean():+.4f}',
        '',
        f'Files: `{prefix}_heatmaps.png`, `{prefix}_marginals.png`, `{prefix}_by_step_block.csv`.',
        '',
    ]
    return lines


def write_summary(sections, baseline_label, method_label, baseline_dir, method_dir,
                   n_baseline, n_method, baseline_clip, method_clip, output_dir):
    lines = [
        '# Step x Block Controller Comparison',
        '',
        f'Baseline ({baseline_label}): `{baseline_dir}` -- {n_baseline} images',
        f'Method ({method_label}): `{method_dir}` -- {n_method} images',
        '',
        'u(b,t) = max(Kp*e + Ki*I + Kg*I_global[*w] + Kd*D, 0) is what the controller '
        'subtracts along b_hat at block b, step t. Every number below is averaged over '
        'images, one value per (diffusion_step, block) cell.',
        '',
    ]
    for section in sections:
        lines += section

    if baseline_clip is not None and method_clip is not None:
        lines += [
            '## Actuator clipping (u = max(u_raw, 0))',
            '',
            f'- Fraction of block visits whose mean pre-clip u was negative: '
            f'baseline = {100 * baseline_clip["frac_visits_negative_raw"]:.1f}%, '
            f'method = {100 * method_clip["frac_visits_negative_raw"]:.1f}%',
            f'- Mean correction mass removed by the clip (mean_u_out - mean_u_raw): '
            f'baseline = {baseline_clip["mean_clipped_mass"]:.4f}, '
            f'method = {method_clip["mean_clipped_mass"]:.4f}',
            '',
            'Clipped mass by block (method):',
            '',
        ]
        lines += [f'- {block}: {v:.4f}' for block, v in method_clip['clipped_mass_by_block'].items()]
        lines.append('')

    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def per_image_table(baseline_df, method_df) -> pd.DataFrame:
    """One row per prompt: mean u / |e| over all (step, block) visits, per run."""
    cols = [c for c in ('mean_u_out', 'mean_abs_e') if c in baseline_df.columns and c in method_df.columns]
    b = baseline_df.groupby(['image_index', 'group', 'prompt'])[cols].mean().add_prefix('baseline_')
    m = method_df.groupby(['image_index', 'group', 'prompt'])[cols].mean().add_prefix('method_')
    table = pd.concat([b, m], axis=1).reset_index()
    if 'mean_u_out' in cols:
        table['u_ratio'] = table['method_mean_u_out'] / table['baseline_mean_u_out'].clip(lower=1e-8)
    return table.sort_values('image_index')


def plot_group_trajectories(baseline_df, method_df, baseline_label, method_label, value_col,
                            value_label, output_path):
    """mean value_col vs step, one line per (group, run) -- the discrimination view."""
    groups = [g for g in baseline_df['group'].unique() if g != 'all'] or ['all']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, g in enumerate(groups):
        for df, label, style in ((baseline_df, baseline_label, '--'), (method_df, method_label, '-')):
            sub = df[df['group'] == g]
            by_step = sub.groupby('diffusion_step')[value_col].mean()
            ax.plot(by_step.index, by_step.values, style, color=colors[i % len(colors)],
                    label=f'{g} ({sub["image"].nunique()} prompts) -- {label}')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlabel('diffusion step')
    ax.set_ylabel(f'{value_label} (mean over blocks & prompts)')
    ax.set_title(f'{value_label} by prompt group')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_all(baseline_dir, method_dir, baseline_label, method_label, output_dir, groups):
    os.makedirs(output_dir, exist_ok=True)
    baseline_df = assign_groups(load_records(baseline_dir), groups)
    method_df = assign_groups(load_records(method_dir), groups)

    per_image_table(baseline_df, method_df).to_csv(os.path.join(output_dir, 'per_prompt.csv'), index=False)

    if groups:
        for col, prefix, label, _ in METRICS:
            if col in baseline_df.columns and col in method_df.columns:
                plot_group_trajectories(baseline_df, method_df, baseline_label, method_label,
                                        col, label, os.path.join(output_dir, f'{prefix}_by_group.png'))
        for name in groups:
            run_comparison(baseline_df[baseline_df['group'] == name], method_df[method_df['group'] == name],
                           baseline_dir, method_dir, f'{baseline_label} [{name}]', f'{method_label} [{name}]',
                           os.path.join(output_dir, name))

    run_comparison(baseline_df, method_df, baseline_dir, method_dir, baseline_label, method_label,
                   os.path.join(output_dir, 'all') if groups else output_dir)


def run_comparison(baseline_df, method_df, baseline_dir, method_dir, baseline_label, method_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sections = []
    written = []
    for col, prefix, label, signed in METRICS:
        if col not in baseline_df.columns or col not in method_df.columns:
            continue
        long_df = build_long_table(baseline_df, method_df, col)
        long_df.to_csv(os.path.join(output_dir, f'{prefix}_by_step_block.csv'), index=False)
        plot_heatmaps(pivot_by_step_block(baseline_df, col), pivot_by_step_block(method_df, col),
                      baseline_label, method_label, label,
                      os.path.join(output_dir, f'{prefix}_heatmaps.png'), signed=signed)
        plot_marginals(long_df, label, baseline_label, method_label,
                       os.path.join(output_dir, f'{prefix}_marginals.png'))
        sections.append(metric_section(long_df, label, f'{label}  [{col}]', prefix))
        written.append(prefix)

    have_u = {'mean_u_out', 'mean_u_raw'} <= set(baseline_df.columns) & set(method_df.columns)
    baseline_clip = clip_stats(baseline_df) if have_u else None
    method_clip = clip_stats(method_df) if have_u else None

    write_summary(sections, baseline_label, method_label, baseline_dir, method_dir,
                  baseline_df['image'].nunique(), method_df['image'].nunique(),
                  baseline_clip, method_clip, output_dir)
    print(f'Wrote {", ".join(written)} (heatmaps/marginals/csv) and summary.md to {output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', required=True,
                         help='pid_records dir for a Kp-only (Ki=Kg=Kd=0) run -- bit-exact to plain CASteer')
    parser.add_argument('--baseline_label', default='CASteer')
    parser.add_argument('--method_dir', required=True,
                         help='pid_records dir for the real-gains run being compared ("our method")')
    parser.add_argument('--method_label', default='Our method')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--groups', default=None,
                         help="Split prompts by image index, e.g. 'unsafe:0-11,safe:12-16'. Each group "
                              "gets its own subdir; the pooled run goes to <output_dir>/all/")
    args = parser.parse_args()

    run_all(args.baseline_dir, args.method_dir, args.baseline_label, args.method_label,
            args.output_dir, parse_groups(args.groups))


if __name__ == '__main__':
    main()
