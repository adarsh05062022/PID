"""
Cross-Kp summary of the unsafe_plus_safe Kp sweep (scripts/diffusion/run_kp_sweep.sh).

compare_step_block_error.py answers "CASteer vs our method at ONE Kp". This
script stacks every arm of the sweep into one table so the question becomes
"how does each controller's applied correction u(b,t) scale with Kp, and does
the gap between them survive when Kp is lowered?" -- u(b,t), not e(b,t), is
what the controller actually does, so it is the primary quantity here and
|e| is carried alongside as context.

Reads each arm's --diag trace (pid_records/*.csv) directly; nothing is
computed from images. Arms are discovered by directory name:

    <sweep_dir>/casteer_kp<kp>/pid_records                       (Ki=Kg=Kd=0)
    <sweep_dir>/adaptive_kg_kp<kp>_ki*_kg*_kd*/pid_records        (our method)

Outputs (all under --output_dir):
    kp_sweep_summary.csv     one row per (kp, method, group): mean u_out, u_raw,
                             |e|, signed e, clip stats, and ours/CASteer ratios
    kp_sweep_by_step.csv     (kp, method, group, diffusion_step) means
    kp_sweep_by_block.csv    (kp, method, group, block) means
    kp_sweep_per_prompt.csv  (kp, method, image_index, group, prompt) means
    u_by_step_kp_sweep.png   u vs step, one panel per group, hue = Kp,
                             dashed = CASteer / solid = ours
    e_by_step_kp_sweep.png   same for mean |e|
    mean_u_vs_kp.png         headline: mean u as a function of Kp per method x group

    python scripts/diffusion/summarize_kp_sweep.py \
        --sweep_dir results/sd14/unsafe_plus_safe_kp_sweep \
        --groups "unsafe:0-11,safe:12-16" \
        --output_dir results/sd14/unsafe_plus_safe_kp_sweep/analysis
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_step_block_error import BLOCK_ORDER, assign_groups, load_records, parse_groups  # noqa: E402

METHOD_LABELS = {'casteer': 'CASteer (Kp only)', 'ours': 'Our method (adaptive_kg)'}
METHOD_STYLE = {'casteer': '--', 'ours': '-'}
VALUE_COLS = ['mean_u_out', 'mean_u_raw', 'mean_abs_e', 'mean_e_signed']

# Kp is an ordered magnitude, so it gets one hue stepped light -> dark; the
# controller is carried by line style. Groups (unsafe / safe) are separate
# panels, and where they share an axis they take two categorical hues.
KP_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
# unsafe/safe is the nudity sweep's split, forget/retain the style sweep's
# (run_vangogh_kp_sweep.sh); the concept-present group takes blue either way.
GROUP_COLORS = {'unsafe': '#2a78d6', 'safe': '#eb6834',
                'forget': '#2a78d6', 'retain': '#eb6834', 'all': '#52514e'}
GRID_COLOR = '#e1e0d9'


def discover_arms(sweep_dir: str) -> list[tuple[float, str, str]]:
    """[(kp, method, run_dir)] for every arm that has a pid_records/ directory."""
    arms = []
    for run_dir in sorted(glob.glob(os.path.join(sweep_dir, '*'))):
        name = os.path.basename(run_dir)
        if not os.path.isdir(os.path.join(run_dir, 'pid_records')):
            continue
        m = re.match(r'^(casteer|adaptive_kg|pid)_kp([0-9.]+)', name)
        if not m:
            continue
        method = 'casteer' if m.group(1) == 'casteer' else 'ours'
        arms.append((float(m.group(2)), method, run_dir))
    if not arms:
        raise FileNotFoundError(f'no casteer_kp*/ or adaptive_kg_kp*/ arms with pid_records under {sweep_dir}')
    return sorted(arms)


def load_sweep(arms, groups) -> pd.DataFrame:
    frames = []
    for kp, method, run_dir in arms:
        df = assign_groups(load_records(os.path.join(run_dir, 'pid_records')), groups)
        df['kp'] = kp
        df['method'] = method
        df['run'] = os.path.basename(run_dir)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _with_all_group(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate every row under group='all' so pooled and per-group means come from one groupby."""
    pooled = df.copy()
    pooled['group'] = 'all'
    return pd.concat([df, pooled], ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_all_group(df)
    keys = ['kp', 'method', 'group']
    cols = [c for c in VALUE_COLS if c in df.columns]
    table = df.groupby(keys)[cols].mean()
    table['n_prompts'] = df.groupby(keys)['image'].nunique()
    table['n_visits'] = df.groupby(keys).size()
    if {'mean_u_out', 'mean_u_raw'} <= set(df.columns):
        # u_out = max(u_raw, 0), so u_out - u_raw is the concept-reinjecting mass
        # the actuator clip removed.
        table['frac_visits_negative_raw'] = df.groupby(keys)['mean_u_raw'].apply(lambda s: float((s < 0).mean()))
        table['mean_clipped_mass'] = (df['mean_u_out'] - df['mean_u_raw']).groupby([df[k] for k in keys]).mean()
    table = table.reset_index()

    # ours / CASteer at the same Kp and group -- the quantity the sweep is about.
    wide = table.pivot(index=['kp', 'group'], columns='method', values=['mean_u_out', 'mean_abs_e'])
    if {'casteer', 'ours'} <= set(table['method']):
        ratios = pd.DataFrame({
            'u_out_ratio_ours_over_casteer': wide[('mean_u_out', 'ours')] / wide[('mean_u_out', 'casteer')].clip(lower=1e-8),
            'abs_e_ratio_ours_over_casteer': wide[('mean_abs_e', 'ours')] / wide[('mean_abs_e', 'casteer')].clip(lower=1e-8),
        }).reset_index()
        table = table.merge(ratios, on=['kp', 'group'], how='left')
    return table.sort_values(['group', 'kp', 'method']).reset_index(drop=True)


def by_step_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_all_group(df)
    cols = [c for c in VALUE_COLS if c in df.columns]
    return df.groupby(['kp', 'method', 'group', 'diffusion_step'])[cols].mean().reset_index()


def by_block_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_all_group(df)
    cols = [c for c in VALUE_COLS if c in df.columns]
    table = df.groupby(['kp', 'method', 'group', 'block'])[cols].mean().reset_index()
    table['block'] = pd.Categorical(table['block'], categories=BLOCK_ORDER, ordered=True)
    return table.sort_values(['kp', 'method', 'group', 'block']).reset_index(drop=True)


def per_prompt_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ('mean_u_out', 'mean_abs_e') if c in df.columns]
    return (df.groupby(['kp', 'method', 'image_index', 'group', 'prompt'])[cols].mean()
              .reset_index().sort_values(['kp', 'method', 'image_index']).reset_index(drop=True))


def _style_axis(ax):
    ax.grid(alpha=0.6, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def plot_by_step(by_step: pd.DataFrame, value_col: str, value_label: str, groups: list[str], output_path: str):
    kps = sorted(by_step['kp'].unique())
    kp_color = {kp: KP_RAMP[i] for i, kp in enumerate(kps)}
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4.6), sharey=True, squeeze=False)
    for ax, group in zip(axes[0], groups):
        sub = by_step[by_step['group'] == group]
        for kp in kps:
            for method in ('casteer', 'ours'):
                s = sub[(sub['kp'] == kp) & (sub['method'] == method)]
                if s.empty:
                    continue
                ax.plot(s['diffusion_step'], s[value_col], METHOD_STYLE[method], color=kp_color[kp],
                        linewidth=1.6, label=f'Kp={kp:g} -- {METHOD_LABELS[method]}')
        ax.axhline(0, color='#c3c2b7', linewidth=0.6)
        ax.set_title(f'{value_label} -- {group} prompts')
        ax.set_xlabel('diffusion step')
        _style_axis(ax)
    axes[0][0].set_ylabel(f'{value_label} (mean over blocks & prompts)')
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mean_vs_kp(summary: pd.DataFrame, value_col: str, value_label: str, groups: list[str], output_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for group in groups:
        for method in ('casteer', 'ours'):
            s = summary[(summary['group'] == group) & (summary['method'] == method)].sort_values('kp')
            if s.empty:
                continue
            ax.plot(s['kp'], s[value_col], METHOD_STYLE[method], marker='o', markersize=5,
                    color=GROUP_COLORS.get(group, '#52514e'), linewidth=1.6,
                    label=f'{group} -- {METHOD_LABELS[method]}')
    ax.set_xticks(sorted(summary['kp'].unique()))
    ax.set_xlabel('Kp')
    ax.set_ylabel(f'{value_label} (mean over steps, blocks & prompts)')
    ax.set_title(f'{value_label} vs Kp')
    ax.legend(fontsize=8, frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', required=True,
                        help='Directory holding casteer_kp*/ and adaptive_kg_kp*/ arms, each with pid_records/')
    parser.add_argument('--groups', default=None,
                        help="Split prompts by image index, e.g. 'unsafe:0-11,safe:12-16'")
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    groups = parse_groups(args.groups)
    arms = discover_arms(args.sweep_dir)
    for kp, method, run_dir in arms:
        print(f'  kp={kp:<4g} {method:<8s} {run_dir}')
    df = load_sweep(arms, groups)

    summary = summary_table(df)
    by_step = by_step_table(df)
    summary.to_csv(os.path.join(args.output_dir, 'kp_sweep_summary.csv'), index=False)
    by_step.to_csv(os.path.join(args.output_dir, 'kp_sweep_by_step.csv'), index=False)
    by_block_table(df).to_csv(os.path.join(args.output_dir, 'kp_sweep_by_block.csv'), index=False)
    per_prompt_table(df).to_csv(os.path.join(args.output_dir, 'kp_sweep_per_prompt.csv'), index=False)

    panel_groups = list(groups) if groups else ['all']
    if 'mean_u_out' in df.columns:
        plot_by_step(by_step, 'mean_u_out', 'applied correction u (post-clip)', panel_groups,
                     os.path.join(args.output_dir, 'u_by_step_kp_sweep.png'))
        plot_mean_vs_kp(summary, 'mean_u_out', 'applied correction u', panel_groups,
                        os.path.join(args.output_dir, 'mean_u_vs_kp.png'))
    plot_by_step(by_step, 'mean_abs_e', 'mean |e|', panel_groups,
                 os.path.join(args.output_dir, 'e_by_step_kp_sweep.png'))

    show = ['kp', 'method', 'group', 'mean_u_out', 'mean_abs_e']
    if 'u_out_ratio_ours_over_casteer' in summary.columns:
        show.append('u_out_ratio_ours_over_casteer')
    print(summary[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'Wrote kp_sweep_*.csv and plots to {args.output_dir}')


if __name__ == '__main__':
    main()
