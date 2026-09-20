"""
Adaptivity check for core/adaptive_kg_pid_steering.py's control law

    u = Kp*e + Ki*I(b,t) + Kg*I_global(t)*kg_weight(b,t) + Kd*D(b,t)

At Ki=Kg=Kd=0 this is bit-exact to CASteer's fixed-Kp Eq. 6: every block and
timestep gets the exact same Kp multiplier on its own error, so the applied
coefficient only moves because the measured error moves, not because the
controller adapts. Turning Ki/Kg/Kd on is supposed to make the multiplier
itself context-dependent. This script checks that directly from --diag
output, using three checks computed from a SINGLE controller run's
pid_records/*.csv (comparing two runs' images would confound "did the gain
adapt" with "steering changed the images, so the measured error diverged
too" -- see the discussion this script followed from):

  1. magnitude -- what fraction of |u| comes from the non-P terms (I+G+D),
                  and does that fraction hold steady or move across
                  timestep/block?
  2. block     -- does kg_weight (the block-aware reweighting of the global
                  term, see adaptive_kg_pid_steering.py's module docstring)
                  actually vary across blocks within a timestep, or does it
                  collapse to ~1 everywhere?
  3. temporal  -- does I_global(t) move over the ~50 denoising steps, or
                  stay pinned near 0?

The Ki=Kg=Kd=0 run is read too, purely as a sanity check that magnitude is
exactly zero there -- confirming the "collapses to CASteer" claim before
trusting the PID run's numbers.

    python scripts/diffusion/analyze_pid_adaptivity.py \
        --baseline_dir results/sd14/unsafe_plus_safe/kp_only_check/adaptive_kg/pid_records \
        --pid_dir results/sd14/unsafe_plus_safe/adaptive_kg_kp2.0_ki0.01_kg0.01_kd0.02/pid_records \
        --output_dir results/sd14/unsafe_plus_safe/adaptivity_check
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

EPS = 1e-8


def load_records(records_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(records_dir, '*.csv')))
    if not paths:
        raise FileNotFoundError(f'no pid_records CSVs found in {records_dir}')
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df['image'] = os.path.splitext(os.path.basename(path))[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['block'] = df['place_in_unet'] + '-' + df['block_index'].astype(str)
    return df


def check_magnitude(df: pd.DataFrame) -> dict:
    """Fraction of |u| contributed by I+G+D, overall / by step / by block --
    the direct test of whether the non-P terms are large enough to matter."""
    non_p = df['mean_abs_i_out'] + df['mean_abs_g_out'] + df['mean_abs_d_out']
    total = df['mean_abs_p_out'] + non_p
    frac = non_p / (total + EPS)
    d = df.assign(non_p_fraction=frac)

    by_step = d.groupby('diffusion_step')['non_p_fraction'].mean().reset_index()
    by_block = d.groupby('block')['non_p_fraction'].mean().reset_index()

    return {
        'overall_mean': float(frac.mean()),
        'overall_max': float(frac.max()),
        'by_step': by_step,
        'by_block': by_block,
    }


def check_block_adaptivity(df: pd.DataFrame) -> dict:
    """Spread of kg_weight ACROSS BLOCKS at a fixed (image, timestep). 0
    everywhere would mean every block gets an identical multiplier despite
    the per-block reweighting math."""
    d = df.dropna(subset=['kg_weight'])
    within_step_std = (
        d.groupby(['image', 'diffusion_step'])['kg_weight']
        .std()
        .reset_index(name='within_step_std')
    )
    by_step = within_step_std.groupby('diffusion_step')['within_step_std'].mean().reset_index()

    return {
        'overall_range': (float(d['kg_weight'].min()), float(d['kg_weight'].max())),
        'mean_within_step_std': float(within_step_std['within_step_std'].mean()),
        'values': d['kg_weight'],
        'by_step': by_step,
    }


def check_temporal_adaptivity(df: pd.DataFrame) -> dict:
    """Trajectory of I_global(t), identical across blocks within a timestep
    by construction, over the ~50 denoising steps."""
    traj = df.groupby(['image', 'diffusion_step'])['i_global'].first().reset_index()
    by_step = traj.groupby('diffusion_step')['i_global'].agg(['mean', 'std']).reset_index()
    term_by_step = df.groupby('diffusion_step')[
        ['mean_abs_p_out', 'mean_abs_i_out', 'mean_abs_g_out', 'mean_abs_d_out']
    ].mean().reset_index()

    return {
        'max_abs_i_global': float(traj['i_global'].abs().max()),
        'final_mean_abs_i_global': float(traj.groupby('image')['i_global'].last().abs().mean()),
        'by_step': by_step,
        'term_by_step': term_by_step,
    }


def plot_magnitude(baseline_mag: dict, pid_mag: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(pid_mag['by_step']['diffusion_step'], pid_mag['by_step']['non_p_fraction'],
            label='PID run', color='#d62728')
    ax.plot(baseline_mag['by_step']['diffusion_step'], baseline_mag['by_step']['non_p_fraction'],
            label='Ki=Kg=Kd=0 (baseline)', color='#7f7f7f', linestyle='--')
    ax.set_xlabel('diffusion step')
    ax.set_ylabel('non-P fraction of |u|  =  (I+G+D) / (P+I+G+D)')
    ax.set_title('Magnitude: non-P contribution over time')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    blocks = pid_mag['by_block'].sort_values('block')
    ax.bar(blocks['block'], blocks['non_p_fraction'], color='#d62728')
    ax.set_xlabel('block')
    ax.set_ylabel('mean non-P fraction of |u|')
    ax.set_title('Magnitude: non-P contribution by block (PID run)')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'magnitude_by_step_and_block.png'), dpi=150)
    plt.close(fig)


def plot_block_adaptivity(baseline_block: dict, pid_block: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.hist(baseline_block['values'], bins=40, alpha=0.6, label='Ki=Kg=Kd=0 (baseline)', color='#7f7f7f')
    ax.hist(pid_block['values'], bins=40, alpha=0.6, label='PID run', color='#d62728')
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1, label='kg_weight = 1 (flat/uniform share)')
    ax.set_xlabel('kg_weight')
    ax.set_ylabel('count (block visits)')
    ax.set_title('kg_weight distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pid_block['by_step']['diffusion_step'], pid_block['by_step']['within_step_std'],
            label='PID run', color='#d62728')
    ax.plot(baseline_block['by_step']['diffusion_step'], baseline_block['by_step']['within_step_std'],
            label='Ki=Kg=Kd=0 (baseline)', color='#7f7f7f', linestyle='--')
    ax.set_xlabel('diffusion step')
    ax.set_ylabel('std(kg_weight) across blocks, same timestep')
    ax.set_title('Block adaptivity: cross-block spread of kg_weight')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'block_adaptivity.png'), dpi=150)
    plt.close(fig)


def plot_temporal_adaptivity(baseline_temp: dict, pid_temp: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for temp, label, color in ((pid_temp, 'PID run', '#d62728'), (baseline_temp, 'Ki=Kg=Kd=0 (baseline)', '#7f7f7f')):
        by_step = temp['by_step']
        ax.plot(by_step['diffusion_step'], by_step['mean'], label=label, color=color,
                linestyle='-' if label == 'PID run' else '--')
        ax.fill_between(by_step['diffusion_step'], by_step['mean'] - by_step['std'],
                         by_step['mean'] + by_step['std'], color=color, alpha=0.15)
    ax.set_xlabel('diffusion step')
    ax.set_ylabel('I_global(t)  (mean +/- std over images)')
    ax.set_title('Temporal adaptivity: I_global trajectory')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    term_by_step = pid_temp['term_by_step']
    for col, label, color in (
        ('mean_abs_p_out', 'P', '#1f77b4'),
        ('mean_abs_i_out', 'I', '#2ca02c'),
        ('mean_abs_g_out', 'G', '#d62728'),
        ('mean_abs_d_out', 'D', '#9467bd'),
    ):
        ax.plot(term_by_step['diffusion_step'], term_by_step[col], label=label, color=color)
    ax.set_yscale('log')
    ax.set_xlabel('diffusion step')
    ax.set_ylabel('mean |term| (log scale)')
    ax.set_title('P/I/G/D term magnitudes over time (PID run)')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'temporal_adaptivity.png'), dpi=150)
    plt.close(fig)


def write_summary(baseline_dir, pid_dir, n_baseline, n_pid,
                   baseline_mag, pid_mag, baseline_block, pid_block, baseline_temp, pid_temp,
                   output_dir):
    summary = {
        'baseline_dir': baseline_dir,
        'pid_dir': pid_dir,
        'n_baseline_images': n_baseline,
        'n_pid_images': n_pid,
        'magnitude': {
            'baseline_overall_mean_non_p_fraction': baseline_mag['overall_mean'],
            'pid_overall_mean_non_p_fraction': pid_mag['overall_mean'],
            'pid_overall_max_non_p_fraction': pid_mag['overall_max'],
        },
        'block_adaptivity': {
            'baseline_mean_within_step_std': baseline_block['mean_within_step_std'],
            'pid_mean_within_step_std': pid_block['mean_within_step_std'],
            'pid_kg_weight_range': pid_block['overall_range'],
        },
        'temporal_adaptivity': {
            'baseline_max_abs_i_global': baseline_temp['max_abs_i_global'],
            'pid_max_abs_i_global': pid_temp['max_abs_i_global'],
            'pid_final_mean_abs_i_global': pid_temp['final_mean_abs_i_global'],
        },
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    def verdict(value, threshold, adaptive_msg, flat_msg):
        return adaptive_msg if value > threshold else flat_msg

    md = f"""# PID Adaptivity Check

Baseline (Ki=Kg=Kd=0): `{baseline_dir}` -- {n_baseline} images
PID run: `{pid_dir}` -- {n_pid} images

Each check is computed from a single run's `--diag` trace
(`pid_records/*.csv`), not by comparing images between runs -- see the
script docstring for why a cross-run image comparison would confound
"did the gain adapt" with "did steering change the images enough that the
measured error itself diverged."

## 1. Magnitude -- is the non-P contribution big enough to matter?

`non_p_fraction = (|I_out|+|G_out|+|D_out|) / (|P_out|+|I_out|+|G_out|+|D_out|)`

- Baseline: mean = {baseline_mag['overall_mean']:.6f} (expected ~0 -- confirms the
  Ki=Kg=Kd=0 reduction really is inert before trusting the PID numbers below)
- PID run: mean = {pid_mag['overall_mean']:.4f}, max = {pid_mag['overall_max']:.4f}
- {verdict(pid_mag['overall_mean'], 0.01,
           "Non-P terms contribute a non-trivial share of |u| on average.",
           "Non-P terms contribute well under 1% of |u| on average -- at these gains the "
           "controller is numerically close to plain CASteer even though I/G/D are nonzero.")}
- See `magnitude_by_step_and_block.png`: left panel shows whether the fraction moves
  over the denoising trajectory, right panel shows whether it differs by block.

## 2. Block adaptivity -- does kg_weight vary across blocks at a fixed timestep?

- Baseline mean within-timestep std(kg_weight) = {baseline_block['mean_within_step_std']:.4f}
- PID run mean within-timestep std(kg_weight) = {pid_block['mean_within_step_std']:.4f}
- PID run kg_weight range = {pid_block['overall_range'][0]:.4f} to {pid_block['overall_range'][1]:.4f}
  (1.0 = every block getting an identical, uniform share of the global term)
- kg_weight's formula (`|e(b,t)| / mean_b(|e(b',t-1)|)`) does not depend on kg_gain, so
  baseline and PID show a similar spread here -- this checks whether the reweighting
  MECHANISM is non-trivial. Whether it actually moves the output depends on kg_gain,
  which is what check 1's `mean_abs_g_out` captures.
- {verdict(pid_block['mean_within_step_std'], 0.05,
           "kg_weight spreads meaningfully across blocks within a timestep -- the "
           "per-block reweighting is not collapsing to a flat multiplier.",
           "kg_weight barely spreads across blocks within a timestep -- blocks are "
           "getting close to the same share of the global term regardless of the "
           "per-block reweighting math.")}
- See `block_adaptivity.png`.

## 3. Temporal adaptivity -- does I_global(t) move over the denoising trajectory?

- Baseline max |I_global| = {baseline_temp['max_abs_i_global']:.4f}
- PID run max |I_global| = {pid_temp['max_abs_i_global']:.4f}, final-step mean |I_global| = {pid_temp['final_mean_abs_i_global']:.4f}
- I_global accumulation also does not depend on kg_gain (only whether it's ever multiplied
  into g_out does), so baseline and PID trajectories should track closely at early steps
  and are expected to diverge later only insofar as the two runs' images themselves diverge.
- {verdict(pid_temp['max_abs_i_global'], 0.1,
           "I_global moves substantially away from 0 over the run -- the cross-block "
           "integral is accumulating real signal, not staying pinned near its initial value.",
           "I_global stays close to 0 throughout -- the cross-block integral term has "
           "little to act on at this scale.")}
- See `temporal_adaptivity.png` (right panel also shows P/I/G/D relative magnitudes over time).
"""
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(md)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', required=True,
                         help='pid_records dir for a Ki=Kg=Kd=0 run (the CASteer-equivalence check)')
    parser.add_argument('--pid_dir', required=True,
                         help='pid_records dir for the real-gains run being tested for adaptivity')
    parser.add_argument('--output_dir', required=True, help='where to write plots/ and summary.{json,md}')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_df = load_records(args.baseline_dir)
    pid_df = load_records(args.pid_dir)

    baseline_mag = check_magnitude(baseline_df)
    pid_mag = check_magnitude(pid_df)
    baseline_block = check_block_adaptivity(baseline_df)
    pid_block = check_block_adaptivity(pid_df)
    baseline_temp = check_temporal_adaptivity(baseline_df)
    pid_temp = check_temporal_adaptivity(pid_df)

    plot_magnitude(baseline_mag, pid_mag, args.output_dir)
    plot_block_adaptivity(baseline_block, pid_block, args.output_dir)
    plot_temporal_adaptivity(baseline_temp, pid_temp, args.output_dir)

    summary = write_summary(
        args.baseline_dir, args.pid_dir,
        baseline_df['image'].nunique(), pid_df['image'].nunique(),
        baseline_mag, pid_mag, baseline_block, pid_block, baseline_temp, pid_temp,
        args.output_dir,
    )

    print(json.dumps(summary, indent=2))
    print(f'\nWrote plots and summary.{{json,md}} to {args.output_dir}')


if __name__ == '__main__':
    main()
