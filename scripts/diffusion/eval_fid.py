"""
FID of a run_with_steering.py --prompts_csv output directory against a real
reference folder, using clean-fid (same library core/eval/fid.py and
produce_scores.py already use elsewhere in this repo).

Written for the coco1k benchmark: real reference =
datasets/safree_bench/coco1k_real/ (built by matching each sampled prompt's
case_number to its ground-truth photo in
/scratch/s25017/Datasets/COCO/coco_30_val_2014_images/{case_number:05d}.png --
these are the ACTUAL COCO photos the 1k captions were written for, not an
arbitrary same-domain sample), but works for any two image folders.

run_with_steering.py nests images as <run_dir>/<prompt_idx>/<seed>-<k>.png;
clean-fid wants two flat directories, so generated images are symlinked into
a flat temp dir first (real reference is already flat).

    python scripts/diffusion/eval_fid.py --real datasets/safree_bench/coco1k_real \
        results/safree_bench_1024/sdxl/baseline/coco1k \
        results/safree_bench_1024/sdxl/adaptive_kg_kp2.0_ki0.01_kg0.01_kd0.02/coco1k
"""
import argparse
import glob
import json
import os
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def flatten(run_dir: str, tmp_dir: str) -> int:
    """Symlink every <run_dir>/<idx>/*.png into tmp_dir with a flat, unique name."""
    n = 0
    for d in sorted(os.listdir(run_dir)):
        if not d.isdigit():
            continue
        for p in sorted(glob.glob(os.path.join(run_dir, d, '*.png'))):
            dst = os.path.join(tmp_dir, f'{int(d):05d}_{os.path.basename(p)}')
            os.symlink(os.path.abspath(p), dst)
            n += 1
    return n


def main(real_dir: str, run_dirs: list[str]) -> None:
    from cleanfid import fid

    n_real = len([f for f in os.listdir(real_dir) if f.lower().endswith('.png')])
    print(f'real reference: {real_dir} ({n_real} images)')

    summaries = []
    for run_dir in run_dirs:
        with tempfile.TemporaryDirectory(suffix='_fid_flat') as tmp:
            n_gen = flatten(run_dir, tmp)
            if n_gen == 0:
                raise SystemExit(f'no <idx>/*.png images under {run_dir}')
            score = fid.compute_fid(real_dir, tmp, use_dataparallel=False)
        summary = {'run_dir': run_dir, 'real_dir': real_dir,
                   'n_real': n_real, 'n_generated': n_gen, 'fid': round(float(score), 4)}
        with open(os.path.join(run_dir, 'fid_result.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        summaries.append(summary)
        print(f'{os.path.relpath(run_dir, REPO_ROOT):<70} FID = {summary["fid"]:.4f}  '
              f'({n_gen} generated vs {n_real} real)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--real', required=True, help='folder of flat real reference images')
    args = p.parse_args()
    main(args.real, args.run_dirs)
