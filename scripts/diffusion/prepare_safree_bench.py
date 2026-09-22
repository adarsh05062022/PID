"""
Build the prompt sets for the SAFREE Table 5 protocol (SDXL / SD-v3 rows) as
uniform CSVs under datasets/safree_bench/, one per benchmark, each with the
columns run_with_steering.py needs:

    prompt        text handed to the model
    seed          per-prompt generator seed (SAFREE: the CSV's evaluation_seed
                  where the benchmark has one, else 42)
    case_number   the source file's id, for tracing an image back to its row
    source        which raw file the row came from

The prompt column and the seed rule per benchmark mirror SAFREE's
generate_safree.py / sdv3/sdv3.py:

    p4d          p4dn_16_prompt.csv           'prompt'      evaluation_seed
    ring_a_bell  nudity-ring-a-bell.csv       'prompt'      42   (SAFREE: 'sensitive prompt')
    mma          mma-diffusion-nsfw-adv-...   'adv_prompt'  42
    unlearndiff  nudity.csv                   'prompt'      evaluation_seed
    coco1k       coco_30k_10k.csv (1k sample) 'prompt'      evaluation_seed

The P4D-N prompts are gated on HuggingFace (joycenerd/p4d, file
p4dn_16_prompt.csv); drop that file into datasets/ once access is granted and
re-run this script -- until then the p4d benchmark is skipped with a warning.

COCO: SAFREE evaluates CLIP/TIFA on 1k random COCO-30k captions. Their sample
is not published, so this draws 1k rows from their coco_30k_10k.csv with a
fixed RNG seed (--coco_sample_seed), which makes the subset reproducible here
but not identical to theirs.

    python scripts/diffusion/prepare_safree_bench.py
"""
import argparse
import os

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASETS = os.path.join(REPO_ROOT, 'datasets')
OUT_DIR = os.path.join(DATASETS, 'safree_bench')
DEFAULT_SEED = 42   # SAFREE's fallback when a benchmark carries no seeds

P4D_NOTE = ('P4D-N prompts not found at {path}. They are gated: request access at '
            'https://huggingface.co/datasets/joycenerd/p4d, download p4dn_16_prompt.csv '
            'into datasets/ and re-run. Skipping p4d.')


def _finish(df: pd.DataFrame, prompt_col: str, source: str, seed_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame({
        'prompt': df[prompt_col].astype(str).str.strip(),
        'seed': df[seed_col].astype(int) if seed_col else DEFAULT_SEED,
        'case_number': df['case_number'] if 'case_number' in df.columns else df.index,
        'source': source,
    })
    keep = out['prompt'].str.len() > 0
    if not keep.all():
        print(f'  dropped {(~keep).sum()} empty prompts from {source}')
    return out[keep].reset_index(drop=True)


def build_p4d() -> pd.DataFrame | None:
    path = os.path.join(DATASETS, 'p4dn_16_prompt.csv')
    if not os.path.exists(path):
        print('WARNING: ' + P4D_NOTE.format(path=path))
        return None
    df = pd.read_csv(path)
    return _finish(df, 'prompt', 'p4dn_16_prompt.csv', 'evaluation_seed')


def build_ring_a_bell() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATASETS, 'nudity-ring-a-bell.csv'))
    col = 'sensitive prompt' if 'sensitive prompt' in df.columns else 'prompt'
    return _finish(df, col, 'nudity-ring-a-bell.csv', None)


def build_mma() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATASETS, 'mma-diffusion-nsfw-adv-prompts.csv'))
    return _finish(df, 'adv_prompt', 'mma-diffusion-nsfw-adv-prompts.csv', None)


def build_unlearndiff() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATASETS, 'nudity.csv'))
    return _finish(df, 'prompt', 'nudity.csv', 'evaluation_seed')


def build_coco1k(n: int, sample_seed: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATASETS, 'coco_30k_10k.csv'))
    df = df[df['prompt'].notna()]
    df = df.sample(n=n, random_state=sample_seed).sort_values('case_number')
    out = _finish(df, 'prompt', 'coco_30k_10k.csv', 'evaluation_seed')
    out['coco_id'] = df['coco_id'].values
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--coco_n', type=int, default=1000)
    p.add_argument('--coco_sample_seed', type=int, default=0)
    p.add_argument('--out_dir', default=OUT_DIR)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    benches = {
        'p4d': build_p4d(),
        'ring_a_bell': build_ring_a_bell(),
        'mma': build_mma(),
        'unlearndiff': build_unlearndiff(),
        'coco1k': build_coco1k(args.coco_n, args.coco_sample_seed),
    }
    for name, df in benches.items():
        if df is None:
            continue
        path = os.path.join(args.out_dir, f'{name}.csv')
        df.to_csv(path, index=False)
        seeds = 'per-prompt evaluation_seed' if df['seed'].nunique() > 1 else f'seed {df["seed"].iloc[0]} for all'
        print(f'{name:<12} {len(df):>5} prompts  {seeds:<30} -> {os.path.relpath(path, REPO_ROOT)}')
