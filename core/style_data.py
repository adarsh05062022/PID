"""
Artist-removal prompt sets, shared by `scripts/style/generate_artist.py` and
`scripts/style/eval_artist.py`.

Each CSV holds 100 rows over 5 artists, 20 prompts each, with columns
`case_number`, `prompt`, `evaluation_seed`, `artist`. The 20 rows naming the
erased artist and the 80 naming the other four are what split every metric into
an erasure half and a collateral-damage half:

    LPIPS_e / Acc_e   over rows whose `artist` is the erased one
    LPIPS_u / Acc_u   over the rest

A method that simply degrades the model moves both halves together; the split is
the only thing that tells erasure apart from damage. `is_erased_prompt` is the
single definition of that membership, so the generator and the scorer cannot
drift apart on it.
"""
import os

import pandas as pd

_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'exp', 'datasets', 'eval', 'artists')

ARTIST_DATASETS = {
    'Van Gogh': os.path.join(_DATASET_DIR, 'big_artist_prompts.csv'),
    'Kelly McKernan': os.path.join(_DATASET_DIR, 'short_niche_art_prompts.csv'),
}

# Slug used in result directory names, so `--artist "Van Gogh"` and the
# directory `..._van_gogh` stay in sync without each caller re-deriving it.
ARTIST_SLUGS = {
    'Van Gogh': 'van_gogh',
    'Kelly McKernan': 'kelly_mckernan',
}


def load_artist_dataset(artist: str, prompts_csv: str | None = None) -> pd.DataFrame:
    path = prompts_csv or ARTIST_DATASETS.get(artist)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f'Prompt set for {artist} not found at {path}. Expected '
            f'big_artist_prompts.csv / short_niche_art_prompts.csv under {_DATASET_DIR}.')
    return pd.read_csv(path)


def is_erased_prompt(row, artist: str) -> bool:
    """Does this row's prompt name the artist being erased?

    Substring match, because the CSVs spell the erased artist as 'Van Gogh'
    while the multi-choice artist lists spell the same painter 'Vincent Van
    Gogh'.
    """
    return artist.lower() in str(row.get('artist', '')).lower()
