"""
CLIP score of a run_with_steering.py --prompts_csv output directory, under the
RECE / SAFREE protocol (RECE/execs/clip_score.py with its default
--model_version base): openai/clip-vit-base-patch32, HF CLIPModel
logits_per_image = 100 * cos(image, caption), averaged over images.

Walks <run_dir>/<prompt_idx>/*.png, looks the caption up by <prompt_idx> in
the prompts CSV the run was generated from, writes <run_dir>/clip_score_result.json
and <run_dir>/clip_score_per_image.csv, and prints one line per run.

    python scripts/diffusion/eval_clip_score.py --prompts_csv datasets/safree_bench/coco1k.csv \
        results/safree_bench/sdxl/baseline/coco1k results/safree_bench/sdxl/adaptive_kg/coco1k
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLIP_ID = 'openai/clip-vit-base-patch32'


def list_images(run_dir: str) -> list[tuple[int, str]]:
    out = []
    for d in os.listdir(run_dir):
        if d.isdigit():
            for p in sorted(glob.glob(os.path.join(run_dir, d, '*.png'))):
                out.append((int(d), p))
    return sorted(out)


@torch.no_grad()
def evaluate(run_dir: str, prompts: list[str], model, processor, device, batch_size: int) -> dict:
    images = list_images(run_dir)
    if not images:
        raise SystemExit(f'no <idx>/*.png images under {run_dir}')
    rows = []
    for start in tqdm(range(0, len(images), batch_size), desc=os.path.basename(run_dir.rstrip('/'))):
        chunk = images[start:start + batch_size]
        pil = [Image.open(p).convert('RGB') for _, p in chunk]
        caps = [prompts[idx] for idx, _ in chunk]
        inputs = processor(text=caps, images=pil, return_tensors='pt', padding=True, truncation=True).to(device)
        out = model(**inputs)
        # logits_per_image[i, i]: image i against its own caption (the
        # off-diagonal pairs are other images' captions).
        scores = out.logits_per_image.diagonal().float().cpu().tolist()
        for (idx, path), cap, sc in zip(chunk, caps, scores):
            rows.append({'prompt_idx': idx, 'image': os.path.relpath(path, run_dir),
                         'prompt': cap, 'clip_score': round(sc, 4)})
    df = pd.DataFrame(rows)
    summary = {
        'run_dir': run_dir,
        'evaluated': len(df),
        'prompts': int(df['prompt_idx'].nunique()),
        'clip_score': round(float(df['clip_score'].mean()), 4),
        'clip_model': CLIP_ID,
        'definition': 'mean over images of HF CLIPModel logits_per_image (100 * cosine), RECE protocol',
    }
    df.to_csv(os.path.join(run_dir, 'clip_score_per_image.csv'), index=False)
    with open(os.path.join(run_dir, 'clip_score_result.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--prompts_csv', required=True, help="CSV the run was generated from ('prompt' column, row order = <prompt_idx>)")
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    df = pd.read_csv(args.prompts_csv)
    prompts = [str(x) for x in df['prompt']]
    model = CLIPModel.from_pretrained(CLIP_ID).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    summaries = [evaluate(d, prompts, model, processor, args.device, args.batch_size) for d in args.run_dirs]
    print(f'\n{"run":<70} {"images":>7} {"CLIP":>7}')
    for s in summaries:
        print(f'{os.path.relpath(s["run_dir"], REPO_ROOT):<70} {s["evaluated"]:>7} {s["clip_score"]:>7.2f}')
