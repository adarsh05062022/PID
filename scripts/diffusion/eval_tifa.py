"""
TIFA score (Hu et al., 2023) of run_with_steering.py --prompts_csv output
directories, in two stages so the expensive question generation runs once per
prompt set and is shared by every run scored on it:

  questions   LLaMA-2 question generation + UnifiedQA filtering for every
              prompt of a CSV -> <questions_json>  (resumable; ~1-2 s/prompt)
  score       VQA on <run_dir>/<prompt_idx>/*.png against the cached questions
              -> <run_dir>/tifa_result.json + tifa_per_image.csv

    python scripts/diffusion/eval_tifa.py questions \
        --prompts_csv datasets/safree_bench/coco1k.csv \
        --questions_json datasets/safree_bench/coco1k_tifa_questions.json

    python scripts/diffusion/eval_tifa.py score \
        --questions_json datasets/safree_bench/coco1k_tifa_questions.json \
        --vqa_model mplug-large \
        results/safree_bench/sdxl/baseline/coco1k results/safree_bench/sdxl/adaptive_kg/coco1k

Runs in the `tifa` conda env (see scripts/diffusion/run_safree_bench.sh), not
in munba3. See core/eval/tifa.py for the model choices.
"""
import argparse
import glob
import json
import os
import sys
from statistics import mean

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
from tqdm import tqdm

from core.eval import tifa


def build_questions(args):
    df = pd.read_csv(args.prompts_csv)
    prompts = [str(x) for x in df['prompt']]

    cache = {}
    if os.path.exists(args.questions_json):
        with open(args.questions_json) as f:
            cache = json.load(f)
        print(f'resuming: {len(cache)} of {len(prompts)} prompts already have questions')

    qg = tifa.get_llama2_pipeline(args.qg_model)
    qa_filter = tifa.UnifiedQAModel(args.unifiedqa_model)

    n_raw = n_kept = 0
    for idx, caption in enumerate(tqdm(prompts, desc='questions')):
        key = str(idx)
        if key in cache:
            continue
        raw = tifa.get_llama2_question_and_answers(qg, caption)
        kept = tifa.filter_question_and_answers(qa_filter, raw)
        n_raw += len(raw)
        n_kept += len(kept)
        cache[key] = {'prompt': caption, 'questions': kept}
        if (idx + 1) % 25 == 0:
            _dump(cache, args.questions_json)
    _dump(cache, args.questions_json)
    empty = sum(1 for v in cache.values() if not v['questions'])
    print(f'wrote {args.questions_json}: {len(cache)} prompts, '
          f'{n_kept}/{n_raw} generated questions survived filtering this session, '
          f'{empty} prompts ended up with no question (excluded from the score)')


def _dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def list_images(run_dir):
    out = []
    for d in os.listdir(run_dir):
        if d.isdigit():
            for p in sorted(glob.glob(os.path.join(run_dir, d, '*.png'))):
                out.append((int(d), p))
    return sorted(out)


def score_run(run_dir, questions, vqa_model, vqa_name):
    images = list_images(run_dir)
    if not images:
        raise SystemExit(f'no <idx>/*.png images under {run_dir}')
    rows, details = [], {}
    for idx, path in tqdm(images, desc=os.path.basename(run_dir.rstrip('/'))):
        entry = questions.get(str(idx))
        if entry is None:
            raise SystemExit(f'prompt {idx} has no entry in the questions file; rebuild it for this CSV')
        if not entry['questions']:
            continue   # nothing survived filtering for this caption, as in tifascore
        res = tifa.tifa_score_single(vqa_model, entry['questions'], path)
        rel = os.path.relpath(path, run_dir)
        rows.append({'prompt_idx': idx, 'image': rel, 'prompt': entry['prompt'],
                     'num_questions': len(entry['questions']), 'tifa_score': round(res['tifa_score'], 4)})
        details[rel] = res['question_details']

    df = pd.DataFrame(rows)
    type_scores = {}
    for qd in details.values():
        for q in qd.values():
            type_scores.setdefault(q['element_type'], []).append(q['scores'])
    summary = {
        'run_dir': run_dir,
        'evaluated': len(df),
        'skipped_no_questions': len(images) - len(df),
        'tifa_score': round(float(df['tifa_score'].mean()), 4),
        'tifa_stdev': round(float(df['tifa_score'].std()), 4),
        'accuracy_by_type': {k: round(mean(v), 4) for k, v in sorted(type_scores.items())},
        'vqa_model': vqa_name,
        'definition': 'mean over images of (fraction of filtered questions the VQA model answers correctly)',
    }
    df.to_csv(os.path.join(run_dir, 'tifa_per_image.csv'), index=False)
    with open(os.path.join(run_dir, 'tifa_result.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    with open(os.path.join(run_dir, 'tifa_question_details.json'), 'w') as f:
        json.dump(details, f, indent=1)
    return summary


def score(args):
    with open(args.questions_json) as f:
        questions = json.load(f)
    vqa_model = tifa.VQAModel(args.vqa_model)
    summaries = [score_run(d, questions, vqa_model, args.vqa_model) for d in args.run_dirs]
    print(f'\n{"run":<70} {"images":>7} {"TIFA":>7}')
    for s in summaries:
        print(f'{os.path.relpath(s["run_dir"], REPO_ROOT):<70} {s["evaluated"]:>7} {s["tifa_score"]:>7.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)

    q = sub.add_parser('questions', help='generate + filter questions for a prompt CSV')
    q.add_argument('--prompts_csv', required=True)
    q.add_argument('--questions_json', required=True)
    q.add_argument('--qg_model', default=tifa.QG_MODEL)
    q.add_argument('--unifiedqa_model', default=tifa.UNIFIEDQA_MODEL)
    q.set_defaults(func=build_questions)

    s = sub.add_parser('score', help='VQA-score run directories against cached questions')
    s.add_argument('run_dirs', nargs='+')
    s.add_argument('--questions_json', required=True)
    s.add_argument('--vqa_model', choices=sorted(tifa.VQA_MODELS), default='mplug-large')
    s.set_defaults(func=score)

    args = p.parse_args()
    args.func(args)
