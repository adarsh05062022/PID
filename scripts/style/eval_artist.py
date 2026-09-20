"""
Score artist-style erasure: LPIPS_e / LPIPS_u and Acc_e / Acc_u.

The four numbers come in two pairs, and only the pairs mean anything -- each
metric is computed twice over the same 100 images, once over the 20 prompts
naming the erased artist and once over the 80 naming the other four:

    LPIPS_e   perceptual distance from the unsteered baseline on erased-artist
              prompts. HIGHER is better: the image moved away from the style.
    LPIPS_u   the same distance on the other artists' prompts. LOWER is better:
              those images should not have moved at all.
    Acc_e     fraction of erased-artist images a VLM still calls that style.
              LOWER is better.
    Acc_u     fraction of other-artist images the VLM still calls their own
              style. HIGHER is better -- the model has not lost those styles.

Either half alone is trivial to win (steer hard enough and everything becomes
noise, or steer not at all). `--append_to` collects arms into one JSON keyed by
artist and method so the pairs can be read side by side.

    python scripts/style/eval_artist.py lpips --artist "Van Gogh" \
        --baseline_dir results/sd14/style/baseline_vangogh/all \
        --method_dir   results/sd14/style/casteer_vectors/adaptive_kg_vangogh/all \
        --tag AdaptiveKg --append_to results/sd14/style/casteer_vectors/comparison_vangogh.json

    python scripts/style/eval_artist.py acc-qwen --artist "Van Gogh" \
        --method_dir results/sd14/style/casteer_vectors/adaptive_kg_vangogh/all \
        --tag AdaptiveKg --append_to results/sd14/style/casteer_vectors/comparison_vangogh.json

    python scripts/style/eval_artist.py report \
        --comparison results/sd14/style/casteer_vectors/comparison_vangogh.json

The VLM judge is Qwen2.5-VL-7B-Instruct, asked one yes/no question per image
("Is this picture in {style} style?"), repeated `--n_runs` times. It is loaded
with `local_files_only=True` from `--vlm_id` (default `$VLM_PATH`), so point
that at a local snapshot.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.style_data import ARTIST_DATASETS, is_erased_prompt, load_artist_dataset

_VLM_DEFAULT = os.environ.get('VLM_PATH', '/storage/s25017/models/Qwen2.5-VL-7B-Instruct')


# ── LPIPS ────────────────────────────────────────────────────────────────────
def compute_lpips(baseline_dir: str, method_dir: str, artist: str,
                  device: str = 'cuda', loss_fn=None) -> dict:
    """LPIPS between each method image and the baseline image from the same
    prompt and seed, split into the erased and unerased halves.

    `loss_fn` lets a caller scoring many arms load AlexNet once and pass it in;
    left None it loads its own, so the single-arm path is unchanged.
    """
    if loss_fn is None:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
    df = load_artist_dataset(artist)

    def _load(path):
        arr = np.asarray(Image.open(path).convert('RGB').resize((512, 512))).astype(np.float32)
        arr = arr / 127.5 - 1.0
        return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    scores_e, scores_u = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='LPIPS'):
        case = row['case_number']
        base_path = os.path.join(baseline_dir, f'{case}.png')
        method_path = os.path.join(method_dir, f'{case}.png')
        if not (os.path.exists(base_path) and os.path.exists(method_path)):
            continue

        with torch.no_grad():
            score = loss_fn(_load(base_path), _load(method_path)).item()

        (scores_e if is_erased_prompt(row, artist) else scores_u).append(score)

    results = {
        'LPIPS_e': round(float(np.mean(scores_e)), 4) if scores_e else 0.0,
        'LPIPS_u': round(float(np.mean(scores_u)), 4) if scores_u else 0.0,
        'n_erased': len(scores_e),
        'n_unerased': len(scores_u),
    }
    print(f"\nLPIPS_e: {results['LPIPS_e']}  ({results['n_erased']} images)  higher = more erased")
    print(f"LPIPS_u: {results['LPIPS_u']}  ({results['n_unerased']} images)  lower = less collateral damage")
    return results


# ── VLM style accuracy ───────────────────────────────────────────────────────
def _qwen_classify(model, processor, img_path: str, style: str, device: str):
    """One yes/no judgement. Returns True/False, or None if generation failed."""
    from qwen_vl_utils import process_vision_info

    question = f'Is this picture in {style} style? Just tell me Yes or No.'
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': Image.open(img_path).convert('RGB')},
        {'type': 'text', 'text': question},
    ]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=5)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0].strip().lower()
        return 'yes' in answer
    except Exception as exc:
        print(f'[WARN] VLM error for {img_path}: {exc}')
        return None


def load_vlm(vlm_id: str = _VLM_DEFAULT, device: str = 'cuda'):
    """Load the judge once. Hoisted out of compute_acc_qwen so a sweep scoring
    many arms pays the 7B load once instead of once per arm."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f'Loading {vlm_id} ...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_id, torch_dtype=torch.float16, local_files_only=True).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(vlm_id, local_files_only=True)
    return model, processor


def compute_acc_qwen(method_dir: str, artist: str, device: str = 'cuda',
                     n_runs: int = 3, vlm_id: str = _VLM_DEFAULT,
                     judge=None) -> tuple:
    """Acc_e / Acc_u from a VLM asked, per image, whether the style is present.

    Erased-artist images are asked about the erased artist; every other image is
    asked about the artist ITS OWN prompt named, which is what makes Acc_u a
    style-preservation measure rather than a second erasure measure.

    `judge` is an already-loaded (model, processor) pair; left None it loads its
    own, so the single-arm path is unchanged.
    """
    model, processor = judge if judge is not None else load_vlm(vlm_id, device)

    df = load_artist_dataset(artist)
    acc_e_runs, acc_u_runs = [], []
    results_log = []

    for run in range(n_runs):
        print(f'\nRun {run + 1}/{n_runs}')
        yes_e = total_e = yes_u = total_u = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'VLM run {run + 1}'):
            case = row['case_number']
            img_path = os.path.join(method_dir, f'{case}.png')
            if not os.path.exists(img_path):
                continue

            erased = is_erased_prompt(row, artist)
            style = artist if erased else str(row.get('artist', ''))

            answer = _qwen_classify(model, processor, img_path, style, device)
            if answer is None:
                continue

            if run == 0:
                results_log.append({
                    'case_number': int(case), 'row_artist': str(row.get('artist', '')),
                    'style_asked': style, 'vlm_yes': bool(answer),
                    'is_erased_prompt': bool(erased),
                })

            if erased:
                total_e += 1
                yes_e += int(answer)
            else:
                total_u += 1
                yes_u += int(answer)

        acc_e_runs.append(yes_e / max(total_e, 1))
        acc_u_runs.append(yes_u / max(total_u, 1))

    results = {
        'Acc_e': round(float(np.mean(acc_e_runs)), 4),
        'Acc_u': round(float(np.mean(acc_u_runs)), 4),
        'model': vlm_id,
        'Acc_e_runs': [round(x, 4) for x in acc_e_runs],
        'Acc_u_runs': [round(x, 4) for x in acc_u_runs],
    }
    print(f"\nAcc_e: {results['Acc_e']}  (avg over {n_runs} runs)  lower = more erased")
    print(f"Acc_u: {results['Acc_u']}  (avg over {n_runs} runs)  higher = style preserved")
    return results, results_log


# ── Result bookkeeping ───────────────────────────────────────────────────────
def append_to_master(path: str, artist: str, tag: str, metric: str, payload: dict) -> None:
    master = {}
    if os.path.exists(path):
        with open(path) as f:
            master = json.load(f)
    master.setdefault(artist, {}).setdefault(tag, {})[metric] = payload
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(master, f, indent=4)
    print(f'Appended [{artist}][{tag}][{metric}] -> {path}')


def print_report(comparison_path: str) -> None:
    """Render a comparison JSON as the four-column table the metrics are read as."""
    with open(comparison_path) as f:
        master = json.load(f)

    for artist, methods in master.items():
        print(f'\n{artist}')
        print(f"{'method':<22}{'LPIPS_e':>10}{'LPIPS_u':>10}{'Acc_e':>10}{'Acc_u':>10}")
        print(f"{'':<22}{'higher':>10}{'lower':>10}{'lower':>10}{'higher':>10}")
        print('-' * 62)
        for tag, metrics in methods.items():
            lp = metrics.get('lpips', {})
            ac = metrics.get('acc', {})
            def _fmt(d, k):
                return f'{d[k]:.4f}' if k in d else '--'
            print(f'{tag:<22}{_fmt(lp, "LPIPS_e"):>10}{_fmt(lp, "LPIPS_u"):>10}'
                  f'{_fmt(ac, "Acc_e"):>10}{_fmt(ac, "Acc_u"):>10}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)

    def _add_common(p):
        p.add_argument('--artist', required=True, choices=list(ARTIST_DATASETS))
        p.add_argument('--method_dir', default=None,
                       help='Directory of {case_number}.png to score (the arm\'s all/ dir)')
        p.add_argument('--method_dirs', default=None,
                       help='Comma-separated arm dirs to score in one process, loading the '
                            'model once. Pair with --tags. Mutually exclusive with --method_dir')
        p.add_argument('--tags', default=None,
                       help='Comma-separated labels, parallel to --method_dirs')
        p.add_argument('--device', default='cuda')
        p.add_argument('--output', default=None,
                       help='Per-run JSON (default: inside --method_dir)')
        p.add_argument('--tag', default=None,
                       help='Method name used as the key in --append_to (e.g. AdaptiveKg)')
        p.add_argument('--append_to', default=None,
                       help='Master comparison JSON to append to (created if missing)')

    lp = sub.add_parser('lpips', help='LPIPS_e / LPIPS_u against the unsteered baseline')
    lp.add_argument('--baseline_dir', required=True,
                    help='Unsteered images from generate_artist.py --no_steer (its all/ dir)')
    _add_common(lp)

    aq = sub.add_parser('acc-qwen', help='Acc_e / Acc_u with a Qwen2.5-VL judge')
    aq.add_argument('--n_runs', type=int, default=3)
    aq.add_argument('--vlm_id', default=_VLM_DEFAULT,
                    help='Local path or HF id for Qwen2.5-VL')
    _add_common(aq)

    rp = sub.add_parser('report', help='Print a comparison JSON as a table')
    rp.add_argument('--comparison', required=True)

    args = parser.parse_args()

    if args.cmd == 'report':
        print_report(args.comparison)
        return

    # One arm or many: --method_dirs/--tags is the batched form, and collapses
    # to the single-arm case so the scoring path below is written once.
    if args.method_dirs:
        dirs = [d for d in args.method_dirs.split(',') if d.strip()]
        tags = [t for t in (args.tags or '').split(',') if t.strip()]
        if len(tags) != len(dirs):
            raise SystemExit(f'--tags has {len(tags)} entries but --method_dirs has {len(dirs)}')
        if args.output:
            raise SystemExit('--output writes a single file; drop it when using --method_dirs')
    elif args.method_dir:
        dirs = [args.method_dir]
        tags = [args.tag or os.path.basename(os.path.dirname(args.method_dir.rstrip('/')))]
    else:
        raise SystemExit('pass --method_dir or --method_dirs')

    # Loaded once for the whole batch; on this NFS mount the model load costs
    # more than scoring an arm does.
    loss_fn = judge = None
    if args.cmd == 'lpips' and len(dirs) > 1:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(args.device)
    elif args.cmd == 'acc-qwen' and len(dirs) > 1:
        judge = load_vlm(args.vlm_id, args.device)

    for method_dir, tag in zip(dirs, tags):
        print(f'\n{"=" * 70}\n{tag}  ({method_dir})\n{"=" * 70}')

        if args.cmd == 'lpips':
            results = compute_lpips(args.baseline_dir, method_dir, args.artist,
                                    args.device, loss_fn=loss_fn)
            out = args.output or os.path.join(method_dir, 'lpips_scores.json')
            with open(out, 'w') as f:
                json.dump(results, f, indent=4)
            print(f'Saved -> {out}')
            if args.append_to:
                append_to_master(args.append_to, args.artist, tag, 'lpips',
                                 {**results, 'method_dir': method_dir,
                                  'baseline_dir': args.baseline_dir})

        elif args.cmd == 'acc-qwen':
            results, log = compute_acc_qwen(method_dir, args.artist,
                                            device=args.device, n_runs=args.n_runs,
                                            vlm_id=args.vlm_id, judge=judge)
            out = args.output or os.path.join(method_dir, 'acc_qwen_scores.json')
            with open(out, 'w') as f:
                json.dump(results, f, indent=4)
            with open(out.replace('.json', '_log.json'), 'w') as f:
                json.dump(log, f, indent=4)
            print(f'Saved -> {out}')
            if args.append_to:
                append_to_master(args.append_to, args.artist, tag, 'acc',
                                 {**results, 'method_dir': method_dir})


if __name__ == '__main__':
    main()
