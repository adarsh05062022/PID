"""
Assemble Table 5 of CASteer (arXiv:2503.09630v5, "Quantitative evaluation of
concrete object erasure") from the scores written by produce_scores.py, and
print it next to the numbers reported in the paper.

Reads, for every concept in CONCEPTS:
    <results_dir>/eval_<concept>/clip_score.tsv   (method, clip_score, clip_accuracy, concept)
    <results_dir>/eval_<concept>/fid.tsv          (method, fid)
The paper's CS is the SPM CLIP score (2.5 * cos) x 100, so clip_score is
scaled by 100 here. FID is against the concept's orig/ directory.

Every method directory found under eval_<concept>/ becomes a row, so extra
arms (beta sweeps, per-step vectors, PID controllers) show up automatically;
rows with a missing concept get '-' for that concept.

Also reports the two summary numbers behind Fig. 4: normalized Snoopy CS
(CS / CS of SD-1.4) and the mean normalized CS / mean FID over the five
preservation concepts.

Usage:
    python scripts/diffusion/make_table5.py [--results_dir results/sd14]
"""
import argparse
import os

import pandas as pd

CONCEPTS = ['snoopy', 'mickey', 'spongebob', 'pikachu', 'dog', 'legislator']
ERASED = 'snoopy'
PRESERVED = [c for c in CONCEPTS if c != ERASED]

# Table 5 as printed in the paper (v5). Columns per concept: (CS, FID); FID is
# undefined for the un-steered SD-1.4 row. Baseline rows are, per the caption,
# "taken from SPM (Lyu et al., 2024) or reproduced" by the CASteer authors.
PAPER = {
    'SD-1.4':            {'snoopy': (78.5, None), 'mickey': (74.7, None),  'spongebob': (74.1, None),  'pikachu': (74.7, None),  'dog': (65.2, None), 'legislator': (61.0, None)},
    'ESD':               {'snoopy': (48.3, None), 'mickey': (58.0, 121.0), 'spongebob': (64.0, 104.7), 'pikachu': (68.6, 68.3),  'dog': (63.9, 49.5), 'legislator': (59.9, 50.9)},
    'SPM':               {'snoopy': (60.9, None), 'mickey': (74.4, 22.1),  'spongebob': (74.0, 21.4),  'pikachu': (74.6, 12.4),  'dog': (65.2, 9.0),  'legislator': (61.0, 5.5)},
    'SAFREE':            {'snoopy': (54.7, None), 'mickey': (68.1, 72.8),  'spongebob': (70.2, 76.6),  'pikachu': (71.9, 46.2),  'dog': (65.4, 70.9), 'legislator': (59.9, 55.4)},
    'Receler (reg=0.1)': {'snoopy': (45.7, None), 'mickey': (55.6, 143.5), 'spongebob': (59.6, 156.2), 'pikachu': (63.5, 121.9), 'dog': (64.0, 68.9), 'legislator': (60.6, 42.7)},
    'Receler (reg=1.0)': {'snoopy': (49.1, None), 'mickey': (63.4, 105.9), 'spongebob': (62.5, 128.5), 'pikachu': (71.5, 62.8),  'dog': (64.0, 50.2), 'legislator': (60.7, 40.1)},
    'DoCo':              {'snoopy': (49.1, None), 'mickey': (74.4, 31.1),  'spongebob': (73.8, 21.6),  'pikachu': (74.7, 14.5),  'dog': (65.2, 10.1), 'legislator': (60.9, 5.6)},
    'Ours':              {'snoopy': (45.8, None), 'mickey': (70.4, 93.0),  'spongebob': (72.4, 81.4),  'pikachu': (74.0, 38.3),  'dog': (66.0, 31.8), 'legislator': (61.0, 40.9)},
    'Ours (clip)':       {'snoopy': (48.5, None), 'mickey': (70.4, 89.6),  'spongebob': (72.5, 81.4),  'pikachu': (73.7, 34.4),  'dog': (65.7, 31.8), 'legislator': (60.8, 37.1)},
}

# Directory name -> row label for the arms that correspond to a paper row.
REPRO_LABELS = {
    'orig': 'SD-1.4',
    'casteer-2.0': 'Ours',
    'casteer-2.0-clip': 'Ours (clip)',
    'casteer-2.0-clip-allsteps': 'Ours (clip, per-step vectors)',
}


def load_scores(results_dir):
    """{method_dir: {concept: (CS x100 or None, FID or None)}} from the tsv files."""
    rows = {}
    for concept in CONCEPTS:
        d = os.path.join(results_dir, f'eval_{concept}')
        cs_path, fid_path = os.path.join(d, 'clip_score.tsv'), os.path.join(d, 'fid.tsv')
        if not os.path.exists(cs_path):
            print(f'[make_table5] no clip_score.tsv for {concept} ({d}); skipping the concept')
            continue
        cs = pd.read_csv(cs_path, sep='\t')
        cs = cs[cs['concept'] == concept]
        fid = pd.read_csv(fid_path, sep='\t') if os.path.exists(fid_path) else pd.DataFrame(columns=['method', 'fid'])
        fid = dict(zip(fid['method'], fid['fid']))
        for _, r in cs.iterrows():
            rows.setdefault(r['method'], {})[concept] = (100.0 * float(r['clip_score']), fid.get(r['method']))
    return rows


def fmt(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and pd.isna(x)) else f'{x:.{nd}f}'


def summary(row, sd14_row):
    """Fig. 4 numbers: normalized Snoopy CS, mean normalized CS and mean FID over the preserved concepts."""
    def norm(c):
        if c not in row or c not in sd14_row or row[c][0] is None or sd14_row[c][0] is None:
            return None
        return row[c][0] / sd14_row[c][0]
    n_snoopy = norm(ERASED)
    n_others = [norm(c) for c in PRESERVED]
    n_others = [x for x in n_others if x is not None]
    fids = [row[c][1] for c in PRESERVED if c in row and row[c][1] is not None]
    return (n_snoopy,
            sum(n_others) / len(n_others) if n_others else None,
            sum(fids) / len(fids) if fids else None)


def build_table(results_dir):
    repro = load_scores(results_dir)
    header = ['Method', 'Snoopy CS↓']
    for c in PRESERVED:
        header += [f'{c.capitalize()} CS↑', f'{c.capitalize()} FID↓']
    header += ['norm. Snoopy CS↓', 'mean norm. CS (others)↑', 'mean FID (others)↓']

    lines = []
    def add(label, row, sd14_row):
        cells = [label, fmt(row.get(ERASED, (None, None))[0])]
        for c in PRESERVED:
            cs, f = row.get(c, (None, None))
            cells += [fmt(cs), fmt(f)]
        cells += [fmt(x, 3) if i < 2 else fmt(x) for i, x in enumerate(summary(row, sd14_row))]
        lines.append(cells)

    for label, row in PAPER.items():
        add(f'{label} (paper)', row, PAPER['SD-1.4'])
    sd14_repro = repro.get('orig', PAPER['SD-1.4'])
    order = [m for m in REPRO_LABELS if m in repro] + sorted(m for m in repro if m not in REPRO_LABELS)
    for m in order:
        label = REPRO_LABELS.get(m, m)
        add(f'{label} (repro: {m})', repro[m], sd14_repro)
    return header, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results/sd14')
    ap.add_argument('--out_prefix', default=None, help='default: <results_dir>/table5_object_erasure')
    args = ap.parse_args()
    out_prefix = args.out_prefix or os.path.join(args.results_dir, 'table5_object_erasure')

    header, lines = build_table(args.results_dir)
    df = pd.DataFrame(lines, columns=header)
    df.to_csv(out_prefix + '.tsv', sep='\t', index=False)

    widths = [max(len(str(x)) for x in [h] + [l[i] for l in lines]) for i, h in enumerate(header)]
    md = ['| ' + ' | '.join(str(h).ljust(w) for h, w in zip(header, widths)) + ' |',
          '|' + '|'.join('-' * (w + 2) for w in widths) + '|']
    for l in lines:
        md.append('| ' + ' | '.join(str(x).ljust(w) for x, w in zip(l, widths)) + ' |')
    md_text = '\n'.join(md)
    with open(out_prefix + '.md', 'w') as f:
        f.write('Table 5 (CASteer, arXiv:2503.09630v5): concrete object erasure, erasing "Snoopy" from SD-1.4.\n')
        f.write('CS = SPM CLIP score x100 (ViT-B/32) against the concept name; FID = clean-fid vs the concept\'s SD-1.4 images '
                '(800 images per concept: 80 CLIP templates x 10). "(paper)" rows are copied from the paper, '
                '"(repro: <dir>)" rows are computed from results/sd14/eval_<concept>/<dir>.\n\n')
        f.write(md_text + '\n')
    print(md_text)
    print(f'\nwrote {out_prefix}.md and {out_prefix}.tsv')


if __name__ == '__main__':
    main()
