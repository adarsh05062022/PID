"""
Render the gain sweep as one results table (PNG + markdown).

Two column groups, one per erased artist, each carrying the same four metrics
with their good direction marked.

Rows are grouped by kp, NOT by method:

    kp     Method                LPIPS_e^  LPIPS_u_  Acc_e_  Acc_u^ | ...
    0.5    CASteer               ...
           Adaptive Kg kd=0.1    ...
           Adaptive Kg kd=0.5    ...
    ------------------------------------------------------------------
    1.0    CASteer               ...

because kp is the axis that dominates every metric -- raise it and all four
move together. Listing all CASteer arms and then all Adaptive Kg arms puts the
two laws four rows apart at a given kp and invites reading down a column, where
the comparison is confounded by kp. Interleaving them holds kp fixed inside
each block, which is the only place the two laws are actually comparable.

For the same reason there is no best-in-column bolding. Across the whole
column the preservation metrics (LPIPS_u low, Acc_u high) are won trivially by
whichever arm steers least -- an arm that does nothing scores perfectly on both
and erases nothing. Bold marks the best value WITHIN a kp block, where the
proportional term is identical and the comparison means something.

Baseline LPIPS cells are blank rather than 0.0000: the baseline IS the LPIPS
reference, so the number is zero by construction and would read as a winning
score in a column where low is good.

    PYTHONPATH=. python scripts/style/make_results_table.py \\
        --comparisons results/sd14/style/casteer_vectors/comparison_sweep_vangogh.json \\
                      results/sd14/style/casteer_vectors/comparison_sweep_kelly.json \\
        --output results/sd14/style/casteer_vectors/results_table.png
"""
import argparse
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# (key in the per-arm dict, section it lives under, header, higher_is_better)
METRICS = [
    ('LPIPS_e', 'lpips', 'LPIPSₑ', True),
    ('LPIPS_u', 'lpips', 'LPIPSᵤ', False),
    ('Acc_e', 'acc', 'Accₑ', False),
    ('Acc_u', 'acc', 'Accᵤ', True),
]

BG = (255, 255, 255)
INK = (17, 17, 17)
MUTED = (122, 122, 122)
RULE = (17, 17, 17)
HAIRLINE = (206, 206, 210)
BAND = (245, 246, 249)
KPINK = (60, 60, 60)

PAD_X = 26
PAD_Y = 22
ROW_H = 33
KP_W = 62
METHOD_W = 196
COL_W = 114
GROUP_GAP = 20


def _font(size: int, bold: bool = False):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
        '/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf' % ('-Bold' if bold else ''),
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse_tag(tag: str):
    """-> (family, kp, kd) with family in {'baseline','casteer','adaptive'}."""
    if tag.lower().startswith('baseline'):
        return ('baseline', None, None)
    kp = re.search(r'kp=([0-9.]+)', tag)
    kd = re.search(r'kd=([0-9.]+)', tag)
    family = 'casteer' if tag.lower().startswith('casteer') else 'adaptive'
    return (family, float(kp.group(1)) if kp else None,
            float(kd.group(1)) if kd else None)


def load_rows(paths):
    """-> ({tag: {artist: {metric: value}}}, ordered artists)"""
    table, artists = {}, []
    for path in paths:
        with open(path) as f:
            master = json.load(f)
        for artist, methods in master.items():
            if artist not in artists:
                artists.append(artist)
            for tag, sections in methods.items():
                cells = table.setdefault(tag, {}).setdefault(artist, {})
                for key, section, _, _ in METRICS:
                    val = sections.get(section, {}).get(key)
                    if val is not None:
                        cells[key] = float(val)
    return table, artists


def build_blocks(table):
    """Group tags into ordered kp blocks.

    -> [(kp_label, [(method_label, tag), ...]), ...] with the baseline first.
    """
    blocks, baseline = [], [t for t in table if parse_tag(t)[0] == 'baseline']
    if baseline:
        blocks.append((None, [('Baseline (no steering)', baseline[0])]))

    kps = sorted({parse_tag(t)[1] for t in table if parse_tag(t)[1] is not None})
    for kp in kps:
        rows = []
        for tag in sorted(table, key=lambda t: (parse_tag(t)[2] or 0.0)):
            fam, tkp, kd = parse_tag(tag)
            if tkp != kp:
                continue
            if fam == 'casteer':
                rows.insert(0, ('CASteer', tag))          # control first in the block
            elif fam == 'adaptive':
                rows.append((f'Adaptive Kg  kd={kd:g}', tag))
        if rows:
            blocks.append((f'{kp:g}', rows))
    return blocks


def block_winners(rows, table, artists):
    """Best value per (artist, metric) WITHIN one kp block.

    Only meaningful with 2+ arms in the block, and never for the baseline
    block, which has no competitor.
    """
    if len(rows) < 2:
        return {}
    best = {}
    for artist in artists:
        for key, _, _, higher in METRICS:
            vals = [table[tag][artist][key] for _, tag in rows
                    if artist in table.get(tag, {}) and key in table[tag][artist]]
            if len(vals) >= 2:
                best[(artist, key)] = max(vals) if higher else min(vals)
    return best


def render(blocks, table, artists, out_path, title=None):
    n_cols = len(METRICS)
    n_rows = sum(len(rows) for _, rows in blocks)
    width = (PAD_X * 2 + KP_W + METHOD_W + len(artists) * (n_cols * COL_W)
             + (len(artists) - 1) * GROUP_GAP)
    head_h = 34 + 28
    title_h = 40 if title else 0
    height = PAD_Y * 2 + title_h + head_h + n_rows * ROW_H + (len(blocks) - 1) * 2 + 20

    img = Image.new('RGB', (width, height), BG)
    d = ImageDraw.Draw(img)

    f_grp = _font(15, bold=True)
    f_col = _font(13, bold=True)
    f_row = _font(13)
    f_row_b = _font(13, bold=True)
    f_num = _font(13)
    f_num_b = _font(13, bold=True)
    f_kp = _font(14, bold=True)
    f_title = _font(17, bold=True)

    x_data = PAD_X + KP_W + METHOD_W

    def group_x(gi):
        return x_data + gi * (n_cols * COL_W + GROUP_GAP)

    y = PAD_Y
    if title:
        d.text((PAD_X, y), title, font=f_title, fill=INK)
        y += title_h

    # ---- artist group headers ----
    for gi, artist in enumerate(artists):
        gx = group_x(gi)
        gw = n_cols * COL_W
        label = f'Remove "{artist}"'
        tw = d.textlength(label, font=f_grp)
        d.text((gx + (gw - tw) / 2, y + 4), label, font=f_grp, fill=INK)
        d.line([(gx + 6, y + 26), (gx + gw - 6, y + 26)], fill=RULE, width=1)
    d.text((PAD_X, y + 32), 'kp', font=f_col, fill=INK)
    d.text((PAD_X + KP_W, y + 32), 'Method', font=f_col, fill=INK)

    # ---- metric headers ----
    for gi in range(len(artists)):
        gx = group_x(gi)
        for ci, (_, _, head, higher) in enumerate(METRICS):
            label = f'{head} {"↑" if higher else "↓"}'
            tw = d.textlength(label, font=f_col)
            d.text((gx + ci * COL_W + (COL_W - tw) - 10, y + 32), label, font=f_col, fill=INK)
    y += head_h
    d.line([(PAD_X, y), (width - PAD_X, y)], fill=RULE, width=2)

    # ---- rows, grouped by kp ----
    band = False
    for bi, (kp_label, rows) in enumerate(blocks):
        winners = block_winners(rows, table, artists) if kp_label else {}
        is_baseline = kp_label is None
        block_h = len(rows) * ROW_H

        if band and not is_baseline:
            d.rectangle([PAD_X, y, width - PAD_X, y + block_h], fill=BAND)
        if not is_baseline:
            band = not band

        if kp_label:
            d.text((PAD_X, y + block_h / 2 - 9), kp_label, font=f_kp, fill=KPINK)

        for ri, (method_label, tag) in enumerate(rows):
            ry = y + ri * ROW_H
            fill = MUTED if is_baseline else INK
            d.text((PAD_X + KP_W, ry + 9), method_label,
                   font=f_row_b if is_baseline else f_row, fill=fill)

            for gi, artist in enumerate(artists):
                gx = group_x(gi)
                cells = table.get(tag, {}).get(artist, {})
                for ci, (key, _, _, _) in enumerate(METRICS):
                    if is_baseline and key.startswith('LPIPS'):
                        txt, is_best = '—', False
                    elif key in cells:
                        txt = f'{cells[key]:.4f}'
                        is_best = (not is_baseline
                                   and (artist, key) in winners
                                   and abs(cells[key] - winners[(artist, key)]) < 1e-9)
                    else:
                        txt, is_best = '·', False
                    font = f_num_b if is_best else f_num
                    tw = d.textlength(txt, font=font)
                    d.text((gx + ci * COL_W + (COL_W - tw) - 10, ry + 9), txt,
                           font=font, fill=MUTED if is_baseline else INK)

        y += block_h
        if bi < len(blocks) - 1:
            d.line([(PAD_X, y), (width - PAD_X, y)], fill=HAIRLINE, width=1)
            y += 2

    d.line([(PAD_X, y + 2), (width - PAD_X, y + 2)], fill=RULE, width=2)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return img.size, n_rows


def markdown(blocks, table, artists) -> str:
    """One table per artist -- a single 10-column table is unreadable in a terminal."""
    out = []
    for artist in artists:
        out.append(f'\n**Remove "{artist}"**\n')
        out.append('| kp | Method | LPIPS_e ↑ | LPIPS_u ↓ | Acc_e ↓ | Acc_u ↑ |')
        out.append('|---|---|---|---|---|---|')
        for kp_label, rows in blocks:
            for ri, (method_label, tag) in enumerate(rows):
                cells = table.get(tag, {}).get(artist, {})
                vals = []
                for key, _, _, _ in METRICS:
                    if kp_label is None and key.startswith('LPIPS'):
                        vals.append('—')
                    elif key in cells:
                        vals.append(f'{cells[key]:.4f}')
                    else:
                        vals.append('·')
                kp_cell = (kp_label if (kp_label and ri == 0) else '')
                out.append(f'| {kp_cell} | {method_label} | ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


def main(args):
    table, artists = load_rows(args.comparisons)
    if args.artist_order:
        want = [a.strip() for a in args.artist_order.split(',')]
        artists = [a for a in want if a in artists] + [a for a in artists if a not in want]
    blocks = build_blocks(table)

    size, n_rows = render(blocks, table, artists, args.output, args.title)
    print(f'{args.output}  ({size[0]}x{size[1]}, {n_rows} rows)')
    print(markdown(blocks, table, artists))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--comparisons', nargs='+', required=True)
    p.add_argument('--output', default='results/sd14/style/casteer_vectors/results_table.png')
    p.add_argument('--title', default=None)
    p.add_argument('--artist_order', default='Van Gogh,Kelly McKernan')
    main(p.parse_args())
