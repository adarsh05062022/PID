"""
Side-by-side grid over the artist-erasure runs: one row per prompt, one column
per arm.

Reads the layout `scripts/style/generate_artist.py` writes
(`<save_dir>/all/<case_number>.png`) and stacks the selected cases into one
figure. Rows are keyed by case number, so every column shows the *same* prompt
at the *same* seed -- which is the only thing that makes the comparison mean
anything. A case missing from one run leaves that cell blank rather than
shifting the row.

Rows are split into two bands, because the two halves answer different
questions: the ERASED band should change (the style is supposed to go), and the
PRESERVED band should NOT (those artists were never targeted). A steering
vector that moves both bands equally is damaging the model, not erasing a
concept.

    python scripts/style/make_vector_grid.py \\
        --artist "Van Gogh" \\
        --cases 20,21,22,23,24,0,40,60,80,81 \\
        --runs "SD-1.4|no steering:results/sd14/style/baseline_vangogh" \\
               "CASteer|orig sv:results/sd14/style/casteer_vectors/casteer_vangogh" \\
               "CASteer|TECA sv:results/sd14/style/teca_vectors/casteer_teca_vangogh" \\
        --output results/sd14/style/vector_comparison/grid_vangogh.png

Each --runs entry is "LABEL:PATH"; a "|" in the label splits it over two header
lines. Order is preserved left to right. PATH may be the run directory or its
`all/` subdirectory.
"""
import argparse
import os
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.style_data import is_erased_prompt, load_artist_dataset

PAD = 8
HEADER_H = 58
BAND_H = 30
PROMPT_W = 300

BG = (238, 239, 243)
HEADER_BG = (24, 26, 33)
HEADER_FG = (244, 245, 249)
HEADER_ACCENT = (217, 152, 46)
PROMPT_BG = (248, 249, 252)
PROMPT_FG = (22, 24, 31)
PROMPT_MUTED = (110, 118, 134)
MISSING_BG = (210, 212, 218)
BAND_ERASE_BG = (46, 124, 114)
BAND_KEEP_BG = (90, 104, 138)
BAND_FG = (255, 255, 255)


def _font(size: int, bold: bool = False):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
        '/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf' % ('-Bold' if bold else ''),
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _resolve(path: str) -> str:
    """Accept either the run dir or its all/ subdirectory."""
    inner = os.path.join(path, 'all')
    return inner if os.path.isdir(inner) else path


def _cell(path: str, size: int):
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    return img.resize((size, size), Image.LANCZOS)


def main(args):
    df = load_artist_dataset(args.artist, args.prompts_csv)
    by_case = {int(r['case_number']): r for _, r in df.iterrows()}

    cases = [int(c) for c in args.cases.split(',') if c.strip()]
    runs = []
    for entry in args.runs:
        label, path = entry.split(':', 1)
        runs.append((label.split('|'), _resolve(path)))

    size = args.cell
    # Rows are grouped: every erased-artist case first, then the rest, each
    # group introduced by a band that says what the reader should expect.
    erased = [c for c in cases if is_erased_prompt(by_case[c], args.artist)]
    kept = [c for c in cases if not is_erased_prompt(by_case[c], args.artist)]
    groups = [('ERASED — these should lose the style', BAND_ERASE_BG, erased),
              ('PRESERVED — these were never targeted, they should not move', BAND_KEEP_BG, kept)]

    n_rows = len(cases)
    n_bands = sum(1 for _, _, g in groups if g)
    width = PROMPT_W + len(runs) * (size + PAD) + PAD
    height = HEADER_H + n_bands * BAND_H + n_rows * (size + PAD) + PAD

    canvas = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    f_head = _font(15, bold=True)
    f_sub = _font(13)
    f_prompt = _font(13)
    f_meta = _font(12, bold=True)
    f_band = _font(13, bold=True)

    # ---- column headers ----
    draw.rectangle([0, 0, width, HEADER_H], fill=HEADER_BG)
    draw.text((PAD, HEADER_H // 2 - 8), f'{args.artist}', font=f_head, fill=HEADER_ACCENT)
    for ci, (label_lines, _) in enumerate(runs):
        x = PROMPT_W + ci * (size + PAD)
        top = label_lines[0]
        sub = label_lines[1] if len(label_lines) > 1 else ''
        w = draw.textlength(top, font=f_head)
        draw.text((x + (size - w) / 2, 10), top, font=f_head, fill=HEADER_FG)
        if sub:
            w2 = draw.textlength(sub, font=f_sub)
            # the sv family is the variable under test -- give it the accent
            colour = HEADER_ACCENT if 'TECA' in sub else (156, 164, 180)
            draw.text((x + (size - w2) / 2, 32), sub, font=f_sub, fill=colour)

    # ---- rows ----
    y = HEADER_H
    for band_label, band_bg, group in groups:
        if not group:
            continue
        draw.rectangle([0, y, width, y + BAND_H], fill=band_bg)
        draw.text((PAD, y + BAND_H // 2 - 7), band_label, font=f_band, fill=BAND_FG)
        y += BAND_H

        for case in group:
            row = by_case[case]
            draw.rectangle([0, y, PROMPT_W - PAD, y + size], fill=PROMPT_BG)
            draw.text((PAD, y + PAD), f"#{case}  {row['artist']}", font=f_meta, fill=PROMPT_MUTED)
            lines = textwrap.wrap(str(row['prompt']), width=38)[:9]
            for li, line in enumerate(lines):
                draw.text((PAD, y + PAD + 22 + li * 17), line, font=f_prompt, fill=PROMPT_FG)

            for ci, (_, path) in enumerate(runs):
                x = PROMPT_W + ci * (size + PAD)
                img = _cell(os.path.join(path, f'{case}.png'), size)
                if img is None:
                    draw.rectangle([x, y, x + size, y + size], fill=MISSING_BG)
                else:
                    canvas.paste(img, (x, y))
            y += size + PAD

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canvas.save(args.output)
    print(f'{args.output}  ({width}x{height}, {n_rows} rows x {len(runs)} cols)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--artist', required=True)
    p.add_argument('--prompts_csv', default=None)
    p.add_argument('--cases', required=True, help='Comma-separated case_numbers, any order')
    p.add_argument('--runs', nargs='+', required=True, help='"LABEL:PATH" per column ("|" splits the label)')
    p.add_argument('--output', required=True)
    p.add_argument('--cell', type=int, default=240, help='Cell size in px')
    main(p.parse_args())
