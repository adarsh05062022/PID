"""
Side-by-side comparison grid: one row per prompt, one column per run.

Reads run directories in the layout run_with_steering.py writes
(`<output_dir>/<prompt_idx>/<seed>-<i>.png`, plus a `prompt.txt` per prompt
dir when --prompts_csv was used) and stacks them into a single figure, with
the prompt in a left-hand panel and a column header per arm.

Rows are keyed by prompt directory name, so every column shows the *same*
prompt at the *same* seed -- which is the only way the comparison means
anything. A prompt missing from one run leaves that cell blank rather than
shifting the row.

    python scripts/diffusion/make_grid.py \
        --runs "SD-1.4:results/sd14/unsafe_plus_safe/baseline" \
               "CASteer b=2.0:results/sd14/unsafe_plus_safe/kp_only_check/casteer" \
        --output results/sd14/unsafe_plus_safe/grid_compare.png

Each --runs entry is "LABEL:PATH". Order is preserved left to right.
"""
import argparse
import glob
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

HEADER_H = 46
PAD = 6
BG = (232, 232, 236)
HEADER_BG = (28, 28, 34)
HEADER_FG = (245, 245, 250)
PROMPT_BG = (243, 245, 252)
PROMPT_FG = (18, 18, 45)
MISSING_BG = (214, 214, 220)


def _font(size: int, bold: bool = False):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
        '/usr/share/fonts/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_w(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def _centered(draw, text, font, x0, width, y, fill):
    draw.text((x0 + max(0, (width - _text_w(draw, text, font)) // 2), y), text, font=font, fill=fill)


def _header(draw, text, x0, width, fill):
    """Draw a column label centred in `width`, wrapping to two lines and
    shrinking the font as needed so neighbouring columns never overlap."""
    for size in (17, 15, 13, 11, 10):
        font = _font(size, bold=True)
        if _text_w(draw, text, font) <= width - 4:
            _centered(draw, text, font, x0, width, (HEADER_H - size) // 2, fill)
            return
        # Two lines: split at the space nearest the middle.
        spaces = [i for i, c in enumerate(text) if c == ' ']
        if spaces:
            cut = min(spaces, key=lambda i: abs(i - len(text) // 2))
            top, bottom = text[:cut], text[cut + 1:]
            if max(_text_w(draw, top, font), _text_w(draw, bottom, font)) <= width - 4:
                y = (HEADER_H - 2 * size - 4) // 2
                _centered(draw, top, font, x0, width, y, fill)
                _centered(draw, bottom, font, x0, width, y + size + 4, fill)
                return
    font = _font(10, bold=True)
    _centered(draw, text, font, x0, width, (HEADER_H - 10) // 2, fill)


def prompt_dirs(run: str) -> list:
    """Prompt sub-directories of a run, numeric ones sorted numerically."""
    names = [n for n in os.listdir(run) if os.path.isdir(os.path.join(run, n)) and n != 'pid_records']
    numeric = sorted((n for n in names if n.isdigit()), key=int)
    return numeric or sorted(names)


def cell_image(run: str, name: str):
    hits = sorted(glob.glob(os.path.join(run, name, '*.png')))
    return hits[0] if hits else None


def read_prompt(runs: list, name: str) -> str:
    for _, path in runs:
        txt = os.path.join(path, name, 'prompt.txt')
        if os.path.exists(txt):
            with open(txt) as f:
                return f.read().strip()
    return name


def main(args):
    runs = []
    for entry in args.runs:
        label, _, path = entry.partition(':')
        if not path:
            raise ValueError(f'--runs entries are "LABEL:PATH"; got {entry!r}')
        runs.append((label, path))

    # Row order comes from the first run; later runs only fill cells.
    names = prompt_dirs(runs[0][1])
    if args.n:
        names = names[:args.n]

    thumb = args.thumb
    prompt_w = args.prompt_width
    f_head = _font(17, bold=True)
    f_prompt = _font(13)
    f_idx = _font(13, bold=True)

    width = prompt_w + len(runs) * (thumb + PAD) + PAD
    height = HEADER_H + len(names) * (thumb + PAD) + PAD
    canvas = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, width, HEADER_H], fill=HEADER_BG)
    _centered(draw, 'prompt', f_head, 0, prompt_w, (HEADER_H - 17) // 2, HEADER_FG)
    for col, (label, _) in enumerate(runs):
        _header(draw, label, prompt_w + col * (thumb + PAD), thumb, HEADER_FG)

    # Wrap width in characters, from the average glyph width of the font.
    char_w = max(1, _text_w(draw, 'n' * 20, f_prompt) // 20)
    wrap_cols = max(12, (prompt_w - 2 * PAD - 24) // char_w)

    for row, name in enumerate(names):
        y0 = HEADER_H + row * (thumb + PAD) + PAD
        draw.rectangle([PAD, y0, prompt_w - PAD, y0 + thumb], fill=PROMPT_BG)
        draw.text((PAD + 6, y0 + 6), f'#{name}', font=f_idx, fill=PROMPT_FG)
        text = textwrap.fill(read_prompt(runs, name), wrap_cols)
        draw.text((PAD + 6, y0 + 26), text, font=f_prompt, fill=PROMPT_FG)

        for col, (_, path) in enumerate(runs):
            x0 = prompt_w + col * (thumb + PAD)
            src = cell_image(path, name)
            if src is None:
                draw.rectangle([x0, y0, x0 + thumb, y0 + thumb], fill=MISSING_BG)
                _centered(draw, 'missing', f_prompt, x0, thumb, y0 + thumb // 2, PROMPT_FG)
                continue
            with Image.open(src) as im:
                canvas.paste(im.convert('RGB').resize((thumb, thumb), Image.LANCZOS), (x0, y0))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canvas.save(args.output)
    print(f'{len(names)} prompts x {len(runs)} runs -> {args.output} ({width}x{height})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, help='"LABEL:PATH" per column, left to right')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=0, help='0 = all prompts')
    p.add_argument('--thumb', type=int, default=256)
    p.add_argument('--prompt_width', type=int, default=300)
    main(p.parse_args())
