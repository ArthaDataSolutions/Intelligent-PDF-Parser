"""Generate sample handwritten financial-report PDFs for testing.

We synthesize fixtures rather than scraping real filings so the repo is
self-contained, license-clean, and reproducible. The goal is to mimic the
*messy* reality of client documents: clumsy handwriting on a phone-photographed
page, with uneven baselines, jittery glyph sizes/rotations, variable ink
pressure, mixed pen strokes, the odd crossed-out correction, and a skewed,
off-white paper background with sensor noise.

Embedding the handwriting as a (skewed, noisy) image is exactly what makes the
page look "scanned/handwritten" to the router heuristic and forces it down the
vision-LLM path.

Usage:  python scripts/generate_samples.py [output_dir]
"""
from __future__ import annotations

import math
import random
import sys
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(7)

# A free handwriting font (SIL OFL). Download is best-effort.
_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf"
)

# Local calligraphic/cursive fonts that ship with many Linux images. Mixing
# several of these per character is what gives the "no two letters alike" look.
_LOCAL_CURSIVE_CANDIDATES = [
    "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrechorus-mediumitalic.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Italic.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]


def _font_files() -> list[str]:
    """Return usable handwriting-ish font files, downloading Caveat if possible."""
    fonts: list[str] = []
    cache = Path.home() / ".cache" / "ipp_fonts"
    cache.mkdir(parents=True, exist_ok=True)
    ttf = cache / "caveat.ttf"
    if not ttf.exists():
        try:
            urllib.request.urlretrieve(_FONT_URL, ttf)  # noqa: S310
        except Exception:
            pass
    if ttf.exists():
        fonts.append(str(ttf))
    fonts.extend(c for c in _LOCAL_CURSIVE_CANDIDATES if Path(c).exists())
    return fonts


_FONT_FILES = _font_files()
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _glyph_tile(ch: str, font, ink: tuple[int, int, int], angle: float) -> Image.Image:
    """Render a single character to its own RGBA tile and rotate it."""
    pad = 14
    try:
        w = max(int(font.getlength(ch)), 1)
    except Exception:
        w = 18
    asc, desc = 1, 1
    try:
        asc, desc = font.getmetrics()
    except Exception:
        pass
    h = asc + desc
    tile = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    # variable ink pressure: alpha + slight darkness jitter
    alpha = random.randint(180, 255)
    jink = tuple(max(0, c + random.randint(-18, 18)) for c in ink)
    d.text((pad, pad), ch, font=font, fill=(*jink, alpha))
    return tile.rotate(angle, expand=True, resample=Image.BICUBIC)


def _write_messy(
    canvas: Image.Image,
    origin: tuple[int, int],
    lines: list[str],
    *,
    base_size: int = 44,
    line_h: int = 78,
    ink: tuple[int, int, int] = (24, 34, 96),
) -> None:
    """Composite clumsy handwriting onto an existing canvas in-place."""
    if not _FONT_FILES:
        # extreme fallback: default bitmap font, still jittered
        d = ImageDraw.Draw(canvas)
        x0, y = origin
        for line in lines:
            d.text((x0, y), line, fill=ink)
            y += line_h
        return

    ox, oy = origin
    for li, line in enumerate(lines):
        # each line gets its own slope and slowly drifting baseline
        slope = random.uniform(-0.045, 0.05)
        line_size = base_size + random.randint(-3, 4)
        x = ox + random.randint(-6, 10)
        baseline = oy + li * line_h + random.randint(-5, 5)
        word_start_x = x
        for ch in line:
            if ch == " ":
                x += line_size * random.uniform(0.28, 0.5)
                word_start_x = x
                continue
            size = max(18, line_size + random.randint(-5, 6))
            font = _font(random.choice(_FONT_FILES), size)
            angle = random.uniform(-13, 13)
            tile = _glyph_tile(ch, font, ink, angle)
            drift = int((x - ox) * slope)              # whole-line slant
            jitter_y = random.randint(-6, 6)           # per-glyph bounce
            canvas.alpha_composite(tile, (int(x), int(baseline + drift + jitter_y)))
            try:
                adv = font.getlength(ch)
            except Exception:
                adv = size * 0.5
            # uneven spacing; occasionally let letters touch/overlap
            x += adv * random.uniform(0.7, 1.02) + random.uniform(-2, 4)
        # occasional crossed-out correction on a word
        if random.random() < 0.18 and x > word_start_x + 30:
            d = ImageDraw.Draw(canvas)
            sy = baseline + line_size * 0.45 + random.randint(-4, 4)
            d.line(
                [(word_start_x, sy), (x - 10, sy + random.randint(-5, 5))],
                fill=(*ink, 230),
                width=random.randint(2, 4),
            )


def _paper(width: int, height: int) -> Image.Image:
    """Off-white photographed-paper canvas with faint noise and shading."""
    base = random.randint(238, 248)
    img = Image.new("RGBA", (width, height), (base, base - 3, base - 8, 255))
    # sensor noise
    noise = Image.effect_noise((width, height), 14).convert("L")
    noise_rgba = Image.merge(
        "RGBA", (noise, noise, noise, Image.new("L", (width, height), 26))
    )
    img = Image.alpha_composite(img, noise_rgba)
    # soft uneven lighting (a darker corner, like a phone shadow)
    shade = Image.new("L", (width, height), 0)
    sd = ImageDraw.Draw(shade)
    cx, cy = random.randint(0, width), random.randint(0, height)
    sd.ellipse([cx - width, cy - height, cx + 40, cy + 40], fill=40)
    shade = shade.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dark.putalpha(shade)
    return Image.alpha_composite(img, dark)


def _scanify(text_canvas: Image.Image) -> Image.Image:
    """Skew + slightly blur the whole page so it reads as a phone scan."""
    angle = random.uniform(-3.2, 3.2)
    skewed = text_canvas.rotate(
        angle, expand=True, resample=Image.BICUBIC, fillcolor=(244, 241, 236, 255)
    )
    return skewed.filter(ImageFilter.GaussianBlur(0.6))


def _handwriting_image(lines: list[str], width: int = 1300, line_h: int = 82) -> Image.Image:
    height = line_h * len(lines) + 140
    canvas = _paper(width, height)
    _write_messy(canvas, (60, 50), lines, line_h=line_h)
    return _scanify(canvas).convert("RGB")


def _img_to_pdf_pixmap(img: Image.Image) -> fitz.Pixmap:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return fitz.Pixmap(buf.getvalue())


def build_quarterly_with_notes(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()

    printed = [
        ("Northwind Industries, Inc.", 20, True),
        ("Q3 FY2026 Financial Results (unaudited)", 13, True),
        ("", 6, False),
        ("Revenue:                      $1,284.6M   (+11.4% YoY)", 11, False),
        ("Gross margin:                  41.2%      (+120 bps YoY)", 11, False),
        ("Operating income:             $208.3M    (16.2% margin)", 11, False),
        ("Net income:                   $151.7M", 11, False),
        ("Diluted EPS:                  $1.07", 11, False),
        ("Free cash flow:               $176.0M", 11, False),
        ("Cash & equivalents:           $612.4M", 11, False),
        ("Total debt:                   $940.0M", 11, False),
        ("", 6, False),
        ("Segment revenue:", 12, True),
        ("   Cloud Platform:            $742.1M   (+19% YoY)", 11, False),
        ("   Hardware:                  $392.5M   (-3% YoY)", 11, False),
        ("   Services:                  $150.0M   (+8% YoY)", 11, False),
        ("", 6, False),
        ("FY2026 guidance: revenue $5.15-5.25B; operating margin ~16.5%.", 11, False),
    ]
    y = 60
    for text, size, bold in printed:
        page.insert_text(
            (60, y),
            text,
            fontsize=size,
            fontname="helv" if not bold else "hebo",
            color=(0, 0, 0),
        )
        y += size + 8

    notes = _handwriting_image(
        [
            "CFO notes - DO NOT release figs below w/o IR sign-off:",
            "  - Hardware -3% mostly FX; underlying +2%",
            "  - One-off legal charge $12.4M in OpEx (Q3 only)",
            "  - Buyback: ~$90M repurchased this qtr",
            "  - Watch: cloud gross margin slipped to 58.1%",
        ],
        width=1250,
        line_h=78,
    )
    pix = _img_to_pdf_pixmap(notes)
    rect = fitz.Rect(55, y + 6, 545, y + 6 + (490 * pix.height / pix.width))
    page.insert_image(rect, pixmap=pix)

    doc.save(str(path))
    doc.close()


def build_handwritten_notes_page(path: Path) -> None:
    """A page that is almost entirely clumsy handwriting -> vision-LLM route."""
    doc = fitz.open()
    page = doc.new_page()
    img = _handwriting_image(
        [
            "Board meeting notes - 14 May 2026",
            "Q3 came in ahead of plan. Revenue $1,284.6M.",
            "Net income $151.7M, EPS $1.07.",
            "Action: revisit FY guidance - lean to high end.",
            "Concern: hardware demand soft in EMEA.",
            "Cash strong at $612M; debt $940M - refinance Q1?",
            "Dividend: hold at $0.18/share.",
        ],
        width=1500,
        line_h=104,
    )
    pix = _img_to_pdf_pixmap(img)
    w = 500
    rect = fitz.Rect(48, 55, 48 + w, 55 + w * pix.height / pix.width)
    page.insert_image(rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "samples"
    )
    out.mkdir(parents=True, exist_ok=True)
    a = out / "sample_q3_report_with_handwriting.pdf"
    b = out / "sample_handwritten_board_notes.pdf"
    build_quarterly_with_notes(a)
    build_handwritten_notes_page(b)
    print(f"Wrote:\n  {a}\n  {b}")


if __name__ == "__main__":
    main()
