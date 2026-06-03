"""Generate sample handwritten financial-report PDFs for testing.

We synthesize fixtures rather than scraping real filings so the repo is
self-contained, license-clean, and reproducible. Each sample mimics the target
problem: a printed quarterly report with CFO handwritten margin notes, plus a
mostly-handwritten "notes" page that should trigger the vision-LLM route.

Handwriting is rendered as an image (optionally with a handwriting TTF if one is
available locally or downloadable); embedding it as an image is exactly what
makes the page look "scanned/handwritten" to the router heuristic.

Usage:  python scripts/generate_samples.py [output_dir]
"""
from __future__ import annotations

import math
import random
import sys
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

random.seed(7)

# A free handwriting font (SIL OFL). Download is best-effort; falls back to the
# PIL default font with per-glyph jitter if unavailable/offline.
_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/"
    "caveat/Caveat%5Bwght%5D.ttf"
)


# Local calligraphic/cursive fonts that ship with many Linux images — used as a
# realistic handwriting stand-in when a true handwriting TTF can't be downloaded
# (e.g. in a locked-down build environment).
_LOCAL_CURSIVE_CANDIDATES = [
    "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrechorus-mediumitalic.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Italic.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
]


def _load_handwriting_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cache = Path.home() / ".cache" / "ipp_fonts"
    cache.mkdir(parents=True, exist_ok=True)
    ttf = cache / "caveat.ttf"
    # 1) cached/downloadable true handwriting font
    if not ttf.exists():
        try:
            urllib.request.urlretrieve(_FONT_URL, ttf)  # noqa: S310
        except Exception:
            ttf = None  # type: ignore[assignment]
    if ttf and ttf.exists():
        try:
            return ImageFont.truetype(str(ttf), size)
        except Exception:
            pass
    # 2) local cursive/calligraphic fallback
    for cand in _LOCAL_CURSIVE_CANDIDATES:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    # 3) last resort
    return ImageFont.load_default()


def _handwriting_image(lines: list[str], width: int = 1100, line_h: int = 70) -> Image.Image:
    font = _load_handwriting_font(46)
    img = Image.new("RGB", (width, line_h * len(lines) + 60), "white")
    draw = ImageDraw.Draw(img)
    y = 30
    ink = (20, 30, 110)  # blue-pen
    for line in lines:
        x = 40
        for ch in line:
            dy = random.randint(-4, 4)
            draw.text((x, y + dy), ch, font=font, fill=ink)
            # advance width (approximate when using default font)
            try:
                w = draw.textlength(ch, font=font)
            except Exception:
                w = 22
            x += w + random.uniform(0.5, 3.0)
        y += line_h
    return img


def _img_to_pdf_pixmap(img: Image.Image) -> fitz.Pixmap:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return fitz.Pixmap(buf.getvalue())


def build_quarterly_with_notes(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()  # A4-ish (default 595x842)

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

    # CFO handwritten margin notes as an embedded image
    notes = _handwriting_image(
        [
            "CFO notes - DO NOT release figs below w/o IR sign-off:",
            "  - Hardware -3% mostly FX; underlying +2%",
            "  - One-off legal charge $12.4M in OpEx (Q3 only)",
            "  - Buyback: ~$90M repurchased this qtr",
            "  - Watch: cloud gross margin slipped to 58.1%",
        ]
    )
    pix = _img_to_pdf_pixmap(notes)
    rect = fitz.Rect(55, y + 10, 540, y + 10 + (485 * pix.height / pix.width))
    page.insert_image(rect, pixmap=pix)

    doc.save(str(path))
    doc.close()


def build_handwritten_notes_page(path: Path) -> None:
    """A page that is almost entirely handwriting -> should route to vision LLM."""
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
        width=1400,
        line_h=90,
    )
    pix = _img_to_pdf_pixmap(img)
    w = 480
    rect = fitz.Rect(55, 60, 55 + w, 60 + w * pix.height / pix.width)
    page.insert_image(rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "samples"
    out.mkdir(parents=True, exist_ok=True)
    a = out / "sample_q3_report_with_handwriting.pdf"
    b = out / "sample_handwritten_board_notes.pdf"
    build_quarterly_with_notes(a)
    build_handwritten_notes_page(b)
    print(f"Wrote:\n  {a}\n  {b}")


if __name__ == "__main__":
    main()
