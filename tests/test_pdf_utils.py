from app.utils.pdf import analyze_pages, page_count, render_page_png


def test_text_pdf_not_flagged(text_pdf):
    assert page_count(text_pdf) == 2
    sigs = analyze_pages(text_pdf)
    assert len(sigs) == 2
    assert all(not s.likely_handwritten for s in sigs)
    assert all(s.char_count > 50 for s in sigs)


def test_scanned_pdf_flagged_for_vision(scanned_pdf):
    sigs = analyze_pages(scanned_pdf)
    assert len(sigs) == 1
    assert sigs[0].likely_scanned is True
    assert sigs[0].likely_handwritten is True
    assert sigs[0].char_count < 20


def test_handwriting_thresholds_can_disable_scanned_signal(scanned_pdf):
    sigs = analyze_pages(
        scanned_pdf,
        scan_char_threshold=0,
        image_area_threshold=1.0,
        text_area_threshold=0.0,
    )
    assert sigs[0].likely_scanned is False
    assert sigs[0].likely_handwritten is False


def test_render_page_png(text_pdf):
    png = render_page_png(text_pdf, 0, dpi=100)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
