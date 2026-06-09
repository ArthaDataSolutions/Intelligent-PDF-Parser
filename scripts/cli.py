"""Command-line entry: parse a PDF and optionally generate analyst Q&A.

Examples:
  python scripts/cli.py parse samples/sample_handwritten_board_notes.pdf
  python scripts/cli.py qa samples/sample_q3_report_with_handwriting.pdf -n 10 -o qa.md
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import Pipeline  # noqa: E402
from app.qa.generator import QAGenerator  # noqa: E402


async def _run(args: argparse.Namespace) -> None:
    raw = Path(args.pdf).read_bytes()
    pipe = Pipeline()
    if args.command == "parse":
        resp = await pipe.parse_only(raw, Path(args.pdf).name)
        print(resp.parse.markdown)
        print("\n---\nbackends:", resp.parse.backend_summary, file=sys.stderr)
        for w in resp.parse.warnings:
            print("warning:", w, file=sys.stderr)
    else:
        peers = [p.strip() for p in (args.peers or "").split(",") if p.strip()]
        resp = await pipe.process(
            raw, Path(args.pdf).name, num_questions=args.num, peers=peers or None
        )
        md = QAGenerator.to_markdown(resp.qa) if resp.qa else "(no Q&A)"
        if args.out:
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"Wrote {args.out}", file=sys.stderr)
        else:
            print(md)
        print("timings_ms:", resp.timings_ms, file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Intelligent PDF Parser CLI")
    sub = p.add_subparsers(dest="command", required=True)
    pp = sub.add_parser("parse", help="parse PDF to markdown")
    pp.add_argument("pdf")
    pq = sub.add_parser("qa", help="parse + generate analyst Q&A")
    pq.add_argument("pdf")
    pq.add_argument("-n", "--num", type=int, default=12, help="number of Q&A pairs")
    pq.add_argument("-o", "--out", help="write Q&A markdown to this file")
    pq.add_argument(
        "-p", "--peers",
        help="comma-separated comparable pharma companies for analyst "
        "peer-comparison questions (e.g. 'Pfizer,Merck,Novartis')",
    )
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
