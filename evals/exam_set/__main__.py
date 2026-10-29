"""
Golden-set extension CLI.

  python -m evals.exam_set generate-sat                     # rewrite sat_math.jsonl
  python -m evals.exam_set generate-reading                 # rewrite ielts_reading.jsonl
  python -m evals.exam_set generate-writing                 # rewrite ielts_writing_prompts.jsonl
  python -m evals.exam_set generate-adversarial             # rewrite the adversarial writing set
  python -m evals.exam_set import-writing9 --limit 1200     # into evals/golden/local/
  python -m evals.exam_set import-cambridge --dir <folder>  # own book files, local OCR
  python -m evals.exam_set import-ieltsbuddy                # instructor-commented essays (polite)
  python -m evals.exam_set build-blind-pool --drafts 41     # needs a key; into ielts-local/blind/
  python -m evals.exam_set import-blind-scores              # owner's scores.csv -> captain_blind
  python -m evals.exam_set manifest                         # print the local manifest
  python -m evals.exam_set summary                          # counts by exam, skill and tier

Third-party data only ever lands in the gitignored local directory (override with
--local-dir or $EDU_GOLDEN_LOCAL_DIR).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from evals.exam_set import (
    blind_pool,
    ielts_reading,
    ielts_writing_adversarial,
    ielts_writing_prompts,
    sat_math,
)
from evals.exam_set.importers import cambridge, ieltsbuddy, writing9
from evals.exam_set.local_store import load_manifest, local_dir, read_items, write_items
from evals.loader import load_exam_items

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--local-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-sat")
    sub.add_parser("generate-reading")
    sub.add_parser("generate-writing")
    sub.add_parser("generate-adversarial")
    w9 = sub.add_parser("import-writing9")
    w9.add_argument("--limit", type=int, default=1200)
    w9.add_argument("--split", choices=list(writing9.SPLITS), default="train")
    w9.add_argument("--per-band-cap", type=int, default=None)
    cam = sub.add_parser("import-cambridge")
    cam.add_argument("--dir", required=True, help="folder of the captain's book text/PDF/scans")
    sub.add_parser("import-ieltsbuddy")
    bp = sub.add_parser("build-blind-pool")
    bp.add_argument("--drafts", type=int, default=41)
    bp.add_argument("--model", default="openai/gpt-4.1-mini")
    bp.add_argument("--blind-dir", default=str(blind_pool.BLIND_DIR))
    bp.add_argument("--ledger", default=".eval-ledger/ledger.json")
    isb = sub.add_parser("import-blind-scores")
    isb.add_argument("--blind-dir", default=str(blind_pool.BLIND_DIR))
    sub.add_parser("manifest")
    sub.add_parser("summary")
    args = p.parse_args(argv)
    directory = local_dir(args.local_dir)

    if args.cmd == "generate-sat":
        count, digest = write_items(GOLDEN_DIR / sat_math.FILE_NAME, sat_math.generate_all())
        print(json.dumps({"file": sat_math.FILE_NAME, "items": count, "sha256": digest}))
    elif args.cmd == "generate-reading":
        count, digest = write_items(GOLDEN_DIR / ielts_reading.FILE_NAME,
                                    ielts_reading.generate_all())
        print(json.dumps({"file": ielts_reading.FILE_NAME, "items": count, "sha256": digest}))
    elif args.cmd == "generate-writing":
        count, digest = write_items(GOLDEN_DIR / ielts_writing_prompts.FILE_NAME,
                                    ielts_writing_prompts.generate_all())
        print(json.dumps({"file": ielts_writing_prompts.FILE_NAME, "items": count,
                          "sha256": digest}))
    elif args.cmd == "generate-adversarial":
        count, digest = write_items(GOLDEN_DIR / ielts_writing_adversarial.FILE_NAME,
                                    ielts_writing_adversarial.generate_all())
        print(json.dumps({"file": ielts_writing_adversarial.FILE_NAME, "items": count,
                          "sha256": digest}))
    elif args.cmd == "import-writing9":
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            record, stats = writing9.import_writing9(
                directory, client, limit=args.limit, split=args.split,
                per_band_cap=args.per_band_cap,
            )
        print(json.dumps({"source": record.source_id, "file": str(directory / record.file),
                          "count": record.count, "tier": record.label_tier,
                          "bands": record.counts_by_overall_band, "stats": stats.__dict__}))
    elif args.cmd == "import-ieltsbuddy":
        with ieltsbuddy.make_client() as client:
            record, buddy_stats = ieltsbuddy.import_ieltsbuddy(directory, client)
        print(json.dumps({"source": record.source_id, "file": str(directory / record.file),
                          "count": record.count, "tier": record.label_tier,
                          "bands": record.counts_by_overall_band,
                          "stats": buddy_stats.__dict__}))
    elif args.cmd == "import-cambridge":
        try:
            record, totals = cambridge.import_folder(Path(args.dir), directory)
        except cambridge.OcrUnavailableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"source": record.source_id, "count": record.count,
                          "tier": record.label_tier, "totals": totals, "notes": record.notes,
                          "extracted_text": str(directory / cambridge.EXTRACT_DIR)}))
    elif args.cmd == "build-blind-pool":
        return _build_blind_pool(args, directory)
    elif args.cmd == "import-blind-scores":
        try:
            scored, empty = blind_pool.import_scores(
                Path(args.blind_dir), time.strftime("%Y-%m-%d")
            )
        except blind_pool.ScoreSheetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        imported = directory / blind_pool.IMPORTED_FILE
        fresh = {i.id for i in scored}
        kept = [i for i in read_items(imported) if i.id not in fresh] if imported.exists() else []
        write_items(imported, [*kept, *scored])
        print(json.dumps({"scored": len(scored), "kept_from_earlier_pools": len(kept),
                          "still_empty": empty,
                          "file": str(directory / blind_pool.IMPORTED_FILE)}))
    elif args.cmd == "manifest":
        print(json.dumps(load_manifest(directory), indent=2))
    else:
        items = load_exam_items(include_local=True, local_directory=directory)
        rows = Counter((i.exam, i.skill, i.label_provenance.tier.value, i.source.kind)
                       for i in items)
        print(json.dumps([
            {"exam": e, "skill": s, "tier": t, "source_kind": k, "n": n}
            for (e, s, t, k), n in sorted(rows.items())
        ], indent=2))
    return 0


def _build_blind_pool(args: argparse.Namespace, directory: Path) -> int:
    """Draft essays through the metered transport (budget ledger, fallbacks), then shuffle."""
    import asyncio
    import os

    import litellm

    from evals.agent_eval.budget import Ledger
    from evals.agent_eval.transport import MeteredTransport

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        print("ERROR: set OPENAI_API_KEY or OPENROUTER_API_KEY (names only)", file=sys.stderr)
        return 2
    try:
        blind_pool.ensure_unscored(Path(args.blind_dir))
    except blind_pool.PoolAlreadyScoredError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    prompts = [
        (i.id, i.prompt) for i in load_exam_items(exam="ielts", skill="writing_t2")
        if i.task_type == "essay_prompt"
    ]
    teacher = [
        i for i in load_exam_items(exam="ielts", skill="writing_t2", include_local=True,
                                   local_directory=directory)
        if i.label_provenance.tier.value == "teacher_graded" and i.response
    ]
    transport = MeteredTransport(Ledger(Path(args.ledger), 20.0), f"blind-pool-{time.time():.0f}")

    async def complete(messages: list[dict[str, str]]) -> str:
        response = await litellm.acompletion(
            model=args.model, messages=messages, temperature=0.9, max_tokens=700, stream=False
        )
        return str(response.choices[0].message.content or "")

    async def go() -> list[blind_pool.Draft]:
        return await blind_pool.draft_essays(prompts, args.drafts, complete)

    with transport.installed():
        drafts = asyncio.run(go())
    counts = blind_pool.build_pool(Path(args.blind_dir), drafts, teacher)
    print(json.dumps({**counts, "dir": args.blind_dir, "spent_usd": transport.spent_this_run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
