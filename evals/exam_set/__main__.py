"""
Golden-set extension CLI.

  python -m evals.exam_set generate-sat                     # rewrite sat_math.jsonl
  python -m evals.exam_set generate-reading                 # rewrite ielts_reading.jsonl
  python -m evals.exam_set import-writing9 --limit 1200     # into evals/golden/local/
  python -m evals.exam_set import-cambridge --dir <folder>  # own book files, local OCR
  python -m evals.exam_set manifest                         # print the local manifest
  python -m evals.exam_set summary                          # counts by exam, skill and tier

Third-party data only ever lands in the gitignored local directory (override with
--local-dir or $EDU_GOLDEN_LOCAL_DIR).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import httpx

from evals.exam_set import ielts_reading, sat_math
from evals.exam_set.importers import cambridge, writing9
from evals.exam_set.local_store import load_manifest, local_dir, write_items
from evals.loader import load_exam_items

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--local-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-sat")
    sub.add_parser("generate-reading")
    w9 = sub.add_parser("import-writing9")
    w9.add_argument("--limit", type=int, default=1200)
    w9.add_argument("--split", choices=list(writing9.SPLITS), default="train")
    w9.add_argument("--per-band-cap", type=int, default=None)
    cam = sub.add_parser("import-cambridge")
    cam.add_argument("--dir", required=True, help="folder of the captain's book text/PDF/scans")
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
    elif args.cmd == "import-writing9":
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            record, stats = writing9.import_writing9(
                directory, client, limit=args.limit, split=args.split,
                per_band_cap=args.per_band_cap,
            )
        print(json.dumps({"source": record.source_id, "file": str(directory / record.file),
                          "count": record.count, "tier": record.label_tier,
                          "bands": record.counts_by_overall_band, "stats": stats.__dict__}))
    elif args.cmd == "import-cambridge":
        try:
            record, totals = cambridge.import_folder(Path(args.dir), directory)
        except cambridge.OcrUnavailableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"source": record.source_id, "count": record.count,
                          "tier": record.label_tier, "totals": totals,
                          "extracted_text": str(directory / cambridge.EXTRACT_DIR)}))
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


if __name__ == "__main__":
    raise SystemExit(main())
