"""
Does retrieval find the right note, and does it stay quiet when no note applies?

Indexes the seed notes (`seed/gre-quant/`, chunked and embedded exactly as the seed job does)
into a throwaway Qdrant container, then replays the labelled queries in
`evals/golden/gre_retrieval.jsonl` through the corpus service's own search code under three
configurations:

    dense          dense vectors only (HYBRID_RETRIEVAL_ENABLED=false)
    hybrid         dense + lexical sparse, fused with RRF (the default)
    hybrid+rerank  hybrid, then the cross-encoder over the top 20 (needs --reranker-url)

Each query has a group: `exact` (a GRE term or symbol), `paraphrase` (a student's own words) or
`off_topic` (no note applies; the right answer is to show nothing). It reports recall@3 and
MRR@3 for the first two groups, how often an off-topic query still gets a note, and a sweep of
the relevance floor (`corpus.search.passes_relevance_floor`) showing what each setting costs and
buys. Run:

    uv run python -m evals.retrieval_eval --embedder-url http://localhost:8080
    uv run python -m evals.retrieval_eval --embedder-url ... --reranker-url http://localhost:8081
    uv run python -m evals.retrieval_eval ... --json-out evals/baselines/retrieval.json

Needs Docker (for Qdrant) and a running embedder (`docker compose up -d tei-embedder`). Nothing
here calls a language model. The embedder is the only slow part, and only the first time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.qdrant_filter_check import start_qdrant, stop_qdrant
from services.corpus.src.corpus.search import passes_relevance_floor, search_candidates
from services.corpus.src.corpus.settings import CorpusSettings
from services.ingestion.src.ingestion import pipeline as ingestion
from services.ingestion.src.ingestion.settings import get_settings as get_ingestion_settings
from shared.src.shared.config import anchor
from shared.src.shared.stats import percentile

GOLDEN_PATH = Path(__file__).parent / "golden" / "gre_retrieval.jsonl"
SEED_DIR = "seed/gre-quant"
EMBEDDER_MODEL = "BAAI/bge-m3"  # what infra/docker-compose.yml serves as tei-embedder
GROUPS = ("exact", "paraphrase", "off_topic")
TOP_K = 3  # what the Quant tutor asks for (agents/quant.py `_retrieve`)
CONFIGS = ("dense", "hybrid", "hybrid+rerank")
# Candidate floors swept per signal: cosine similarity and the reranker's 0..1 score.
COSINE_FLOORS = tuple(round(0.30 + 0.025 * i, 3) for i in range(0, 17))
RERANK_FLOORS = tuple(round(0.05 * i, 2) for i in range(0, 20))


@dataclass(frozen=True)
class Query:
    id: str
    group: str
    query: str
    expect: tuple[str, ...]  # acceptable "<file>#<section>" labels; empty for off_topic


@dataclass(frozen=True)
class Hit:
    label: str
    score: float  # the score the corpus returns (RRF, cosine or reranker, by configuration)
    dense_score: float | None
    reranked: bool


@dataclass(frozen=True)
class QueryRun:
    query: Query
    hits: tuple[Hit, ...]
    latency_ms: float


def load_queries(path: Path = GOLDEN_PATH) -> list[Query]:
    queries = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            queries.append(Query(row["id"], row["group"], row["query"], tuple(row["expect"])))
    return queries


def seed_labels(seed_dir: Path) -> set[str]:
    """Every "<file>#<section>" the seed notes contain, as the chunker will label them."""
    labels: set[str] = set()
    for path in sorted(seed_dir.glob("*.txt")):
        chunks = ingestion.build_chunks(path.read_bytes(), path.name, "gre", path.name)
        labels.update(f"{c['source_uri']}#{c['page_or_section']}" for c in chunks)
    return labels


def hit_label(hit: dict[str, Any]) -> str:
    payload = hit["payload"]
    return f"{payload['source_uri']}#{payload['page_or_section']}"


def first_rank(expect: Sequence[str], labels: Sequence[str]) -> int | None:
    """1-based rank of the first acceptable label, or None if none was retrieved."""
    for rank, label in enumerate(labels, 1):
        if label in expect:
            return rank
    return None


def floor_settings(floor: float, rerank: bool) -> CorpusSettings:
    """The corpus settings a swept floor stands for (a rerank floor sits on the cosine floor)."""
    if rerank:
        return CorpusSettings(rerank_relevance_floor=floor)
    return CorpusSettings(dense_relevance_floor=floor)


def kept(run: QueryRun, settings: CorpusSettings | None) -> list[Hit]:
    """The hits the corpus floor lets through; settings None means no floor (the old behaviour)."""
    if settings is None:
        return list(run.hits)
    return [
        h
        for h in run.hits
        if passes_relevance_floor(
            {"score": h.score, "dense_score": h.dense_score, "reranked": h.reranked}, settings
        )
    ]


def quality(
    runs: Sequence[QueryRun], settings: CorpusSettings | None = None
) -> dict[str, Any]:
    """recall@3 and MRR@3 per on-topic group; share of off-topic queries still shown a note."""
    out: dict[str, Any] = {}
    for group in GROUPS:
        group_runs = [r for r in runs if r.query.group == group]
        if not group_runs:
            continue
        if group == "off_topic":
            shown = sum(1 for r in group_runs if kept(r, settings))
            out[group] = {"n": len(group_runs), "notes_shown": shown / len(group_runs)}
            continue
        ranks = [
            first_rank(r.query.expect, [h.label for h in kept(r, settings)]) for r in group_runs
        ]
        out[group] = {
            "n": len(group_runs),
            "recall_at_3": sum(1 for k in ranks if k is not None) / len(ranks),
            "mrr_at_3": sum(1 / k for k in ranks if k is not None) / len(ranks),
        }
    return out


def sweep(
    runs: Sequence[QueryRun], floors: Sequence[float], rerank: bool = False
) -> list[dict[str, Any]]:
    """Per floor: on-topic recall@3 kept, and the off-topic queries still shown a note."""
    on_topic = [r for r in runs if r.query.group != "off_topic"]
    off_topic = [r for r in runs if r.query.group == "off_topic"]
    rows = []
    for floor in floors:
        settings = floor_settings(floor, rerank)
        found = sum(
            1
            for r in on_topic
            if first_rank(r.query.expect, [h.label for h in kept(r, settings)]) is not None
        )
        rows.append(
            {
                "floor": floor,
                "on_topic_recall_at_3": found / len(on_topic),
                "off_topic_notes_shown": sum(1 for r in off_topic if kept(r, settings))
                / len(off_topic),
            }
        )
    return rows


def best_floor(rows: Sequence[dict[str, Any]], max_off_topic: float = 0.10) -> dict[str, Any]:
    """
    The floor that keeps the most on-topic recall while at most `max_off_topic` of the off-topic
    queries still get a note; among equals, the lowest (the least aggressive). If no floor in
    the sweep is quiet enough, the highest one.
    """
    quiet = [r for r in rows if r["off_topic_notes_shown"] <= max_off_topic]
    if not quiet:
        return max(rows, key=lambda r: r["floor"])
    return max(quiet, key=lambda r: (r["on_topic_recall_at_3"], -r["floor"]))


# ----------------------------------------------------------------------- running
async def run_config(queries: Sequence[Query], settings: CorpusSettings) -> list[QueryRun]:
    runs = []
    for q in queries:
        start = time.perf_counter()
        ranked = await search_candidates(q.query, "eval", "", "gre", TOP_K, settings=settings)
        latency = (time.perf_counter() - start) * 1000
        hits = tuple(
            Hit(
                hit_label(h),
                float(h["score"]),
                h.get("dense_score"),
                bool(h.get("reranked")),
            )
            for h in ranked
        )
        runs.append(QueryRun(q, hits, latency))
    return runs


async def index_seed(seed_dir: Path, qdrant_url: str, embedder_url: str) -> int:
    settings = get_ingestion_settings()
    settings.qdrant_url, settings.tei_embedder_url = qdrant_url, embedder_url
    total = 0
    for path in sorted(seed_dir.glob("*.txt")):
        chunks = ingestion.build_chunks(path.read_bytes(), path.name, "gre", path.name)
        total += (await ingestion.run_ingestion_pipeline(chunks, {}))["chunks_indexed"]
    return total


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"queries: {report['queries']}  chunks indexed: {report['chunks']}  top_k: {TOP_K}",
        "",
        "config          group       n  recall@3  MRR@3  off-topic notes shown",
    ]
    for name, cfg in report["configs"].items():
        if cfg is None:
            lines.append(f"{name:<15} not measured: {report['skipped'][name]}")
            continue
        for group in GROUPS:
            g = cfg["quality"].get(group)
            if g is None:
                continue
            if group == "off_topic":
                body = f"{'-':>8} {'-':>6}  {g['notes_shown']:>10.2f}"
            else:
                body = f"{g['recall_at_3']:>8.2f} {g['mrr_at_3']:>6.2f}  {'-':>10}"
            lines.append(f"{name:<15} {group:<11}{g['n']:>2} {body}")
        lat = cfg["latency_ms"]
        lines.append(f"{'':<15} latency per query p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms")
    for name, cfg in report["configs"].items():
        if cfg is None:
            continue
        lines += [
            "",
            f"floor sweep, {name} ({cfg['signal']})",
            "  floor  on-topic recall@3  off-topic notes shown",
        ]
        lines += [
            f"  {r['floor']:>5.3f}  {r['on_topic_recall_at_3']:>16.2f}  "
            f"{r['off_topic_notes_shown']:>21.2f}"
            for r in cfg["sweep"]
        ]
        lines.append(f"  suggested floor: {cfg['suggested_floor']['floor']}")
    return "\n".join(lines)


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_queries()
    seed_dir = anchor(SEED_DIR)
    name, port = start_qdrant()
    try:
        qdrant_url = f"http://127.0.0.1:{port}"
        chunks = await index_seed(seed_dir, qdrant_url, args.embedder_url)
        base = {"qdrant_url": qdrant_url, "tei_embedder_url": args.embedder_url}
        settings = {
            "dense": CorpusSettings(**base, hybrid_retrieval_enabled=False),
            "hybrid": CorpusSettings(**base, hybrid_retrieval_enabled=True),
            "hybrid+rerank": CorpusSettings(
                **base, hybrid_retrieval_enabled=True, tei_reranker_url=args.reranker_url
            ),
        }
        report: dict[str, Any] = {
            "embedder": EMBEDDER_MODEL,
            "queries": len(queries),
            "chunks": chunks,
            "configs": {},
            "skipped": {},
        }
        for config in CONFIGS:
            if config == "hybrid+rerank" and not args.reranker_url:
                report["configs"][config] = None
                report["skipped"][config] = "no --reranker-url (the reranker is opt-in)"
                continue
            runs = await run_config(queries, settings[config])
            reranked = config == "hybrid+rerank"
            unranked = sum(1 for r in runs if r.hits and not all(h.reranked for h in r.hits))
            if reranked and unranked:
                report["configs"][config] = None
                report["skipped"][config] = (
                    f"the reranker did not answer for {unranked} of {len(runs)} queries "
                    "(it fails open to the RRF order)"
                )
                continue
            rows = sweep(runs, RERANK_FLOORS if reranked else COSINE_FLOORS, rerank=reranked)
            latencies = [r.latency_ms for r in runs]
            report["configs"][config] = {
                "signal": "reranker score" if reranked else "dense cosine similarity",
                "quality": quality(runs),
                "latency_ms": {
                    "p50": statistics.median(latencies),
                    "p95": percentile(latencies, 0.95),
                },
                "sweep": rows,
                "suggested_floor": best_floor(rows),
                "misses": [
                    {"id": r.query.id, "query": r.query.query, "got": [h.label for h in r.hits]}
                    for r in runs
                    if r.query.group != "off_topic"
                    and first_rank(r.query.expect, [h.label for h in r.hits]) is None
                ],
                "runs": [asdict(r) for r in runs],
            }
        return report
    finally:
        stop_qdrant(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval eval: recall, MRR and the relevance floor")
    p.add_argument("--embedder-url", default="http://localhost:8080")
    p.add_argument("--reranker-url", default="", help="TEI /rerank base URL; empty skips rerank")
    p.add_argument(
        "--json-out", type=Path, help="write the summary (no per-query runs): the baseline file"
    )
    p.add_argument("--runs-out", type=Path, help="write the full report, every query's hits")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(evaluate(args))
    print(format_report(report))
    if args.runs_out:
        _write(args.runs_out, report)
    if args.json_out:
        summary = {
            **report,
            "configs": {
                name: None if cfg is None else {k: v for k, v in cfg.items() if k != "runs"}
                for name, cfg in report["configs"].items()
            },
        }
        _write(args.json_out, summary)
    return 0


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
