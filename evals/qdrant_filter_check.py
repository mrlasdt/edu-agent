"""
Does the real corpus access filter find the right chunks, and how fast?

Builds a synthetic multi-tenant corpus (clustered random vectors, no model
calls), loads it into a throwaway `qdrant/qdrant` container, and replays the
production filter (`build_search_filter`: test_type AND (global OR school OR
candidate)) as many different callers. For each caller it compares Qdrant's
top-k with an exact brute-force top-k over the same filtered set, so a filter
that silently drops results shows up as low recall@k instead of passing
unnoticed.

Per variant it reports recall@k, mean rows returned, p50/p95 latency and which
payload fields are indexed. Run:

    uv run python -m evals.qdrant_filter_check                 # 100k points
    uv run python -m evals.qdrant_filter_check --points 20000  # quicker

Needs Docker. The container is started on a free local port and removed on
exit, including on Ctrl-C. Nothing here touches your real Qdrant volume.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import socket
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from services.corpus.src.corpus.retriever import build_search_filter
from shared.src.shared.embedder import EMBED_DIM
from shared.src.shared.vectorstore import (
    COLLECTION,
    DENSE_VECTOR,
    ensure_collection,
)

# The server version the compose file pins (infra/docker-compose.yml).
QDRANT_IMAGE = "qdrant/qdrant:v1.18.0"
TEST_TYPES = ("gre", "ielts")
# A wider search beam measured next to Qdrant's default, to show whether it helps.
PROBE_EF = 400

# How the corpus splits across tiers and how concentrated each tier is.
# Fractions are shares of all chunks; schools and candidates follow a Zipf-like
# law (a few large schools, a long tail of small ones), as real tenants do.
PROFILES: dict[str, dict[str, float | int]] = {
    "global-heavy": {  # a shared question bank dominates; tenants add a little
        "global": 0.60, "school": 0.30, "candidate": 0.10,
        "schools": 40, "candidates": 2000, "gre_share": 0.8,
    },
    "tenant-heavy": {  # most chunks are private uploads: filters are selective
        "global": 0.10, "school": 0.45, "candidate": 0.45,
        "schools": 100, "candidates": 5000, "gre_share": 0.8,
    },
}


# --------------------------------------------------------------------- corpus
@dataclass
class Corpus:
    vectors: np.ndarray  # (n, dim) float32, L2-normalised
    test_type: np.ndarray  # (n,) int, index into TEST_TYPES
    tier: np.ndarray  # (n,) int: 0 global, 1 school, 2 candidate
    school: np.ndarray  # (n,) int, -1 unless tier == school
    candidate: np.ndarray  # (n,) int, -1 unless tier == candidate
    candidate_school: np.ndarray  # (candidates,) int: each candidate's school
    centers: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))

    def payloads(self) -> list[dict[str, Any]]:
        tiers = ("global", "school", "candidate")
        return [
            {
                "tier": tiers[int(t)],
                "test_type": TEST_TYPES[int(tt)],
                "school_id": school_name(int(s)) if s >= 0 else None,
                "candidate_id": candidate_name(int(c)) if c >= 0 else None,
            }
            for t, tt, s, c in zip(self.tier, self.test_type, self.school, self.candidate)
        ]


def school_name(i: int) -> str:
    return f"school-{i:04d}"


def candidate_name(i: int) -> str:
    return f"cand-{i:06d}"


def _zipf_choice(rng: np.random.Generator, n_items: int, size: int) -> np.ndarray:
    weights = 1.0 / np.arange(1, n_items + 1)
    return rng.choice(n_items, size=size, p=weights / weights.sum())


def _unit(x: np.ndarray) -> np.ndarray:
    unit: np.ndarray = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return unit


def sample_vectors(
    rng: np.random.Generator, centers: np.ndarray, size: int, spread: float = 0.5
) -> np.ndarray:
    """Points from a Gaussian mixture: topical clusters, like real embeddings."""
    which = rng.integers(0, len(centers), size=size)
    return _unit(centers[which] + spread * rng.standard_normal((size, centers.shape[1])))


def make_corpus(n: int, dim: int, profile: str, seed: int) -> Corpus:
    p = PROFILES[profile]
    rng = np.random.default_rng(seed)
    n_schools, n_candidates = int(p["schools"]), int(p["candidates"])
    tier = rng.choice(3, size=n, p=[p["global"], p["school"], p["candidate"]])
    school = np.where(tier == 1, _zipf_choice(rng, n_schools, n), -1)
    candidate = np.where(tier == 2, _zipf_choice(rng, n_candidates, n), -1)
    centers = rng.standard_normal((200, dim))
    return Corpus(
        vectors=sample_vectors(rng, centers, n),
        test_type=(rng.random(n) >= p["gre_share"]).astype(int),
        tier=tier,
        school=school,
        candidate=candidate,
        candidate_school=_zipf_choice(rng, n_schools, n_candidates),
        centers=centers,
    )


@dataclass(frozen=True)
class Caller:
    candidate: int
    school: int  # -1 = the candidate has no school
    test_type: str = "gre"


def sample_callers(corpus: Corpus, count: int, seed: int) -> list[Caller]:
    """Callers who own at least one chunk, so every query has a private slice."""
    rng = np.random.default_rng(seed + 1)
    owners = np.unique(corpus.candidate[corpus.candidate >= 0])
    picked = rng.choice(owners, size=min(count, len(owners)), replace=False)
    callers = []
    for i, c in enumerate(picked):
        no_school = i % 10 == 9  # one in ten has no school: school_id == ""
        callers.append(Caller(int(c), -1 if no_school else int(corpus.candidate_school[c])))
    return callers


def caller_mask(corpus: Corpus, caller: Caller) -> np.ndarray:
    """
    The set of chunks the caller may see, evaluated in plain numpy so it is
    independent of Qdrant's filter engine. Mirrors the ACL in
    `corpus.retriever._build_acl_filter`: an empty school grants no school tier.
    """
    allowed = corpus.tier == 0
    if caller.school >= 0:
        allowed = allowed | ((corpus.tier == 1) & (corpus.school == caller.school))
    allowed = allowed | ((corpus.tier == 2) & (corpus.candidate == caller.candidate))
    wanted: np.ndarray = allowed & (corpus.test_type == TEST_TYPES.index(caller.test_type))
    return wanted


def caller_filter(caller: Caller) -> models.Filter:
    return build_search_filter(
        candidate_name(caller.candidate),
        school_name(caller.school) if caller.school >= 0 else "",
        caller.test_type,
    )


def exact_top_k(corpus: Corpus, mask: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    sims = corpus.vectors @ query
    sims[~mask] = -np.inf
    return {int(j) for j in np.argpartition(-sims, k)[:k]}


# ------------------------------------------------------------------- measuring
@dataclass
class Measurement:
    variant: str
    profile: str
    points: int
    indexed_fields: list[str]
    queries: int
    matched_share_median: float
    matched_share_min: float
    recall_at_k: float
    recall_min: float
    rows_returned_mean: float
    p50_ms: float
    p95_ms: float
    build_s: float
    note: str = ""


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def measure(
    client: QdrantClient,
    corpus: Corpus,
    callers: list[Caller],
    queries: np.ndarray,
    *,
    collection: str,
    k: int,
    search_params: models.SearchParams | None,
) -> dict[str, Any]:
    recalls: list[float] = []
    rows: list[int] = []
    latencies: list[float] = []
    shares: list[float] = []
    for caller, query in zip(callers, queries):
        mask = caller_mask(corpus, caller)
        truth = exact_top_k(corpus, mask, query, k)
        started = time.perf_counter()
        response = client.query_points(
            collection_name=collection,
            query=query.tolist(),
            using=DENSE_VECTOR,
            query_filter=caller_filter(caller),
            limit=k,
            search_params=search_params,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        got = {int(p.id) for p in response.points}
        recalls.append(len(got & truth) / len(truth))
        rows.append(len(got))
        shares.append(float(mask.mean()))
    return {
        "queries": len(callers),
        "matched_share_median": statistics.median(shares),
        "matched_share_min": min(shares),
        "recall_at_k": float(np.mean(recalls)),
        "recall_min": min(recalls),
        "rows_returned_mean": float(np.mean(rows)),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
    }


# --------------------------------------------------------------------- qdrant
def start_qdrant() -> tuple[str, int]:
    """Start a throwaway container on a free port; return (name, port)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    name = f"edu-agent-filter-check-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker", "run", "-d", "--rm", "--name", name,
            "-p", f"127.0.0.1:{port}:6333", QDRANT_IMAGE,
        ],
        check=True,
        capture_output=True,
    )
    atexit.register(stop_qdrant, name)
    client = QdrantClient(host="127.0.0.1", port=port, prefer_grpc=False)
    deadline = time.time() + 60
    while True:
        try:
            client.get_collections()
            break
        except Exception:
            if time.time() > deadline:
                raise RuntimeError("qdrant did not become ready within 60s") from None
            time.sleep(0.5)
    client.close()
    return name, port


def stop_qdrant(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def wait_green(client: QdrantClient, collection: str, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while client.get_collection(collection).status != models.CollectionStatus.GREEN:
        if time.time() > deadline:
            raise RuntimeError(f"collection {collection} not green after {timeout}s")
        time.sleep(0.5)


def indexed_fields(client: QdrantClient, collection: str) -> list[str]:
    return sorted(client.get_collection(collection).payload_schema or {})


async def _ensure(url: str, collection: str, dim: int) -> None:
    client = AsyncQdrantClient(url=url)
    try:
        await ensure_collection(client, collection=collection, dim=dim)
    finally:
        await client.close()


def drop_payload_indexes(client: QdrantClient, collection: str) -> None:
    for name in indexed_fields(client, collection):
        client.delete_payload_index(collection, name)


def upload(client: QdrantClient, corpus: Corpus, collection: str) -> None:
    client.upload_collection(
        collection_name=collection,
        vectors={DENSE_VECTOR: corpus.vectors},
        payload=corpus.payloads(),
        ids=range(len(corpus.vectors)),
        batch_size=256,
        parallel=1,
    )


VARIANTS = {
    "no-index": "collection as it was created before the fix: payload indexes removed",
    "ensure-then-load": "ensure_collection first (creates the indexes), then load the points",
    "load-then-ensure": "load points unindexed, then ensure_collection (retrofit of an old collection)",
}


def run_variant(
    variant: str,
    corpus: Corpus,
    callers: list[Caller],
    queries: np.ndarray,
    *,
    profile: str,
    url: str,
    host: str,
    port: int,
    dim: int,
    k: int,
    efs: list[int | None],
) -> list[Measurement]:
    """Build one collection the way `variant` says, then measure it at each search ef."""
    client = QdrantClient(host=host, port=port, prefer_grpc=False, timeout=300)
    collection = f"{COLLECTION}_{variant.replace('-', '_')}"
    try:
        started = time.perf_counter()
        asyncio.run(_ensure(url, collection, dim))
        if variant == "ensure-then-load":
            upload(client, corpus, collection)
        else:
            drop_payload_indexes(client, collection)
            upload(client, corpus, collection)
            if variant == "load-then-ensure":
                asyncio.run(_ensure(url, collection, dim))
        wait_green(client, collection)
        build_s = time.perf_counter() - started
        fields = indexed_fields(client, collection)
        out = []
        for ef in efs:
            params = models.SearchParams(hnsw_ef=ef) if ef else None
            stats = measure(
                client, corpus, callers, queries, collection=collection, k=k, search_params=params
            )
            out.append(
                Measurement(
                    variant=variant,
                    profile=profile,
                    points=len(corpus.vectors),
                    indexed_fields=fields,
                    build_s=round(build_s, 1),
                    note=f"hnsw_ef={ef}" if ef else "default hnsw_ef",
                    **stats,
                )
            )
        return out
    finally:
        client.delete_collection(collection)
        client.close()


# --------------------------------------------------------------------- report
def format_table(rows: list[Measurement]) -> str:
    head = (
        "| profile | variant | indexed fields | filter matches (median / min) "
        "| recall@k mean / min | rows returned | p50 ms | p95 ms |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {m.profile} | {m.variant} ({m.note}) | {', '.join(m.indexed_fields) or 'none'} "
        f"| {m.matched_share_median:.1%} / {m.matched_share_min:.1%} "
        f"| {m.recall_at_k:.3f} / {m.recall_min:.2f} | {m.rows_returned_mean:.2f} "
        f"| {m.p50_ms:.2f} | {m.p95_ms:.2f} |"
        for m in rows
    ]
    return head + "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--points", type=int, default=100_000)
    ap.add_argument(
        "--dim", type=int, default=EMBED_DIM, help="vector size (default: production embedder's)"
    )
    ap.add_argument("--queries", type=int, default=200, help="callers to replay per profile")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument(
        "--ef",
        nargs="+",
        type=int,
        default=[0, PROBE_EF],
        help="search-time hnsw_ef values to measure; 0 = Qdrant's default (the build ef_construct)",
    )
    ap.add_argument("--json-out", help="also write the measurements as JSON to this path")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    efs: list[int | None] = [e or None for e in args.ef]
    name, port = start_qdrant()
    url = f"http://127.0.0.1:{port}"
    print(f"{QDRANT_IMAGE} on port {port} (container {name})", file=sys.stderr)
    results: list[Measurement] = []
    try:
        for profile in args.profiles:
            corpus = make_corpus(args.points, args.dim, profile, args.seed)
            callers = sample_callers(corpus, args.queries, args.seed)
            queries = sample_vectors(
                np.random.default_rng(args.seed + 2), corpus.centers, len(callers)
            )
            for variant in args.variants:
                print(f"  {profile} / {variant} ...", file=sys.stderr, flush=True)
                results += run_variant(
                    variant, corpus, callers, queries, profile=profile, url=url,
                    host="127.0.0.1", port=port, dim=args.dim, k=args.k, efs=efs,
                )
    finally:
        stop_qdrant(name)
    print(format_table(results))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([asdict(m) for m in results], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
