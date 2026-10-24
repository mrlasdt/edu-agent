"""
Importer for the candidate-submitted IELTS Writing Task 2 essays on IELTS Buddy
(ieltsbuddy.com "essay samples" pages for bands 4 to 8).

Each sample is a real candidate essay, a stated band in its heading ("Topic: X (Band 6.5)") and
per-criterion bands with a comment from "an experienced IELTS instructor". That is better than
a site's automatic score but it is not an official examiner, so the tier is `teacher_graded`.

Polite by construction: robots.txt is fetched and honoured, one fixed User-Agent that names this
tool, a pause between requests, every page cached in the gitignored local directory (re-runs make
no requests), public pages only, no login. Downloaded pages are untrusted data: they are only
parsed (stdlib HTML parser, no scripts run) and never executed. Everything lands in the
gitignored local directory; nothing from the site is committed.
"""
from __future__ import annotations

import hashlib
import html
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from evals.exam_set.local_store import SourceRecord, register_source
from evals.exam_set.schema import (
    CRITERIA,
    TIER_CAVEATS,
    Design,
    ExamItem,
    LabelProvenance,
    Labels,
    LabelTier,
    Source,
)

SOURCE_ID = "ieltsbuddy_essays"
FILE_NAME = "ielts_ieltsbuddy.jsonl"
CACHE_DIR = "ieltsbuddy_cache"
BASE = "https://www.ieltsbuddy.com"
PAGES = (
    "ielts-band-4-essay-samples.html",
    "ielts-band-5-essay-samples.html",
    "ielts-band-6-essay-samples.html",
    "ielts-band-7-essay-samples.html",
    "ielts-band-8-essay-samples.html",
    "ielts-writing-task-2-band-6-examples.html",
)
USER_AGENT = "edu-agent-golden-importer (personal research; one fetch per page, cached)"
LICENCE_STATUS = (
    "copyrighted (ieltsbuddy.com candidate submissions and instructor comments); public pages "
    "fetched politely for personal local use only, never committed"
)
DEFAULT_PAUSE_S = 3.0
FETCH_TIMEOUT_S = 30.0

_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "section",
               "article", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"script", "style", "noscript", "template"}
_TOPIC = re.compile(r"^Topic:\s*(?P<title>.*?)\s*(?:\(\s*band\s*(?P<band>\d(?:\.[05])?)\s*\))?\s*$",
                    re.I)
_BAND_ONLY = re.compile(r"^\(?\s*band\s*(?P<band>\d(?:\.[05])?)\s*\)?$", re.I)
_CRITERION = re.compile(
    r"^(?P<name>task\s+response|coherence\s*(?:&|and)\s*cohesion|lexical\s+resource|"
    r"grammatical\s+range\s*(?:&|and)\s*accuracy)\s*:\s*(?P<band>\d(?:\.[05])?)\s*$",
    re.I,
)
_CRITERION_KEY = {"task": "TR", "coherence": "CC", "lexical": "LR", "grammatical": "GRA"}
# Lines that are part of the question, as opposed to the essay that follows it.
_QUESTION = re.compile(
    r"\b(discuss|to what extent|do you agree|agree or disagree|what (?:are|is|do|does|problems|"
    r"factors|solutions|measures|causes|reasons)|how (?:can|could|realistic|should)|why (?:do|is|"
    r"are)|outweigh|give your|your own opinion|your opinion)\b",
    re.I,
)
MAX_QUESTION_LINE = 160
# Where the page's own furniture (related-links, share bar, comment form) starts after the last
# criterion comment.
_FOOTER = re.compile(r"^(?:comments$|you might like|any comments or questions|join ielts buddy|"
                     r"share$|related|see also|more band)", re.I)


class _TextExtractor(HTMLParser):
    """Visible text with one line per block element."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def page_lines(page_html: str) -> list[str]:
    parser = _TextExtractor()
    parser.feed(page_html)
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    return [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n") if ln.strip()]


def round_overall(criteria: dict[str, float]) -> float:
    """Mean of the four criteria rounded to the nearest half band (a quarter rounds up)."""
    mean = sum(criteria[c] for c in CRITERIA) / len(CRITERIA)
    return float(int(mean * 2 + 0.5)) / 2


@dataclass
class ParsedEssay:
    title: str
    stated_band: float | None
    prompt: str
    essay: str
    criteria: dict[str, float]
    comments: dict[str, str] = field(default_factory=dict)


def _split_prompt(body: list[str]) -> tuple[list[str], list[str]]:
    """The question is the first line plus any short question-like lines that follow it."""
    if not body:
        return [], []
    end = 1
    while end < len(body) and len(body[end]) <= MAX_QUESTION_LINE and _QUESTION.search(body[end]):
        end += 1
    return body[:end], body[end:]


def _half_band(text: str) -> float | None:
    value = float(text)
    return value if 0.0 <= value <= 9.0 and (value * 2) % 1 == 0 else None


def parse_page(page_html: str) -> list[ParsedEssay]:
    """Every complete essay (prompt, essay, four criterion bands) on one page."""
    lines = page_lines(page_html)
    starts = [i for i, ln in enumerate(lines) if _TOPIC.match(ln)]
    out: list[ParsedEssay] = []
    for a, b in zip(starts, [*starts[1:], len(lines)], strict=True):
        block = lines[a:b]
        head = _TOPIC.match(block[0])
        assert head is not None
        stated = _half_band(head.group("band")) if head.group("band") else None
        body = block[1:]
        if body and (m := _BAND_ONLY.match(body[0])):
            stated, body = _half_band(m.group("band")), body[1:]
        if "Comments" not in body:
            continue
        cut = body.index("Comments")
        prompt_lines, essay_lines = _split_prompt(body[:cut])
        tail = body[cut + 1:]
        stop = next((k for k, ln in enumerate(tail) if _FOOTER.match(ln)), len(tail))
        tail = tail[:stop]
        criteria: dict[str, float] = {}
        comments: dict[str, list[str]] = {}
        current: str | None = None
        for ln in tail:
            if (m := _CRITERION.match(ln)) is not None:
                key = _CRITERION_KEY[m.group("name").split()[0].lower()]
                band = _half_band(m.group("band"))
                if band is not None:
                    criteria[key] = band
                current = key
                comments.setdefault(key, [])
            elif current is not None:
                comments[current].append(ln)
        if len(criteria) != len(CRITERIA) or not prompt_lines or not essay_lines:
            continue
        out.append(
            ParsedEssay(
                title=head.group("title"),
                stated_band=stated,
                prompt="\n".join(prompt_lines),
                essay="\n\n".join(essay_lines),
                criteria=criteria,
                comments={k: " ".join(v).strip() for k, v in comments.items()},
            )
        )
    return out


def to_item(parsed: ParsedEssay, page: str, n: int) -> ExamItem:
    computed = round_overall(parsed.criteria)
    digest = hashlib.sha256(parsed.essay.encode("utf-8")).hexdigest()[:8]
    comment = "\n".join(f"{k}: {parsed.comments[k]}" for k in CRITERIA if parsed.comments.get(k))
    return ExamItem(
        id=f"ib_{digest}",
        exam="ielts",
        skill="writing_t2",
        task_type="essay_score",
        prompt=parsed.prompt,
        response=parsed.essay,
        labels=Labels(
            overall=parsed.stated_band if parsed.stated_band is not None else computed,
            criteria=parsed.criteria,
        ),
        label_provenance=LabelProvenance(
            tier=LabelTier.teacher_graded,
            n_raters=1,
            note=TIER_CAVEATS[LabelTier.teacher_graded],
        ),
        source=Source(
            kind="third_party_local",
            license="copyrighted; fetched for personal local use",
            source_id=SOURCE_ID,
            url=f"{BASE}/{page}",
            attribution="IELTS Buddy (ieltsbuddy.com) instructor-commented candidate essay",
        ),
        design=Design(generator="scraped"),
        split="calibration",
        examiner_comment=comment or None,
        extras={
            "topic": parsed.title,
            "page": page,
            "n_on_page": n,
            "stated_overall": parsed.stated_band,
            "computed_overall": computed,
        },
    )


def make_client() -> httpx.Client:
    return httpx.Client(
        timeout=FETCH_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )


def fetch_robots(client: httpx.Client) -> Callable[[str], bool]:
    """Fetch robots.txt once and return its verdict per URL; unreachable or missing allows."""
    parser = RobotFileParser()
    try:
        resp = client.get(f"{BASE}/robots.txt")
    except httpx.HTTPError:
        return lambda url: True
    if resp.status_code in (401, 403):
        return lambda url: False
    if resp.status_code >= 400:
        return lambda url: True
    parser.parse(resp.text.splitlines())
    return lambda url: parser.can_fetch(USER_AGENT, url)


@dataclass
class ImportStats:
    pages_fetched: int = 0
    pages_cached: int = 0
    pages_disallowed: int = 0
    pages_failed: int = 0
    essays_parsed: int = 0
    skipped_duplicate: int = 0


def load_page(
    client: httpx.Client, page: str, cache: Path, stats: ImportStats,
    allowed: Callable[[str], bool], sleep: Callable[[float], None], pause_s: float,
) -> str | None:
    cached = cache / page
    if cached.exists():
        stats.pages_cached += 1
        return cached.read_text(encoding="utf-8", errors="replace")
    url = f"{BASE}/{page}"
    if not allowed(url):
        stats.pages_disallowed += 1
        return None
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError:
        stats.pages_failed += 1
        return None
    cache.mkdir(parents=True, exist_ok=True)
    cached.write_text(resp.text, encoding="utf-8")
    stats.pages_fetched += 1
    sleep(pause_s)
    return resp.text


def import_ieltsbuddy(
    directory: Path,
    client: httpx.Client,
    *,
    pages: tuple[str, ...] = PAGES,
    pause_s: float = DEFAULT_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[SourceRecord, ImportStats]:
    stats = ImportStats()
    cache = directory / CACHE_DIR
    items: list[ExamItem] = []
    seen: set[str] = set()
    allowed: Callable[[str], bool] | None = None
    for page in pages:
        if allowed is None and not (cache / page).exists():
            allowed = fetch_robots(client)  # only when a page really needs fetching
        body = load_page(client, page, cache, stats, allowed or (lambda url: True), sleep, pause_s)
        if body is None:
            continue
        for n, parsed in enumerate(parse_page(body), start=1):
            stats.essays_parsed += 1
            item = to_item(parsed, page, n)
            if item.id in seen:
                stats.skipped_duplicate += 1
                continue
            seen.add(item.id)
            items.append(item)
    record = register_source(
        directory,
        source_id=SOURCE_ID,
        name="IELTS Buddy candidate essays with instructor comments",
        url=f"{BASE}/ielts-sample-essays.html",
        licence_status=LICENCE_STATUS,
        items=items,
        file_name=FILE_NAME,
        caveat=TIER_CAVEATS[LabelTier.teacher_graded],
        notes=(
            f"{len(items)} essays from {stats.pages_fetched + stats.pages_cached} pages "
            f"({stats.pages_fetched} fetched, {stats.pages_cached} cached, "
            f"{stats.pages_disallowed} disallowed by robots.txt, {stats.pages_failed} failed); "
            "overall is the band stated in the heading, criterion bands are the instructor's"
        ),
    )
    return record, stats
