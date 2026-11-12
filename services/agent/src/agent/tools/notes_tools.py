"""
notes_search - the tool that lets the tutor decide when to look something up.

In the up-front design every tutoring turn searches the corpus with the student's whole message
and puts the top notes in front of the model, whether or not the turn needs one. Here the model
asks: it writes a short concept ("factoring a quadratic"), the corpus answers with numbered
notes, and a turn that needs nothing new (a follow-up, a "thanks", a restated attempt) searches
nothing and carries no notes.

The numbers matter. A reply cites a note as [N], and the gateway strips any [N] that was not
retrieved for the turn, so the notes of all searches in one turn share one numbering: the same
note found twice keeps its number, a new one takes the next. `chunks` is that list, in order,
and its length is the source count the turn reports.

The corpus call is the product's own `RetrieveCorpus` with the relevance floor on, so a concept
the notes do not cover returns an empty list rather than the three nearest wrong notes.
"""
from __future__ import annotations

import logging
from typing import Any

from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus

_log = logging.getLogger(__name__)

TOOL_NAME = "notes_search"
MAX_CONCEPT_CHARS = 120
TOP_K = 3

NOTES_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Look up the app's study notes on one GRE Quant concept, method or formula, "
                "for example `factoring a quadratic`, `average speed`, `percent change` or "
                "`quantitative comparison strategy`. Call it when a hint would be better for a "
                "standard method or fact the student may not have. Do not call it for a follow-up "
                "that needs no new fact (the student answered your question, asked what to do "
                "next, or said thanks), and never to look up the problem itself. "
                "Write the concept in a few words; do not paste the problem or the student's "
                "message. "
                "Returns up to 3 notes as `notes`, each with `n` (cite it as [n] right after "
                "the hint that uses it), `section` and `text`. An empty `notes` means no note "
                "covers that concept: give a hint without citing anything. "
                "On failure it returns `error` saying what to change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": (
                            f"The concept to look up, at most {MAX_CONCEPT_CHARS} characters. "
                            "Example: `difference of squares`."
                        ),
                    }
                },
                "required": ["concept"],
                "additionalProperties": False,
            },
        },
    }
]


class NotesTools:
    """One turn's notes: runs the searches the model asks for and keeps what they returned."""

    def __init__(
        self, retrieve: RetrieveCorpus, *, candidate_id: str, test_type: str, top_k: int = TOP_K
    ) -> None:
        self._retrieve = retrieve
        self._candidate_id = candidate_id
        self._test_type = test_type
        self._top_k = top_k
        self.chunks: list[CorpusChunk] = []
        self.concepts: list[str] = []

    def _number(self, chunk: CorpusChunk) -> int:
        """The note's number in this turn: an earlier search's number if it was found before."""
        for n, seen in enumerate(self.chunks, start=1):
            if (seen.source_uri, seen.page_or_section, seen.text) == (
                chunk.source_uri, chunk.page_or_section, chunk.text,
            ):
                return n
        self.chunks.append(chunk)
        return len(self.chunks)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Never raises: a corpus that is down becomes an error the model can read."""
        concept = arguments.get("concept")
        if not isinstance(concept, str) or not concept.strip():
            return {"error": "concept is required: a few words naming what to look up, "
                             "e.g. `difference of squares`."}
        concept = concept.strip()
        if len(concept) > MAX_CONCEPT_CHARS:
            return {
                "error": f"concept is {len(concept)} characters; the limit is "
                f"{MAX_CONCEPT_CHARS}. Send the idea in a few words (e.g. `factoring a "
                "quadratic`), not the problem text."
            }
        self.concepts.append(concept)
        try:
            found = await self._retrieve.retrieve(
                query=concept,
                candidate_id=self._candidate_id,
                school_id="",
                test_type=self._test_type,
                top_k=self._top_k,
            )
        except Exception:
            _log.warning("notes_search: the corpus could not be reached", exc_info=True)
            return {
                "error": "the notes could not be reached just now. Give a hint without "
                "them and do not cite a note."
            }
        return {
            "notes": [
                {"n": self._number(c), "section": c.page_or_section, "text": c.text}
                for c in found
            ]
        }
