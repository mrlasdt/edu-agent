from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def turn_log(tmp_path):
    """
    Turns write their measurement record to a log; point it at a temp file so tests never
    touch the real `.turn-log/`. Tests that assert on records use the returned TurnLog.
    """
    from shared.src.shared.turn_meter import TurnLog

    log = TurnLog(tmp_path / "turn-log" / "turns.jsonl")
    with patch("shared.src.shared.turn_meter.default_turn_log", return_value=log):
        yield log


@pytest.fixture(autouse=True)
def no_live_corpus_service():
    """
    Agents build a RetrieveCorpusSkill by default; keep tests off a corpus service that
    may be running on localhost (e.g. the docker compose stack). Tests that exercise
    retrieval inject their own skill.
    """
    with patch(
        "services.agent.src.agent.skills.retrieve_corpus.RetrieveCorpusSkill._call_corpus",
        new=AsyncMock(return_value=[]),
    ):
        yield
