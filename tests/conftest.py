from unittest.mock import AsyncMock, patch

import pytest


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
