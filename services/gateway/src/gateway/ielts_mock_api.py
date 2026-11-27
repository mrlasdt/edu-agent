"""
HTTP API for the full IELTS mock (`/ielts/mock/...`, JSON).

A mock is a parent record over the Reading and Writing attempts. This API only creates the
parent, creates each module's attempt in order and reports the combined result; the module itself
is sat through the existing module APIs (`/ielts/reading/attempts/...`,
`/ielts/writing/attempts/...`) with the attempt id the mock reports. The overall band is worked
out in code (see `ielts/mock.py`).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from services.gateway.src.gateway.ielts.mock import (
    LoadedMock,
    MockReport,
    build_report,
    can_start,
    load_mock,
)
from services.gateway.src.gateway.ielts_api import _limit, get_ledger, get_store
from services.gateway.src.gateway.ielts_reading_api import create_attempt as create_reading
from services.gateway.src.gateway.ielts_store import AttemptStore, MockNotFoundError
from services.gateway.src.gateway.ielts_writing_api import (
    CreateModuleAttempt,
)
from services.gateway.src.gateway.ielts_writing_api import (
    create_attempt as create_writing,
)
from shared.src.shared.budget import Ledger

router = APIRouter(prefix="/ielts/mock", tags=["ielts-mock"])

Store = Annotated[AttemptStore, Depends(get_store)]
SpendLedger = Annotated[Ledger, Depends(get_ledger)]


def _loaded(store: AttemptStore, mock_id: str) -> LoadedMock:
    try:
        return load_mock(store, store.get_mock(mock_id))
    except MockNotFoundError:
        raise HTTPException(404, "mock not found") from None


@router.post("", status_code=201)
def create_mock(store: Store) -> MockReport:
    _limit()
    return build_report(load_mock(store, store.create_mock()))


@router.get("")
def list_mocks(store: Store) -> list[MockReport]:
    return [build_report(load_mock(store, m)) for m in store.recent_mocks()]


@router.get("/{mock_id}")
def get_mock(mock_id: str, store: Store) -> MockReport:
    return build_report(_loaded(store, mock_id))


@router.post("/{mock_id}/modules/{key}/attempt", status_code=201)
def start_module(mock_id: str, key: str, store: Store, ledger: SpendLedger) -> MockReport:
    """Create the module's attempt (the same one its own page makes) and link it to the mock."""
    loaded = _loaded(store, mock_id)
    why_not = can_start(loaded, key)
    if why_not:
        raise HTTPException(409, why_not)
    attempt = (
        create_reading(store) if key == "reading"
        else create_writing(CreateModuleAttempt(), store, ledger)
    )
    store.link_mock_attempt(mock_id, key, attempt.id)
    return build_report(_loaded(store, mock_id))
