from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/api/Regression", tags=["Regression"])


@router.get("/Test", response_class=PlainTextResponse)
def test() -> str:
    return "ping"
