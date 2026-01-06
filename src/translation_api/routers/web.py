"""Web UI endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from ..api.dependencies import router_limiter

router = APIRouter(prefix="", tags=["Web UI"])


@router.get("/", dependencies=[Depends(router_limiter)])
async def root():
    """Access the built-in web interface for testing translations."""
    return RedirectResponse(url="/static/index.html")


@router.get("/ui")
async def web_ui():
    """Access the built-in web interface for testing translations."""
    return RedirectResponse(url="/static/index.html")


@router.get("/compare")
async def compare_ui():
    """Access the translation comparison UI."""
    return RedirectResponse(url="/static/compare.html")
