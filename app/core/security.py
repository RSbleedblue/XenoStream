from fastapi import Header, HTTPException, Query, status

from app.core.settings import get_settings


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


def verify_api_key_or_query(
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(
        default=None,
        description="Same as x-api-key when the client cannot set headers (e.g. browser <audio> / some native players). Prefer the header in production.",
    ),
) -> None:
    """Accept API key from header or query; query is less private (may appear in logs, referrers)."""
    settings = get_settings()
    if not settings.api_key:
        return
    if (x_api_key or api_key) != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
