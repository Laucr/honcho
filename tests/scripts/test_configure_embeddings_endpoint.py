"""Unit tests for the pre-migration embedding endpoint probe."""

import pytest

from scripts.configure_embeddings import (
    _verify_embedding_endpoint,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.asyncio
async def test_verify_embedding_endpoint_accepts_matching_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_embed(_text: str) -> list[float]:
        return [0.0] * 1536

    monkeypatch.setattr("src.embedding_client.embedding_client.embed", fake_embed)
    await _verify_embedding_endpoint()


@pytest.mark.asyncio
async def test_verify_embedding_endpoint_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_embed(_text: str) -> list[float]:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("src.embedding_client.embedding_client.embed", fake_embed)
    with pytest.raises(SystemExit, match="pgvector was not changed: model unavailable"):
        await _verify_embedding_endpoint()
