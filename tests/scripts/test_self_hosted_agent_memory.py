"""Static contract tests for the honcho-* self-hosted memory stack."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compose_matches_agent_memory_contract() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()

    assert '"127.0.0.1:8787:8000"' in compose
    assert "\n  database:" in compose
    assert "\n  redis:" in compose
    assert "\n  embeddings:" in compose
    assert "ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-" in compose
    assert "\n  deriver:" not in compose
    assert "condition: service_healthy" in compose
    assert (
        "HONCHO_EMBEDDING_MODEL_PATH:-./models/"
        "paraphrase-multilingual-MiniLM-L12-v2-tei" in compose
    )
    assert ":/models/embedding:ro" in compose
    assert "ollama" not in compose.lower()


def test_compose_disables_text_generation_and_uses_local_embeddings() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "DERIVER_ENABLED: \"false\"" in compose
    assert "SUMMARY_ENABLED: \"false\"" in compose
    assert "DREAM_ENABLED: \"false\"" in compose
    assert "PEER_CARD_ENABLED: \"false\"" in compose
    assert "EMBEDDING_VECTOR_DIMENSIONS: \"384\"" in compose
    assert "HONCHO_EMBEDDING_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" in compose
    assert "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE: never" in compose
    assert "http://embeddings:80/v1" in compose
    assert "- mean" in compose
    assert "LLM_OPENAI_API_KEY" not in compose
    assert "LLM_ANTHROPIC_API_KEY" not in compose
    assert "LLM_GEMINI_API_KEY" not in compose


def test_entrypoint_probes_endpoint_before_api_start() -> None:
    entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text()

    provision = entrypoint.index("scripts/provision_db.py")
    configure = entrypoint.index(
        "scripts/configure_embeddings.py --yes --verify-endpoint"
    )
    start = entrypoint.index("fastapi run")

    assert provision < configure < start


def test_skill_environment_defaults_match_compose_port() -> None:
    env_template = (ROOT / ".env.template").read_text()

    assert "HONCHO_BASE_URL=http://localhost:8787" in env_template
    assert "HONCHO_WORKSPACE=claude-code" in env_template
    assert "HONCHO_OBSERVER=claude-code" in env_template
    assert "HONCHO_OBSERVED=user" in env_template
    assert (
        "HONCHO_EMBEDDING_MODEL=sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2" in env_template
    )
    assert (
        "HONCHO_EMBEDDING_MODEL_PATH=./models/"
        "paraphrase-multilingual-MiniLM-L12-v2-tei" in env_template
    )
    assert (
        "HONCHO_EMBEDDING_IMAGE=ghcr.io/huggingface/"
        "text-embeddings-inference:" in env_template
    )
