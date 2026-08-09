#!/bin/sh
set -e

echo "Running database migrations..."
/app/.venv/bin/python scripts/provision_db.py

if [ "${HONCHO_CONFIGURE_EMBEDDINGS:-false}" = "true" ]; then
    echo "Configuring pgvector dimensions..."
    /app/.venv/bin/python scripts/configure_embeddings.py --yes --verify-endpoint
fi

echo "Starting API server..."
exec /app/.venv/bin/fastapi run --host 0.0.0.0 src/main.py
