# Database Quickstart

This project uses PostgreSQL for conversation and message storage. The tests in `Test/test_conversation_flow.py` expect a Postgres database named `conversations_db` to be reachable at `localhost:5432`. To keep your development machine clean, the preferred workflow spins up a disposable Docker container and points the tests at it.

## 1. Prerequisites

- Docker Desktop / Docker Engine installed and running.
- Python virtual environment (the project includes `unityvenv/`).
- `psycopg2-binary` installed in your virtual environment (`unityvenv/bin/python -m pip install psycopg2-binary`).

## 2. Start PostgreSQL via Docker

```bash
# remove any stale container and start a fresh PostgreSQL 15 instance
docker rm -f unify-postgres 2>/dev/null || true
docker run --rm -d --name unify-postgres \
    -e POSTGRES_PASSWORD=pass \
    -e POSTGRES_DB=conversations_db \
    -p 5432:5432 \
    postgres:15
```

This exposes the database on `localhost:5432` with username `postgres` and password `pass`.

## 3. Run the pytest conversation test

Make sure the virtual environment is activated (e.g., `source unityvenv/bin/activate`). Then run:

```bash
DATABASE_URL=postgresql://postgres:pass@localhost:5432/conversations_db \
    unityvenv/bin/python -m pytest Test/test_conversation_flow.py
```

The test fixture uses `database_utils.get_connection_from_env()` so it will honor `DATABASE_URL` or the `DB_*` env vars.

## 4. Tear down the database

When you’re done with testing, stop the container:

```bash
docker stop unify-postgres
```

That frees the RAM/CPU the database was using.

## 5. Troubleshooting

- If you see `OperationalError: connection refused`, ensure Docker is running and the container is healthy (`docker ps`).
- If `psycopg2` is missing, install it inside the virtual environment: `unityvenv/bin/python -m pip install psycopg2-binary`.
- Adjust `DATABASE_URL` env var if you change the credentials or want to point at a different Postgres host.
