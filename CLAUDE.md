# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XenoStream is a self-hosted Text-to-Speech (TTS) API service built with FastAPI and Gunicorn, wrapping the Coqui XTTS v2 model. It provides REST and WebSocket APIs for voice synthesis with voice profile management and streaming capabilities.

## Development Commands

```bash
# Setup (uses uv package manager)
uv pip install -r requirements.txt
cp .env.example .env

# Run locally (development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run locally (production mode)
gunicorn -c gunicorn.conf.py app.main:app

# Docker
docker compose up --build -d
docker compose logs -f xtts-api

# Test WebSocket TTS interactively
python scripts/test_ws_tts.py --api-key <key> [--voice-id <id>]

# Verify service readiness
curl http://localhost:8000/readyz
```

There is no test suite or linter configured yet.

## Architecture

**Layered design:** API routes → Service layer → Engine → Coqui XTTS model

- **Entry point:** `app/main.py` → `app/app_factory.py` (`create_app()` factory wires all dependencies)
- **API layer** (`app/api/routes/`): FastAPI routers for health, TTS, and voice management
- **Service layer** (`app/services/`):
  - `engine.py` — `XTTSEngine`: model loading, inference, semaphore-based concurrency control
  - `tts_service.py` — `TTSService`: orchestration, text validation, resilient splitting for XTTS 400-token limit, streaming chunk preparation
  - `voice_store.py` — `VoiceStore`: file-based voice profile CRUD with UUID-based IDs
- **Core** (`app/core/`): `settings.py` (Pydantic Settings from `.env`), `security.py` (API key verification)
- **Schemas** (`app/api/schemas/`): Pydantic request/response models

**Key design decisions:**
- Single Gunicorn worker per container (GPU memory safety); scale horizontally with multiple containers
- `BoundedSemaphore` in `XTTSEngine` limits concurrent inference — returns 429 when full
- `synthesize_resilient()` auto-splits text by sentences when XTTS hits its 400 GPT-token assertion, then concatenates audio via NumPy
- Model loading is lazy (on first request) unless `PRELOAD_MODEL=true`
- API key auth via `x-api-key` header (disabled when `API_KEY` env var is unset)

## API Endpoints

- `GET /healthz` / `GET /readyz` — health and model readiness
- `POST /v1/tts` — synthesize text, returns JSON with audio file path
- `POST /v1/tts/file` — synthesize text, returns WAV file directly
- `GET /v1/speakers` — list pre-trained speakers from XTTS
- `WebSocket /v1/tts/ws?api_key=<key>` — real-time streaming (send text, receive Base64 PCM16 chunks)
- `POST /v1/voices` — upload custom voice WAV (multipart)
- `GET /v1/voices` — list stored voice profiles
- `DELETE /v1/voices/{voice_id}` — remove a voice profile

## Configuration

All config is via environment variables (see `.env.example`). Key settings: `API_KEY`, `DEVICE` (cpu/cuda), `MODEL_NAME`, `MAX_TEXT_CHARS`, `STREAM_CHUNK_CHARS`, `MAX_CONCURRENT_REQUESTS`, `PRELOAD_MODEL`.

## Docker

The Dockerfile uses `python:3.11-slim`, installs Rust toolchain for tokenizers, ffmpeg, and libsndfile. Runs as non-root `appuser`. Model cache is volume-mounted at `/home/appuser/.local/share/tts`.
