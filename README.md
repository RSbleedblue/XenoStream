# XTTS FastAPI (Production-Oriented Self-Hosted Setup)

This project provides a self-hosted XTTS (Coqui) service with FastAPI and Gunicorn.

## Architecture

Code is organized by responsibility:

- `app/main.py`: thin entrypoint (`app = create_app()`).
- `app/app_factory.py`: app wiring, lifespan, router registration.
- `app/core/`: environment config and auth dependency.
- `app/services/`: XTTS inference engine, voice profile storage, orchestration logic.
- `app/api/routes/`: FastAPI route handlers by domain (health, voices, tts).
- `app/api/schemas/`: request/response contracts.

## What You Get

- FastAPI app with `/healthz`, `/readyz`, `/v1/tts`, `/v1/tts/file`
- Voice profile API with `/v1/voices` for upload/list/delete and `voice_id` reuse
- Real-time socket streaming via `/v1/tts/ws` for chunked call-style responses
- Optional API key auth via `x-api-key`
- Concurrency control to protect GPU/CPU under load
- Dockerized runtime with Gunicorn + Uvicorn workers
- Environment-based configuration for production deployment

## 1) Quick Start (Local Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/healthz
```

## 2) Quick Start (Docker)

```bash
cp .env.example .env
docker compose up --build -d
```

Check service:

```bash
curl http://localhost:8000/readyz
```

## 3) Configure For Production

Update `.env`:

- `API_KEY`: set a strong secret
- `DEVICE`: `cuda` on GPU hosts, `cpu` otherwise
- `COQUI_TOS_AGREED=1`: required for non-interactive startup after you accept Coqui CPML/commercial terms
- `MAX_CONCURRENT_REQUESTS`: tune to your machine
- `MAX_TEXT_CHARS`: protect against oversized requests
- `PRELOAD_MODEL=true`: warm model on boot to reduce first request latency
- `DEFAULT_SPEAKER_WAV_PATH=/absolute/path/to/reference.wav`: optional, use one default cloned voice for continuous requests

For public deployments, put Nginx/Traefik in front with TLS and request-size limits.

## 4) API Usage

### Generate metadata JSON

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "text": "Hello from XTTS.",
    "language": "en",
    "voice_id": "<voice_id_from_upload>",
    "speed": 1.0
  }'
```

### Generate and download WAV

```bash
curl -X POST http://localhost:8000/v1/tts/file \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "text": "This is a generated audio file.",
    "language": "en",
    "speaker_wav_path": "/absolute/path/to/reference.wav",
    "speed": 1.0
  }' \
  --output sample.wav
```

If you want call-like continuous TTS with a fixed cloned voice, set `DEFAULT_SPEAKER_WAV_PATH` once in `.env`, then omit `speaker_wav_path` in each request.

### Production Voice Pattern (Recommended)

1. Upload voice once and save the returned `voice_id` in your app/database:

```bash
curl -X POST http://localhost:8000/v1/voices \
  -H "x-api-key: change-me" \
  -F "file=@/absolute/path/to/reference.wav"
```

2. Reuse `voice_id` for every low-latency text-to-speech request:

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "text": "Hello from continuous voice mode.",
    "language": "en",
    "voice_id": "<voice_id>",
    "speed": 1.0
  }'
```

3. Manage profiles:

```bash
curl -H "x-api-key: change-me" http://localhost:8000/v1/voices
curl -X DELETE -H "x-api-key: change-me" http://localhost:8000/v1/voices/<voice_id>
```

### Real-Time Call Mode (WebSocket)

Use this for continuous call-like interactions where text arrives over time and audio is returned chunk-by-chunk:

- Endpoint: `ws://localhost:8000/v1/tts/ws?api_key=change-me`
- Client sends:
  - `{"type":"config","voice_id":"<voice_id>","language":"en","speed":1.0}`
  - `{"type":"text","text":"hello this is chunked in near real-time"}`
- Server returns multiple `audio_chunk` messages with Base64 WAV payloads:
  - `{"type":"audio_chunk","seq":0,"chunk_index":0,"total_chunks":N,"format":"pcm16_base64","sample_rate":24000,"audio":"..."}`
  - `{"type":"done","seq":...}` when message processing completes.

This mode avoids waiting for full long-form synthesis and starts sending playable chunks earlier. PCM16 chunks are lower-overhead than WAV chunks for better streaming latency.

## 5) Production Notes

- Keep `workers=1` for a single loaded XTTS model instance per container (GPU memory safety).
- Scale horizontally with multiple containers behind a reverse proxy/load balancer.
- Persist `outputs/` and model cache volume for predictable startup.
- Monitor latency and adjust `MAX_CONCURRENT_REQUESTS` conservatively.
- Restrict network access and keep API key secret.

## 6) Run with Gunicorn (non-Docker)

```bash
gunicorn -c gunicorn.conf.py app.main:app
```

## 7) Common Issues

- Model load is slow on first run: model files are being downloaded.
- `503 /readyz`: model is not loaded yet.
- CUDA not used: verify PyTorch CUDA build and set `DEVICE=cuda`.
