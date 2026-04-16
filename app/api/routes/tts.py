import logging
import base64

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.dependencies import get_tts_service
from app.api.schemas.tts import SynthesizeRequest, SynthesizeResponse
from app.core.security import verify_api_key
from app.services.tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["tts"])


def _split_text_for_retry(text: str) -> list[str]:
    txt = (text or "").strip()
    if len(txt) <= 1:
        return [txt]
    mid = len(txt) // 2
    split_at = txt.rfind(" ", 0, mid)
    if split_at <= 0:
        split_at = mid
    left = txt[:split_at].strip()
    right = txt[split_at:].strip()
    out = []
    if left:
        out.append(left)
    if right:
        out.append(right)
    return out or [txt]


@router.post("/tts", response_model=SynthesizeResponse, dependencies=[Depends(verify_api_key)])
def synthesize(payload: SynthesizeRequest, tts_service: TTSService = Depends(get_tts_service)) -> SynthesizeResponse:
    try:
        request_id, out_file = tts_service.synthesize_resilient(payload)
        return SynthesizeResponse(request_id=request_id, output_file=str(out_file), sample_rate=24000)
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except AssertionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled TTS failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal synthesis failure",
        ) from exc


@router.post("/tts/file", dependencies=[Depends(verify_api_key)])
def synthesize_file(payload: SynthesizeRequest, tts_service: TTSService = Depends(get_tts_service)) -> FileResponse:
    try:
        _, out_file = tts_service.synthesize_resilient(payload)
        return FileResponse(path=out_file, media_type="audio/wav", filename=out_file.name)
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except AssertionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled TTS failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal synthesis failure",
        ) from exc


@router.websocket("/tts/ws")
async def stream_tts_socket(
    websocket: WebSocket,
    api_key: str | None = Query(default=None),
) -> None:
    settings = websocket.app.state.settings
    tts_service: TTSService = websocket.app.state.tts_service

    if settings.api_key and api_key != settings.api_key:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    await websocket.send_json(
        {
            "type": "ready",
            "message": "send {'type':'text','text':'...'} or optional {'voice_id','language','speed'}",
            "audio_format": "pcm16_base64",
            "sample_rate": 24000,
        }
    )

    session = {
        "voice_id": None,
        "language": "en",
        "speed": 1.0,
    }
    seq = 0

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = (msg.get("type") or "text").lower()

            if msg_type == "config":
                session["voice_id"] = msg.get("voice_id")
                session["language"] = msg.get("language") or session["language"]
                session["speed"] = float(msg.get("speed") or session["speed"])
                await websocket.send_json({"type": "config_ack", "session": session})
                continue

            if msg_type != "text":
                await websocket.send_json({"type": "error", "detail": "Unsupported message type"})
                continue

            text = (msg.get("text") or "").strip()
            if not text:
                await websocket.send_json({"type": "error", "detail": "text is required"})
                continue

            language = msg.get("language") or session["language"]
            speed = float(msg.get("speed") or session["speed"])
            voice_id = msg.get("voice_id") if "voice_id" in msg else session["voice_id"]

            try:
                resolved_speaker_wav_path = tts_service.prepare_stream_voice(
                    voice_id=voice_id,
                )
            except FileNotFoundError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc), "code": 404})
                continue
            except ValueError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc), "code": 422})
                continue

            chunks = tts_service.chunk_text_for_stream(text)
            if not chunks:
                await websocket.send_json({"type": "error", "detail": "text is empty"})
                continue

            pending_chunks = list(chunks)
            current_idx = 0
            while pending_chunks:
                chunk = pending_chunks.pop(0)
                try:
                    audio_bytes = await run_in_threadpool(
                        tts_service.synthesize_chunk_pcm16_prepared,
                        text=chunk,
                        language=language,
                        speed=speed,
                        speaker_wav_path=resolved_speaker_wav_path,
                    )
                except TimeoutError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc), "code": 429})
                    break
                except FileNotFoundError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc), "code": 404})
                    break
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc), "code": 422})
                    break
                except AssertionError as exc:
                    detail = str(exc)
                    # XTTS hard limit: 400 GPT text tokens. Split and retry progressively.
                    if "maximum of 400 tokens" in detail and len(chunk) > 8:
                        split_chunks = _split_text_for_retry(chunk)
                        if len(split_chunks) > 1:
                            pending_chunks = split_chunks + pending_chunks
                            continue
                    await websocket.send_json({"type": "error", "detail": detail, "code": 422})
                    break
                except Exception:
                    logger.exception("Unhandled socket TTS failure")
                    await websocket.send_json(
                        {"type": "error", "detail": "Internal synthesis failure", "code": 500}
                    )
                    break

                await websocket.send_json(
                    {
                        "type": "audio_chunk",
                        "seq": seq,
                        "chunk_index": current_idx,
                        "total_chunks": max(len(chunks), 1),
                        "text": chunk,
                        "format": "pcm16_base64",
                        "sample_rate": 24000,
                        "audio": base64.b64encode(audio_bytes).decode("ascii"),
                    }
                )
                seq += 1
                current_idx += 1

            await websocket.send_json({"type": "done", "seq": seq})

    except WebSocketDisconnect:
        return
