"""
Mock OpenAI-compatible upstream for the free-threading benchmark.

Exposes /v1/chat/completions with a configurable streaming response so the
benchmark measures litellm's own CPU + scheduling cost, not the upstream
provider's latency.

Run standalone:
    python -m tests.load_tests.free_threading_benchmark.mock_upstream \
        --port 18080 --tokens 64 --inter-token-ms 8 --first-token-ms 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import AsyncIterator

from aiohttp import web


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


async def _stream_chunks(
    *,
    model: str,
    tokens: int,
    first_token_ms: int,
    inter_token_ms: int,
) -> AsyncIterator[bytes]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }

    # role chunk
    yield _sse(
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )

    await asyncio.sleep(first_token_ms / 1000.0)

    for i in range(tokens):
        if i > 0:
            await asyncio.sleep(inter_token_ms / 1000.0)
        yield _sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"tok{i} "},
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield _sse(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield b"data: [DONE]\n\n"


def make_app(args: argparse.Namespace) -> web.Application:
    async def chat_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        stream = bool(body.get("stream"))
        model = body.get("model", "gpt-4o-mini")

        if not stream:
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
            return web.json_response(response)

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        async for chunk in _stream_chunks(
            model=model,
            tokens=args.tokens,
            first_token_ms=args.first_token_ms,
            inter_token_ms=args.inter_token_ms,
        ):
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": "gpt-4o-mini", "object": "model", "owned_by": "mock"},
                ],
            }
        )

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--tokens",
        type=int,
        default=64,
        help="number of content chunks per streamed response",
    )
    parser.add_argument(
        "--first-token-ms",
        type=int,
        default=40,
        help="simulated time-to-first-token at the upstream",
    )
    parser.add_argument(
        "--inter-token-ms",
        type=int,
        default=8,
        help="delay between subsequent streamed tokens",
    )
    args = parser.parse_args()

    web.run_app(make_app(args), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
