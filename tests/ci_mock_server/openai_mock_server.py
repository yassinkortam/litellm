"""
Mock OpenAI server for CI tests.
Replaces the Railway-hosted mock server with a local process.

Supports:
- /v1/chat/completions (streaming and non-streaming)
- /v1/models
- /v1/embeddings
- Special model behaviors (429 errors, slow responses)
"""

import asyncio
import json
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock OpenAI Server for CI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SLOW_RESPONSE_DELAY = 5.0  # seconds for slow-endpoint model


@app.get("/health")
@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-5-mini", "object": "model", "owned_by": "mock"},
            {"id": "my-fake-model", "object": "model", "owned_by": "mock"},
            {"id": "my-fake-model-2", "object": "model", "owned_by": "mock"},
            {"id": "fake", "object": "model", "owned_by": "mock"},
            {"id": "slow-endpoint", "object": "model", "owned_by": "mock"},
            {"id": "429", "object": "model", "owned_by": "mock"},
            {"id": "bad-model", "object": "model", "owned_by": "mock"},
            {"id": "slow-model", "object": "model", "owned_by": "mock"},
            {"id": "fast-endpoint", "object": "model", "owned_by": "mock"},
        ],
    }


def _extract_model_name(model: str) -> str:
    """Extract the model name from the full model string (e.g., 'openai/429' -> '429')."""
    if "/" in model:
        return model.split("/")[-1]
    return model


async def _generate_streaming_response(response_id: str, created: int, model: str):
    """Generate a streaming chat completion response."""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "This is a mock response."},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    done_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model")
    stream = body.get("stream", False)

    model_name = _extract_model_name(model)

    # Handle 429 model - simulate rate limiting
    if model_name == "429":
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": "Rate limit exceeded. Please retry after some time.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    # Handle slow-endpoint model - add delay
    if model_name in ("slow-endpoint", "slow-model"):
        await asyncio.sleep(SLOW_RESPONSE_DELAY)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:
        return StreamingResponse(
            _generate_streaming_response(response_id, created, model),
            media_type="text/event-stream",
        )

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a mock response."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input", [""])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.0] * 1536}
            for i in range(len(inputs))
        ],
        "model": body.get("model", "mock-embedding"),
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


# Fine-tuning endpoints (for Azure fine-tuning tests)
@app.post("/v1/fine_tuning/jobs")
@app.post("/fine_tuning/jobs")
async def create_fine_tuning_job(request: Request):
    body = await request.json()
    return {
        "object": "fine_tuning.job",
        "id": f"ftjob-{uuid.uuid4().hex[:12]}",
        "model": body.get("model", "gpt-3.5-turbo"),
        "created_at": int(time.time()),
        "status": "created",
        "training_file": body.get("training_file", "file-abc123"),
    }


@app.get("/v1/fine_tuning/jobs")
@app.get("/fine_tuning/jobs")
async def list_fine_tuning_jobs():
    return {
        "object": "list",
        "data": [],
        "has_more": False,
    }


# Catch-all for OpenAPI spec requests (for MCP tests that may hit the base URL)
@app.get("/openapi.json")
async def openapi_spec():
    return app.openapi()


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the mock server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mock OpenAI Server for CI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
