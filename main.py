import os
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pyngrok import ngrok
import uuid

from mem0 import AsyncMemoryClient
from openai import OpenAI
import asyncio

load_dotenv()

m = AsyncMemoryClient()

o = OpenAI(api_key=os.environ["CEREBRAS_API_KEY"], base_url="https://api.cerebras.ai/v2")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Long Term Memory Voice Agent", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def search_memory(messages: list[dict], user_id: str) -> list[dict]:
    # Build a query from the last 2 user and last 2 assistant messages
    collected: list[tuple[str, str]] = []
    counts = {"user": 0, "assistant": 0}
    for message in reversed(messages):
        role = message.get("role")
        if role in ("user", "assistant") and counts.get(role, 0) < 2:
            content = message.get("content", "").strip()
            if content:
                collected.append((role, content))
                counts[role] += 1
        if counts["user"] >= 2 and counts["assistant"] >= 2:
            break

    if not collected:
        return messages

    collected.reverse()  # chronological order
    query = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in collected
    ).strip()

    # Fetch related memories
    results = await m.search(query=query, user_id=user_id, threshold=0.2)

    # Build a readable context block from related memories
    lines = []
    if isinstance(results, list):
        for item in results:
            # Prefer common fields; fall back to stringified item
            if isinstance(item, dict):
                content = (
                    item.get("memory")
                    or item.get("content")
                    or item.get("text")
                )
            else:
                content = str(item)
            if content:
                lines.append(f"- {content}")

    if not lines:
        return messages

    context_block = "\n".join(
        [
            "Relevant memories (only use if necessary/useful)",
            *lines,
        ]
    )

    # Ensure a single system message exists and append to it
    updated = list(messages)
    system_index = next((i for i, msg in enumerate(updated) if msg.get("role") == "system"), None)
    if system_index is None:
        updated.insert(0, {"role": "system", "content": ""})
        system_index = 0

    system_content = updated[system_index].get("content", "")
    if system_content:
        system_content = f"{system_content}\n{context_block}"
    else:
        system_content = context_block.lstrip("\n")
    updated[system_index]["content"] = system_content

    # Deduplicate any additional system messages
    deduped = []
    system_kept = False
    for i, msg in enumerate(updated):
        if msg.get("role") == "system":
            if i == system_index and not system_kept:
                deduped.append(msg)
                system_kept = True
            continue
        deduped.append(msg)

    return deduped


async def background_add_memory(user_id: str, messages: list[dict]):
    try:
        # Collect all non-system messages
        payload = [
            {"role": msg.get("role"), "content": msg.get("content", "").strip()}
            for msg in messages
            if msg.get("role") != "system" and msg.get("content")
        ]

        if payload:
            # Store raw messages to avoid rejection of short texts and reduce server-side inference cost
            await m.add(payload, user_id=user_id)
    except Exception as e:
        logger.warning(f"Background add memory failed: {e}")


@app.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    print(payload)

    messages = payload.get("messages", [])
    logger.info(f"Received chat completion request with {len(messages)} messages")

    try:
        start_time = time.time()
        phone_number = (payload.get("customer") or {}).get("number") or (payload.get("call") or {}).get("customer", {}).get("number")
        user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, phone_number)) if phone_number else "anonymous"

        # Non-blocking memory add
        asyncio.create_task(background_add_memory(user_id, messages))

        messages_with_memories = await search_memory(messages, user_id)

        stream = o.chat.completions.create(
            messages=messages_with_memories,
            model=payload.get("model", "qwen-3-235b-a22b-instruct-2507"),
            stream=payload.get("stream", True),
            max_completion_tokens=payload.get("max_tokens", 250),
            temperature=payload.get("temperature", 1.0),
            top_p=payload.get("top_p", 1.0),
        )

        def generate():
            first_chunk = True
            for chunk in stream:
                if first_chunk:
                    first_chunk = False
                    end_time = time.time()
                    logger.info(f"TTFT: {end_time - start_time:.4f} seconds")
                json_data = chunk.model_dump_json()
                yield f"data: {json_data}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn

    # Setup ngrok tunnel
    logger.info("Setting up ngrok tunnel...")
    ngrok_tunnel = ngrok.connect(8000)
    logger.info(f"Public URL: {ngrok_tunnel.public_url}")
    print(f"Public URL: {ngrok_tunnel.public_url}")
    print("Client initialised")

    # Run the FastAPI server
    logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)