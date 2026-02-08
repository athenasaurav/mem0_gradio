# 🧠 Long-Term Memory Chat Agent

A Gradio-based chat application with **long-term memory** powered by [Mem0](https://mem0.ai/) and [OpenAI](https://openai.com/) models. Users are identified by phone number (hashed to UUID) so memories persist across sessions.


## Features

- **Long-term memory** via Mem0 — the AI remembers past conversations per user
- **OpenAI model auto-fetch** — enter your API key and all available models appear in a dropdown
- **Custom system prompt** — supports long, multi-line system prompts (defaults to "You are a helpful assistant")
- **Phone-based user ID** — phone number is hashed into a UUID5 for privacy; no verification needed
- **Latency dashboard** — displays Memory Search time, TTFT, total LLM time, memory save time, and memories retrieved
- **Streaming responses** — tokens appear in real-time as the model generates them
- **Public share link** — launches with `share=True` for instant HTTPS access

## Architecture

```
User (Gradio UI) → Chat Input
        ↓
  Search Mem0 for relevant memories
        ↓
  Inject memories into system prompt
        ↓
  Send to OpenAI LLM (streaming)
        ↓
  Stream response back to UI
        ↓
  Save conversation to Mem0 (background)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# There could be mistmatch due to gradio and mem0 version
# Here is my setup:
gradio                            5.22.0
gradio_client                     1.8.0
gradio_webrtc                     0.0.31
mem0ai                            1.0.3
openai                            2.17.0

# Run the app
python app.py
```

The app will print a local URL (`http://0.0.0.0:7860`) and a public Gradio share link (`https://xxxxx.gradio.live`).

## Configuration (via UI)

All configuration is done directly in the Gradio interface — no `.env` file needed.

| Field | Required | Description |
|---|---|---|
| Phone Number | Yes | Creates a unique user ID for memory. Any string works. |
| OpenAI API Key | Yes | Your OpenAI key. Click "Fetch" to load available models. |
| Model | Yes | Select from auto-fetched models after entering API key. |
| Mem0 API Key | Optional | Enables long-term memory. Leave blank to disable. |
| System Prompt | Optional | Custom instructions for the AI. Blank = default helpful assistant. |
| Max Tokens | Optional | Max response length (default: 1024). |
| Temperature | Optional | Creativity control (default: 0.7). |

## How Memory Works

1. **On each message**, the last 2 user + 2 assistant messages are used to search Mem0 for semantically relevant memories (threshold: 0.2).
2. Retrieved memories are injected into the system prompt as context: `"Relevant memories (only use if necessary/useful)"`.
3. After the LLM responds, the full conversation is saved to Mem0 in the background.
4. Each user is identified by a UUID5 hash of their phone number, so memories persist across sessions.

## Files

| File | Description |
|---|---|
| `app.py` | Main application — Gradio UI + Mem0 + OpenAI integration |
| `main.py` | Original VAPI/FastAPI version from the video (kept for reference) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Optional environment variable template |
| `.gitignore` | Git ignore rules |

## Dependencies

| Package | Purpose |
|---|---|
| `gradio` | Web UI framework |
| `openai` | OpenAI API client |
| `mem0ai` | Mem0 long-term memory SDK |
| `python-dotenv` | Optional env var loading |
