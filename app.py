import time
import uuid
import logging
import gradio as gr
from openai import OpenAI
from mem0 import MemoryClient

# ---------------------------------------------------------------------------
# Gradio version detection
# ---------------------------------------------------------------------------
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user_id(phone: str) -> str:
    """Deterministic UUID5 from a phone number string."""
    phone = phone.strip()
    if not phone:
        return "anonymous"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, phone))


def fetch_models(api_key: str):
    """Return a sorted list of model IDs available for the given OpenAI API key."""
    if not api_key or not api_key.strip():
        return gr.update(choices=[], value=None)
    try:
        client = OpenAI(api_key=api_key.strip())
        models = client.models.list()
        names = sorted([m.id for m in models.data])
        default = None
        for preferred in ("gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"):
            if preferred in names:
                default = preferred
                break
        if default is None and names:
            default = names[0]
        return gr.update(choices=names, value=default)
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return gr.update(choices=[f"Error: {e}"], value=None)


# ---------------------------------------------------------------------------
# Memory helpers (sync — Gradio runs in threads)
# ---------------------------------------------------------------------------

def search_memory(mem_client, messages: list, user_id: str):
    """Search Mem0 for relevant memories and inject into system prompt.
    Returns (updated_messages, memory_count)."""
    if mem_client is None:
        return messages, 0

    # Build query from last 2 user + 2 assistant messages
    collected = []
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
        return messages, 0

    collected.reverse()
    query = "\n".join(
        f"{'User' if r == 'user' else 'Assistant'}: {t}" for r, t in collected
    ).strip()

    try:
        results = mem_client.search(
            query=query,
            filters={"AND": [{"user_id": user_id}]},
            version="v2",
        )
    except Exception as e:
        logger.warning(f"Memory search failed: {e}")
        return messages, 0

    lines = []
    # v2 may return {"results": [...]} or a plain list
    items = results
    if isinstance(results, dict) and "results" in results:
        items = results["results"]
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                content = item.get("memory") or item.get("content") or item.get("text")
            else:
                content = str(item)
            if content:
                lines.append(f"- {content}")

    if not lines:
        return messages, 0

    context_block = "\n".join(["Relevant memories (only use if necessary/useful)", *lines])

    updated = list(messages)
    sys_idx = next((i for i, m in enumerate(updated) if m.get("role") == "system"), None)
    if sys_idx is None:
        updated.insert(0, {"role": "system", "content": ""})
        sys_idx = 0

    sys_content = updated[sys_idx].get("content", "")
    updated[sys_idx] = {
        "role": "system",
        "content": f"{sys_content}\n{context_block}" if sys_content else context_block,
    }

    # Deduplicate system messages
    deduped, kept = [], False
    for i, msg in enumerate(updated):
        if msg.get("role") == "system":
            if i == sys_idx and not kept:
                deduped.append(msg)
                kept = True
            continue
        deduped.append(msg)
    return deduped, len(lines)


def add_memory(mem_client, user_id: str, messages: list):
    """Store conversation messages in Mem0."""
    if mem_client is None:
        return
    try:
        payload = [
            {"role": msg["role"], "content": msg["content"].strip()}
            for msg in messages
            if msg.get("role") != "system" and msg.get("content")
        ]
        if payload:
            mem_client.add(payload, user_id=user_id)
    except Exception as e:
        logger.warning(f"Memory add failed: {e}")


# ---------------------------------------------------------------------------
# Chat function (generator for streaming)
# ---------------------------------------------------------------------------

def chat_respond(
    user_message,
    history,
    openai_key,
    mem0_key,
    model_name,
    system_prompt,
    phone_number,
    max_tokens,
    temperature,
):
    """Process a user message: search memory -> call LLM -> save memory -> stream response."""

    # --- Validation ---
    if not openai_key or not openai_key.strip():
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "⚠️ Please enter your **OpenAI API key** first."},
        ]
        yield history, "*Waiting for configuration...*"
        return
    if not model_name:
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "⚠️ Please select a **model** first (click Fetch Models)."},
        ]
        yield history, "*Waiting for configuration...*"
        return
    if not phone_number or not phone_number.strip():
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "⚠️ Please enter a **phone number** to identify your memory session."},
        ]
        yield history, "*Waiting for configuration...*"
        return

    # --- Append user message to history ---
    history = history + [{"role": "user", "content": user_message}]
    yield history, "*Processing...*"

    # --- Setup clients ---
    openai_client = OpenAI(api_key=openai_key.strip())
    mem_client = None
    if mem0_key and mem0_key.strip():
        try:
            mem_client = MemoryClient(api_key=mem0_key.strip())
        except Exception as e:
            logger.warning(f"Mem0 client init failed: {e}")

    user_id = make_user_id(phone_number)

    # --- Build messages list for the LLM ---
    sys_prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_prompt}]
    for msg in history:
        if msg["role"] in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # --- Memory search ---
    mem_search_start = time.time()
    messages_with_mem, mem_count = search_memory(mem_client, messages, user_id)
    mem_search_ms = (time.time() - mem_search_start) * 1000

    # --- LLM call (streaming) ---
    llm_start = time.time()
    try:
        stream = openai_client.chat.completions.create(
            model=model_name,
            messages=messages_with_mem,
            stream=True,
            max_completion_tokens=int(max_tokens),
            temperature=float(temperature),
        )
    except Exception as e:
        history = history + [{"role": "assistant", "content": f"❌ **LLM Error:** {e}"}]
        yield history, f"*Error: {e}*"
        return

    # --- Stream response ---
    assistant_text = ""
    ttft = None
    history = history + [{"role": "assistant", "content": ""}]

    for chunk in stream:
        if ttft is None:
            ttft = (time.time() - llm_start) * 1000

        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            assistant_text += delta.content
            history[-1] = {"role": "assistant", "content": assistant_text}
            yield history, "*Streaming...*"

    total_ms = (time.time() - llm_start) * 1000

    # --- Save memory ---
    mem_save_start = time.time()
    full_convo = messages + [{"role": "assistant", "content": assistant_text}]
    add_memory(mem_client, user_id, full_convo)
    mem_save_ms = (time.time() - mem_save_start) * 1000

    # --- Build latency info ---
    ttft_str = f"**{ttft:.0f} ms**" if ttft is not None else "N/A"
    latency_info = (
        f"### 📊 Latency Stats\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Memory Search | **{mem_search_ms:.0f} ms** |\n"
        f"| TTFT (Time to First Token) | {ttft_str} |\n"
        f"| Total LLM Streaming | **{total_ms:.0f} ms** |\n"
        f"| Memory Save | **{mem_save_ms:.0f} ms** |\n"
        f"| Memories Retrieved | **{mem_count}** |\n"
        f"| User ID | `{user_id[:12]}...` |\n"
        f"| Model | `{model_name}` |"
    )

    yield history, latency_info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    # Chatbot kwargs — handle differences between Gradio versions
    chatbot_kwargs = dict(
        label="Chat",
        height=620,
    )
    if GRADIO_MAJOR >= 6:
        chatbot_kwargs["buttons"] = ["copy", "copy_all"]
    else:
        chatbot_kwargs["show_copy_button"] = True
        # Gradio 4.x defaults to 'tuples' format; must explicitly set 'messages'
        chatbot_kwargs["type"] = "messages"

    with gr.Blocks(title="Memory Chat Agent") as demo:
        gr.Markdown(
            "# 🧠 Long-Term Memory Chat Agent\n"
            "Chat with an AI that **remembers** across sessions using "
            "[Mem0](https://mem0.ai/). Powered by OpenAI models."
        )

        with gr.Row():
            # ---- Left sidebar: Config ----
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### ⚙️ Configuration")

                phone_number = gr.Textbox(
                    label="📱 Phone Number (User ID)",
                    placeholder="+1234567890",
                    info="Creates a unique memory session. No verification.",
                )

                openai_key = gr.Textbox(
                    label="🔑 OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="🤖 Model",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    fetch_btn = gr.Button("🔄 Fetch", scale=1)

                mem0_key = gr.Textbox(
                    label="🧠 Mem0 API Key",
                    type="password",
                    placeholder="m0-...",
                    info="Optional — leave blank to disable long-term memory.",
                )

                system_prompt = gr.Textbox(
                    label="📝 System Prompt",
                    placeholder="You are a helpful assistant.",
                    info="Leave blank for default. Supports long prompts.",
                    lines=6,
                    max_lines=20,
                )

                with gr.Row():
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=50,
                        maximum=4096,
                        value=1024,
                        step=50,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.05,
                    )

                gr.Markdown("---")
                latency_display = gr.Markdown(
                    value="*Latency stats appear here after first message.*",
                )

            # ---- Right: Chat area ----
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(**chatbot_kwargs)
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message here...",
                        scale=5,
                        show_label=False,
                        lines=1,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                clear_btn = gr.Button("🗑️ Clear Chat")

        # ---- Event wiring ----
        fetch_btn.click(
            fn=fetch_models,
            inputs=[openai_key],
            outputs=[model_dropdown],
        )

        chat_inputs = [
            user_input,
            chatbot,
            openai_key,
            mem0_key,
            model_dropdown,
            system_prompt,
            phone_number,
            max_tokens,
            temperature,
        ]
        chat_outputs = [chatbot, latency_display]

        send_btn.click(
            fn=chat_respond,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(fn=lambda: "", outputs=[user_input])

        user_input.submit(
            fn=chat_respond,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(fn=lambda: "", outputs=[user_input])

        clear_btn.click(
            fn=lambda: ([], "*Latency stats appear here after first message.*"),
            outputs=[chatbot, latency_display],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
