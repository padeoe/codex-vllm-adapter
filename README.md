# codex-vllm-adapter

Run **OpenAI Codex CLI** against your own **vLLM** server.

Codex speaks the Responses API, but it sends several things vLLM cannot represent. The
result is not a clean error — it looks like a bad model:

> the model replies with one sentence and then stops instead of doing the work

This is a ~200-line, zero-dependency proxy that sits in front of **stock, unmodified
vLLM** and removes exactly those things. No forked image, no patched source.

```
Codex CLI ──▶ codex-vllm-adapter ──▶ vLLM (official image)
              strips 4 request shapes    responses stream straight back, unparsed
```

## Quick start

```bash
pip install codex-vllm-adapter        # or: git clone && pip install -e .
codex-vllm-adapter --upstream http://127.0.0.1:8000 --listen 127.0.0.1:8010
```

Point Codex at the adapter in `~/.codex/config.toml`:

```toml
model_provider = "myvllm"
model = "Qwen3.8-27B-FP8"        # your REAL model id -- see the warning below
review_model = "Qwen3.8-27B-FP8"
model_context_window = 262144     # your model's true window
model_auto_compact_token_limit = 230000

[model_providers.myvllm]
name = "myvllm"
base_url = "http://127.0.0.1:8010/v1"
wire_api = "responses"
requires_openai_auth = false
```

Then run `codex`. Requires **Codex CLI ≥ 0.148** (older versions use `wire_api =
"chat"`) and Python ≥ 3.9.

> ### ⚠️ Do not name your model `gpt-5.6-*`
>
> Codex's built-in catalog marks the entire `gpt-5.6-*` family `code_mode_only`. Under
> those names Codex sends **no tools at all** — just one freeform JavaScript `exec` tool
> that no JSON-tool-calling backend can answer. **No proxy can fix this**, because the
> tool definitions never arrive in usable form. Use your real model id.
>
> This single setting is the difference between "the model is useless" and "the model
> works". See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## What it removes

| Request shape | Without the adapter |
| --- | --- |
| `additional_tools` input items | `400 — N validation errors` on every `codex resume` |
| tools whose type isn't `function` (`apply_patch` is `custom`) | `Fatal error: tool apply_patch invoked with incompatible payload`, 2–6 wasted turns per task |
| `encrypted_content` on reasoning items | `400 — Encrypted content is not supported.` |
| `compaction` items | `500` — `'ResponseCompactionItem' object has no attribute 'get'`, which Codex shows as "Reconnecting… 1/5" then "experiencing high demand" |
| `custom_tool_call` / `custom_tool_call_output` records | `500` — `'ResponseCustomToolCall' object has no attribute 'get'`. These dominate real history: one 11,386-item thread held 12,120 of each |

Everything else is forwarded byte-for-byte. Only `POST /v1/responses` is touched;
`/v1/models`, `/v1/chat/completions` and `/metrics` pass straight through. Responses are
**never parsed** — they stream back as raw bytes, so SSE needs no special handling.

## Verified

Measured against **stock `vllm/vllm-openai:v0.27.1`**, same server, adapter on and off:

| Test | Direct to vLLM | Through adapter |
| --- | --- | --- |
| Protocol suite (one request per failure mode) | 3/5 | **5/5** |
| Codex agentic task ×3 (edit a file, run it, verify) | — | **3/3, 0 tool errors, 2 real tool calls each** |
| Resume a real 11,386-item recorded thread | `500` ×30, then gives up | **pass in one request, 0 retries** |
| Unit tests (no GPU, no vLLM) | — | **12/12 in ~1 ms** |

On that resume the adapter removed 108 `custom_tool_call`, 111 `custom_tool_call_output`
and 1 `compaction` item, and stripped `encrypted_content` from 55 reasoning items —
after which the model correctly summarised a conversation it had never seen in its
original form.

Model under test: Qwen3.8-27B-FP8 on an H800. The adapter itself is model-agnostic; only
the chat template in `tools/` is Qwen-specific.

## Also needed for Qwen3.5 / Qwen3.8

Two things are **not** adapter concerns but will bite you:

1. **Chat template.** The stock template rejects Codex's `developer` role
   (`Unexpected message role.`) and raises on `reasoning_effort` values it doesn't know.
   Generate a fixed one and pass it with `--chat-template` (a supported vLLM flag):
   ```bash
   python tools/patch_chat_template.py /path/to/model chat_template.codex.jinja
   vllm serve ... --chat-template chat_template.codex.jinja
   ```
   It also hard-disables thinking, because Codex requests `xhigh` by default, under
   which the model spends its entire output budget inside `<think>` and returns nothing.
   Disabling it costs no tool-calling accuracy and is ~13% faster.

2. **Serve under your real model id**, per the warning above.

`examples/docker-compose.yml` wires the official vLLM image and the adapter together
with both of these already set.

## Why a proxy and not a patched vLLM image

Short version: every one of these fixes is a *request-side* transformation, so none of
them need to be inside vLLM. Patches anchored to vLLM's source have to be regenerated
and re-verified every release — and we shipped one that applied cleanly to a new version
and was still subtly wrong, which only a real recorded thread exposed.

The long version, with the comparison table and the evidence, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
python -m pytest tests/ -q      # or: python tests/test_sanitize.py
```

The unit tests need no GPU and no vLLM — that is deliberate, and is the main practical
argument for this design.

## License

MIT
