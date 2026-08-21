# codex-vllm-adapter

Run **OpenAI Codex CLI/Desktop** against your own **vLLM** server.

Codex speaks the Responses API, but it sends several things vLLM cannot represent. The
result is not a clean error — it looks like a bad model:

> the model replies with one sentence and then stops instead of doing the work

This is a small, zero-dependency proxy that sits in front of **stock, unmodified vLLM**
and rewrites exactly those things. No forked image, no patched source.

```
Codex CLI/Desktop ──▶ codex-vllm-adapter ──▶ vLLM (official)
                      strips 4 request shapes    responses stream straight back, unparsed
                      sets the thinking policy
```

Tested end to end on **Qwen3.8-27B-FP8**; see
[docs/MODEL-COMPATIBILITY.md](docs/MODEL-COMPATIBILITY.md) for what that does and does not
imply about your model.

## Quick start

### 1. Start vLLM

Any vLLM ≥ 0.27 serving the **Responses API** will do — the official image, unmodified.
Qwen3.5/3.8 needs one preparatory step, because its stock chat template does not know
Codex's `developer` role:

```bash
python tools/patch_chat_template.py \
    /models/Qwen3.8-27B-FP8/chat_template.jinja chat_template.codex.jinja
```

That is the only template change, and `--chat-template` is a supported vLLM flag rather
than a fork. Then, exactly as measured in this repo (one 80 GB H800):

```bash
docker run -d --name codex-vllm --gpus all --ipc=host \
  -v /models/Qwen3.8-27B-FP8:/models/qwen38:ro \
  -v "$PWD/chat_template.codex.jinja":/etc/chat_template.jinja:ro \
  -p 127.0.0.1:8000:8000 \
  --entrypoint vllm vllm/vllm-openai:v0.27.1 serve /models/qwen38 \
    --served-model-name Qwen3.8-27B-FP8 \
    --chat-template /etc/chat_template.jinja \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.72 \
    --max-model-len 262144 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Two flags carry more weight than they look:

* `--served-model-name` must be your **real** model id — see the warning below;
* `--enable-auto-tool-choice --tool-call-parser` is what makes tool calling work at all.
  Pick the parser that matches your model family (`ls vllm/tool_parsers/` in the image).

The checkpoint ships an MTP head, so adding
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` is roughly a 1.7×
single-stream speedup. Wait for `Application startup complete`, then confirm with
`curl -s http://127.0.0.1:8000/v1/models`.

### 2. Start the adapter

```bash
pip install codex-vllm-adapter        # or: git clone && pip install -e .
codex-vllm-adapter --upstream http://127.0.0.1:8000 --listen 127.0.0.1:8010
```

Add `-v` to see what it changes per request, `--thinking on` to let the model reason.

### 3. Point Codex at the adapter

In `~/.codex/config.toml`:

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

Then run `codex`, or restart the Desktop app. Requires a Codex build **≥ 0.148** (older
ones use `wire_api = "chat"`) and Python ≥ 3.9 for the adapter itself.

> ### ⚠️ Do not name your model `gpt-5.6-*`
>
> Codex's built-in catalog marks the entire `gpt-5.6-*` family `code_mode_only`. Under
> those names Codex sends **no tools at all** — just one freeform JavaScript `exec` tool
> that no JSON-tool-calling backend can answer. **No proxy can fix this**, because the
> tool definitions never arrive in usable form. Use your real model id.
>
> This single setting is the difference between "the model is useless" and "the model
> works". See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## What it changes

| Request shape | Without the adapter |
| --- | --- |
| `additional_tools` input items | `400 — N validation errors` on every `codex resume` |
| tools whose type isn't `function` (`apply_patch` is `custom`) | `Fatal error: tool apply_patch invoked with incompatible payload`, 2–6 wasted turns per task |
| `encrypted_content` on reasoning items | `400 — Encrypted content is not supported.` |
| `compaction` items | `500` — `'ResponseCompactionItem' object has no attribute 'get'`, which Codex shows as "Reconnecting… 1/5" then "experiencing high demand" |
| `custom_tool_call` / `custom_tool_call_output` records | `500` — `'ResponseCustomToolCall' object has no attribute 'get'`. These dominate real history: one 11,386-item thread held 12,120 of each |

and one thing it *sets* rather than removes:

| Request field | Why |
| --- | --- |
| `chat_template_kwargs.enable_thinking` | Codex sends no reasoning effort by default, which Qwen's template reads as *maximum* thinking. Off at every level unless you say otherwise — `--thinking on`, or per level via `--thinking-config`. Two of the five efforts Codex can send (`high`, `minimal`) are otherwise a **400**. See [docs/THINKING.md](docs/THINKING.md) |

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
| Every reasoning effort Codex can send | 2 of 6 are `400` | **6/6 accepted** |
| Unit tests (no GPU, no vLLM) | — | **33/33 in ~10 ms** |

On that resume the adapter removed 108 `custom_tool_call`, 111 `custom_tool_call_output`
and 1 `compaction` item, and stripped `encrypted_content` from 55 reasoning items —
after which the model correctly summarised a conversation it had never seen in its
original form.

Model under test: Qwen3.8-27B-FP8 on an H800. The adapter itself is model-agnostic; only
the chat template helper in `tools/` is Qwen-specific. A second family (gpt-oss-20b) was
spot-checked, with instructive results — see
[docs/MODEL-COMPATIBILITY.md](docs/MODEL-COMPATIBILITY.md).

## The two server-side things that are not adapter concerns

Both appear in the quick start; they are called out again because they cause failures
that look like adapter or model problems and are neither:

1. **The chat template.** Qwen3.5/3.8's stock template rejects Codex's `developer` role
   (`Unexpected message role.`). `tools/patch_chat_template.py` fixes that and *only*
   that — thinking is deliberately left alone, because the stock template already gates
   it on `enable_thinking`, which the adapter sets per request.
2. **Serving under your real model id**, per the warning above.

`examples/docker-compose.yml` wires the official vLLM image and the adapter together with
both already set, plus a thinking policy file.

## Why a proxy and not a patched vLLM image

Short version: every one of these fixes is a *request-side* transformation, so none of
them need to be inside vLLM. Patches anchored to vLLM's source have to be regenerated
and re-verified every release — and we shipped one that applied cleanly to a new version
and was still subtly wrong, which only a real recorded thread exposed.

The long version, with the comparison table and the evidence, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
python -m unittest discover -s tests -v     # 33 tests, no GPU, no vLLM, ~10 ms
```

That the unit tests need no GPU and no vLLM is deliberate, and is the main practical
argument for this design. CI runs them on Python 3.9 / 3.12 / 3.13.

Documentation map:

| File | What is in it |
| --- | --- |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | symptom-first index of every error we hit, and its real cause |
| [docs/THINKING.md](docs/THINKING.md) | reasoning effort: the mechanism, the measurements, the config file |
| [docs/MODEL-COMPATIBILITY.md](docs/MODEL-COMPATIBILITY.md) | will this work with *your* model — three checks, and a worked counter-example |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | why a proxy rather than a patched vLLM, with the evidence |

## License

MIT
