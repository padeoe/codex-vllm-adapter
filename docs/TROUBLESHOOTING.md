# Troubleshooting

Symptom-first, because every one of these presents as something other than its cause.

## "The model replies with one sentence and then stops instead of doing the work"

**Cause: your `model` is a `gpt-5.6-*` name.** This is a client-side setting and no
server or proxy can fix it.

Codex has a compiled-in model catalog, and every `gpt-5.6-*` entry is marked
`"tool_mode": "code_mode_only"`. Under those names Codex sends **no top-level `tools`
field at all**. The whole tool set arrives as an `additional_tools` item declaring one
namespace whose only real entry is `exec` — a *custom (freeform)* tool whose payload is
raw JavaScript for a V8 isolate. A backend that speaks JSON tool calling cannot answer
that, so every agentic turn dies with:

```
ERROR codex_core::tools::router: error=Fatal error: tool exec invoked with incompatible payload
```

**Fix:** set `model` to your actual served model id (e.g. `Qwen3.8-27B-FP8`). Names
outside Codex's catalog carry no tool-mode policy, so Codex declares ordinary function
tools. Measured on a real agentic task, 3 runs each:

| Configuration | Result |
| --- | --- |
| `gpt-5.6-sol` (code mode) | completes, but 6–10 `incompatible payload` errors per task |
| real model id, tools unfiltered | completes, 2–6 errors (all `apply_patch`) |
| real model id + adapter | **completes, 0 errors** |

Things that do **not** work, all measured — don't retry them: `codex features disable
code_mode_host`, `-c features.code_mode=false`, renaming the `codex-code-mode-host`
binary, or a `model_catalog_json` override setting `tool_mode: "direct"`.

## `400 — N validation errors` on every `codex resume`

Codex records `{"type":"additional_tools",...}` at the head of code-mode threads. That
type is not in OpenAI's public `ResponseInputItem` union, so pydantic rejects the whole
request. The adapter drops the item.

## `400 — Encrypted content is not supported.`

Threads recorded against real OpenAI carry encrypted reasoning. The ciphertext cannot be
decrypted outside OpenAI. The adapter strips the field, keeping any readable summary
text, and drops the item only if nothing readable remains.

## `500` and Codex shows "Reconnecting… 1/5" then "experiencing high demand"

Two different upstream crashes present this way. Check the server log:

* `AttributeError: 'ResponseCompactionItem' object has no attribute 'get'`
* `AttributeError: 'ResponseCustomToolCall' object has no attribute 'get'`
* `KeyError: 'role'` in `chat_utils._parse_chat_message_content`

All come from vLLM returning an unrecognised item as if it were a chat message. The
adapter removes these before they reach vLLM. Note the misleading client message: this is
not rate limiting and not "high demand", it is a 500 being retried five times.

**If the class name in the traceback is one the adapter doesn't know yet**, add it
without waiting for a release — the type name is the snake_case form of the class:

```bash
codex-vllm-adapter --drop-item-type some_new_item_type ...
```

Please also open an issue with that traceback line so it can ship in the defaults.

## `Fatal error: tool apply_patch invoked with incompatible payload`

Even under a non-code-mode model, Codex declares `apply_patch` as `{"type":"custom"}`.
The adapter removes non-function tools, so the model uses the shell instead. Neither
`apply_patch_tool_type: "function"` nor `-c features.apply_patch_freeform=false` changes
the declaration.

## `Unexpected message role.` or a Jinja error mentioning `reasoning_effort`

Chat-template problem, not an adapter problem. Qwen3.5/3.8's stock template rejects
Codex's `developer` role and accepts only `low`/`medium`/`xhigh` for `reasoning_effort`,
while Codex may send `high` or `minimal`. Generate a patched template with
`tools/patch_chat_template.py` and pass it via `--chat-template`.

## The model spends its whole output budget thinking and returns nothing

Codex requests `reasoning_effort: xhigh` by default. There is no "off" level and vLLM
cannot force template kwargs, so the template must hard-disable thinking. Build it with
`THINKING=off` (the default in `tools/patch_chat_template.py`).

Disabling thinking does **not** hurt tool calling — measured 3/3 pass with 0 tool errors
either way, and it is ~13% faster.

## `Failed to run pre-sampling compact` when resuming a long thread

Client-side: `model_context_window` is larger than what the server actually serves, so
Codex lets the prompt grow past the real limit and compaction fires too late. Set it to
the true window (e.g. `262144`) and `model_auto_compact_token_limit` below it.

## `codex resume` shows none of my old threads

Codex filters the resume picker by the **provider table name**, not the model id. If you
renamed your provider, old threads are still on disk but hidden. Either keep the old
table name or accept the split — the model id is *not* part of the filter, so resuming a
thread under a different model works and only prints a cosmetic warning.

## `warning: Model metadata for X not found`

Cosmetic. Codex falls back to default metadata. The value that actually matters is
`model_context_window`, which you set explicitly. Silence it by supplying a
`model_catalog_json` if you like — note it is a **file path**, not inline JSON, and it
*replaces* Codex's built-in model list rather than extending it.

## Changes to config seem ignored

Codex runs a background **app-server** that outlives sessions and caches some state at
startup. If `curl` behaves correctly but the interactive TUI does not, restart it:

```bash
pkill -f 'app-server --listen'
```
