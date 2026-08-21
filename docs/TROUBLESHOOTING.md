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

## `Unexpected message role.`

Chat-template problem, not an adapter problem: Qwen3.5/3.8's stock template does not know
Codex's `developer` role. Generate a patched template with `tools/patch_chat_template.py`
and pass it via `--chat-template`.

## `400 Unexpected reasoning effort high. Supported types are xhigh, medium, and low.`

You set `model_reasoning_effort = "high"` (or `"minimal"`) in `~/.codex/config.toml`.
Qwen's template knows only `low`/`medium`/`xhigh` and raises on anything else.

The adapter's **default** policy already prevents this: the raise lives inside the
`enable_thinking` block, and a disabled block cannot raise. If you want thinking *on*,
use `--thinking on`, which remaps `high`→`xhigh` and `minimal`→`low` instead of
forwarding a name the template will reject. See [THINKING.md](THINKING.md).

## The model thinks when I did not ask it to

Codex sends `reasoning: {"summary": "auto"}` with **no `effort`**, and Qwen's template
reads a missing effort as `xhigh` with thinking on — so a default install lands on the
most expensive setting available. The adapter disables thinking on every request unless
you configure otherwise. If you still see thinking, check:

* `--thinking keep` is not set (that mode deliberately injects nothing);
* the template actually declares the variable — `grep -c enable_thinking
  chat_template.jinja`. If it is 0 the template ignores the kwarg and only effort
  remapping will help;
* the model is not reasoning-only (gpt-oss, DeepSeek-R1), which has no off switch at all.

Turning thinking off does **not** hurt tool calling: measured 3/3 agentic passes with 0
tool errors either way, and it is faster (13 s vs 17 s mean). See [THINKING.md](THINKING.md).

## `400 tool type namespace not supported` / `tool type custom not supported`

Codex declares tools that are not plain JSON functions — `apply_patch` as `custom`, a
`namespace`-typed multi-agent tool, `web_search`. Backends reject or mishandle these. The
adapter removes them, which is what makes the shell path work instead. Seeing this error
means traffic is reaching vLLM without passing through the adapter — check that Codex's
`base_url` points at the adapter's port, not vLLM's.

## `500 HarmonyError: Unexpected token N while expecting start token 200006`

Not an adapter problem, and not fixable by one. On gpt-oss under vLLM 0.27.1 this fires
the moment the model emits a tool call, with or without Codex involved. See
[MODEL-COMPATIBILITY.md](MODEL-COMPATIBILITY.md) for the isolation steps.

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
