# Why an adapter, and not a patched vLLM image

This project started as a set of patches applied directly to vLLM's source inside the
container. That worked, and it is how the original deployment ran in production for a
day. It is not what you should use, and this document explains the reasoning so you can
disagree with it on the evidence rather than on assertion.

## The finding that decides it

Every known Codex/vLLM incompatibility is caused by **something in the request body**.

We verified this by locating each patch in vLLM's source. All of them live in
`vllm/entrypoints/openai/responses/`:

| Fix | Where it lived | Direction |
| --- | --- | --- |
| drop `additional_tools` items | `protocol.py`, a `mode="before"` request validator | request |
| drop non-function tools | same validator | request |
| skip encrypted reasoning | `utils.py` `_construct_message_from_response_item` | request → prompt |
| skip compaction / role-less items | same function | request → prompt |

Nothing touches `build_response_output_items`, the output path. That has two
consequences:

1. An adapter sitting **in front of** stock vLLM can produce an identical effect by
   removing the same things from the request before vLLM ever sees them.
2. The adapter never has to parse a response. It streams bytes straight through, so
   SSE works with no special handling and a future change to vLLM's event format
   cannot break it.

## Comparison

| | Patch vLLM source | Pre-built image | **Adapter (this project)** |
| --- | --- | --- | --- |
| Survives a vLLM upgrade | ✗ anchored to source text | ✗ rebuild + republish | ✓ no coupling to internals |
| Testable without a GPU | ✗ needs a 27B model on 80GB | ✗ same | ✓ 10 unit tests, ~1 ms |
| Users run official images | ✓ | ✗ must trust yours | ✓ |
| Distribution size | ~2 KB of scripts | 21 GB per release | ~10 KB, zero deps |
| Works with a hosted endpoint | ✗ needs the server | ✗ | ✓ point it anywhere |
| Extra network hop | none | none | one, in-process JSON edit |
| Blast radius of a bug | shadows vLLM internals | same | request rewriting only |

## The upgrade argument, concretely

The patch scripts anchor on exact upstream source strings and refuse to apply if an
anchor moves. That is the good failure. The bad failure is subtler and we hit it:

Going from vLLM 0.22.0 to 0.27.1, all four patch hunks applied **cleanly** and the
server booted fine. Resuming a real 11,386-item thread then returned `500 KeyError:
'role'`, because upstream's fallback returns *any* dict as a chat message and our guard
only checked `isinstance(item, dict)`. The patch was not rejected; it was simply no
longer sufficient. Only an end-to-end run against real recorded history exposed it.

A patch that applies cleanly and is still wrong is the failure mode you cannot automate
away — and every vLLM release is a fresh roll of that dice. vLLM shipped 0.22 → 0.27.1
in roughly two months.

The adapter has no anchors. It removes item types by name from a JSON body. Upstream can
refactor `utils.py` freely.

## Why not the pre-built image

It is the friendliest thing to hand someone: one `docker run`. But for a project meant
to be shared:

* you inherit a release treadmill — every vLLM version needs a rebuild, a re-test and a
  21 GB push, or your users fall behind on kernels and model support;
* many organisations will not run an unofficial inference image, and reviewing a 21 GB
  image is not practical;
* it does not help anyone whose endpoint is hosted or proxied, where you cannot swap the
  server at all;
* it couples your users' vLLM version to yours, so a user who needs a newer vLLM for an
  unrelated model is stuck.

The adapter has none of these properties, and if someone *does* want one container, the
`examples/docker-compose.yml` here composes the official vLLM image with the adapter and
gets the same ergonomics without anyone shipping a fork.

## The honest weakness

The adapter identifies unrepresentable items by **type name**, from a blocklist. vLLM's
own patched version could use a more general rule (`pass a dict through only if it has a
role`) because it runs after pydantic has parsed the body and can see which items became
opaque objects. The adapter sees plain JSON and cannot.

That difference is not theoretical — it bit during validation. The first version of the
blocklist held only `additional_tools` and `compaction`, which passed every unit test and
every synthetic protocol probe, and still failed on a real thread: `custom_tool_call` and
`custom_tool_call_output` records also become opaque objects, and there were 12,120 of
each in one thread.

Two things mitigate it:

* `--drop-item-type TYPE` (repeatable) extends the blocklist at runtime, so a new item
  type does not require a release. The 500's traceback names the class that failed
  (`'ResponseCustomToolCall' object has no attribute 'get'`), which tells you the type to
  add.
* Unknown types are **passed through**, not swallowed. The failure mode is a loud 500
  with a name in it, never silently dropped history.

The alternative — an allowlist of representable types — would fail closed instead, but
would silently discard any legitimate new Codex item type. For a tool whose job is
preserving conversation history, failing loud beats failing quiet.

## What the adapter deliberately does not fix

**Choosing a code-mode-only model id.** Codex's compiled-in catalog marks the whole
`gpt-5.6-*` family `"tool_mode": "code_mode_only"`. Under those names Codex sends no
top-level `tools` at all and offers a single freeform JavaScript `exec` tool, which no
JSON-tool-calling backend can answer. No proxy can repair that, because the tool
definitions never arrive in a usable form. The fix is a client-side setting: use your
real model id. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**The chat template.** Qwen3.5/3.8's stock template rejects Codex's `developer` role and
raises on unknown `reasoning_effort` values. That is fixed with a patched template file
passed via `--chat-template`, which is a supported vLLM flag and not a modification of
vLLM. `tools/patch_chat_template.py` generates it.
