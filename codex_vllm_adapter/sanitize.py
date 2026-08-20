"""Rewrite a Codex Responses-API request into one vLLM can represent.

Every transformation here is **request-side**. That is not a coincidence, it is the
finding this whole project rests on: each incompatibility between Codex and vLLM is
caused by something in the request body, so none of them require touching vLLM's
internals or its response path. The adapter therefore never parses a response -- it
streams bytes straight back -- which is why SSE works without special handling.

Each rule below documents the exact failure it prevents. All four were observed against
real Codex traffic on vLLM 0.22.0 and 0.27.1.
"""

from __future__ import annotations

from typing import Any

# Item types Codex records that have no representation in vLLM's ResponseInputItem
# union. Passing them through fails the whole request, so they are removed.
UNSUPPORTED_ITEM_TYPES = frozenset({
    # {"type": "additional_tools", "role": "developer", "tools": [...]} -- Codex writes
    # this at the head of every code-mode thread. Not in OpenAI's public item union, so
    # pydantic rejects the entire request: `400 - N validation errors`, and every
    # `codex resume` on such a thread fails.
    #
    # Note this is a *lossy* drop by design. Under a code-mode-only model the tool set
    # lives ONLY here, so dropping it leaves the model blind -- the fix for that is to
    # not use a code-mode-only model id (see docs/TROUBLESHOOTING.md), not to render
    # this item into the prompt. Rendering it was tried and is worse: the model then
    # calls namespaced code-mode tools directly with malformed arguments.
    "additional_tools",
    # Codex's auto-compaction marker. vLLM returns unrecognised items as chat messages
    # and then reads `["role"]` off them, so this produced either
    # `AttributeError: 'ResponseCompactionItem' object has no attribute 'get'` or
    # `KeyError: 'role'` -- a 500, which Codex shows as "Reconnecting... 1/5" followed
    # by "We're currently experiencing high demand". It carries only OpenAI-encrypted
    # content, so there is no text to recover anyway.
    "compaction",
    # The recorded calls to, and results of, freeform ("custom") tools -- Codex's
    # `apply_patch` and code-mode `exec`. vLLM parses these into ResponseCustomToolCall
    # objects and then hits the same fallback: `'ResponseCustomToolCall' object has no
    # attribute 'get'`, a 500. These dominate real history -- one 11k-item thread held
    # 12,120 of each -- so resume is impossible without removing them.
    #
    # We remove the *live* custom tool declarations too (see the tools[] rule below), so
    # dropping their recorded history is consistent: the model is not shown transcripts
    # of a tool it can no longer call. A future version could instead rewrite them into
    # function_call / function_call_output pairs to preserve more context; that is left
    # out deliberately, because presenting a call to an undeclared tool invites the
    # model to hallucinate calling it again.
    "custom_tool_call",
    "custom_tool_call_output",
})


def sanitize_request(
    body: Any, extra_drop_types: frozenset[str] | set[str] = frozenset()
) -> tuple[Any, dict[str, int]]:
    """Return (rewritten_body, counts_of_what_changed).

    Safe to call on any JSON body: anything unrecognised is returned untouched, so a
    request that needs no rewriting is passed through byte-for-byte.

    `extra_drop_types` extends the blocklist at runtime (`--drop-item-type` on the CLI).
    This exists because the blocklist is the one part of the design that is not
    self-maintaining: if a future Codex or vLLM introduces another item type vLLM cannot
    represent, you should not have to wait for a release. The symptom is a 500 whose
    server-side traceback ends in `object has no attribute 'get'` or `KeyError: 'role'`;
    the name in that message tells you which type to add.
    """
    if not isinstance(body, dict):
        return body, {}

    drop_types = UNSUPPORTED_ITEM_TYPES | frozenset(extra_drop_types)

    stats: dict[str, int] = {}

    def bump(key: str, n: int = 1) -> None:
        if n:
            stats[key] = stats.get(key, 0) + n

    items = body.get("input")
    if isinstance(items, list):
        kept = []
        for item in items:
            if not isinstance(item, dict):
                kept.append(item)
                continue

            itype = item.get("type")

            if itype in drop_types:
                bump("dropped_%s" % itype)
                continue

            if itype == "reasoning":
                # vLLM raises `ValueError: Encrypted content is not supported.` on any
                # reasoning item carrying encrypted_content. That ciphertext is
                # OpenAI-side and cannot be decrypted locally, so strip the field. If
                # nothing readable is left, the item has no content at all -- drop it
                # rather than send an empty reasoning block.
                if item.get("encrypted_content"):
                    item = {k: v for k, v in item.items() if k != "encrypted_content"}
                    bump("stripped_encrypted_reasoning")
                if not item.get("content") and not item.get("summary"):
                    bump("dropped_empty_reasoning")
                    continue

            kept.append(item)

        if len(kept) != len(items):
            body = dict(body)
            body["input"] = kept
        elif stats:
            # encrypted_content was stripped in place on copies; rebuild the list.
            body = dict(body)
            body["input"] = kept

    tools = body.get("tools")
    if isinstance(tools, list):
        # vLLM can only emit JSON-argument function calls. Codex declares `apply_patch`
        # as {"type": "custom"} -- a freeform tool whose payload is raw text -- and may
        # also declare tool_search / web_search. Offering a tool the backend cannot
        # answer wastes 2-6 turns per task with
        # `Fatal error: tool apply_patch invoked with incompatible payload`
        # before the model gives up and falls back to the shell. Removing them means it
        # goes to the shell immediately.
        kept_tools = [
            t for t in tools
            if not (isinstance(t, dict) and t.get("type") not in (None, "function"))
        ]
        if len(kept_tools) != len(tools):
            bump("dropped_non_function_tools", len(tools) - len(kept_tools))
            body = dict(body)
            body["tools"] = kept_tools

    return body, stats
