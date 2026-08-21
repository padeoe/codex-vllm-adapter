#!/usr/bin/env python3
"""Patch a Qwen3.5/3.8 chat template so it accepts Codex's `developer` role.

Usage:
    python tools/patch_chat_template.py SRC.jinja DST.jinja [options]
    vllm serve ... --chat-template DST.jinja

`--chat-template` is a supported vLLM flag, so this is configuration, not a fork.

What is patched by default (and why it cannot be done from the adapter)
-----------------------------------------------------------------------
The Responses API has no `system` role; it has `instructions` plus items with role
`developer`. vLLM renders `instructions` into a leading system message, and Codex also
sends a developer message. The stock template then fails twice over:

    Unexpected message role.                    <- role=developer is not handled
    System message must be at the beginning.    <- system followed by developer

Both are *rendering* decisions that live in the template. This patch folds every leading
system/developer message into one system message, and renders a stray later developer
message as a user turn instead of raising.

What is NOT patched by default
------------------------------
Thinking. It used to be hardcoded off here, which was a build-time answer to a run-time
question: the stock template already gates its whole reasoning block on
`enable_thinking`, and the adapter sets that per request from a policy file. Leave this
alone and control thinking with `--thinking` / `--thinking-config`.

The flags below exist for people not running the adapter, or running a client that
sends efforts the adapter is not configured to remap.
"""

from __future__ import annotations

import argparse
import sys

# --- 1. developer role ------------------------------------------------------------

ANCHOR_NO_MESSAGES = (
    "{%- if not messages %}\n"
    "    {{- raise_exception('No messages provided.') }}\n"
    "{%- endif %}"
)

FOLD_LEADING = ANCHOR_NO_MESSAGES + """
{#- PATCH: fold leading system/developer messages into a single system message. -#}
{%- set _norm = namespace(parts=[], rest=[], head=true) %}
{%- for _m in messages %}
    {%- if _norm.head and (_m.role == 'system' or _m.role == 'developer') %}
        {%- set _txt = render_content(_m.content, false, true)|trim %}
        {%- if _txt %}
            {%- set _norm.parts = _norm.parts + [_txt] %}
        {%- endif %}
    {%- else %}
        {%- set _norm.head = false %}
        {%- set _norm.rest = _norm.rest + [_m] %}
    {%- endif %}
{%- endfor %}
{%- if _norm.parts %}
    {%- set messages = [{'role': 'system', 'content': _norm.parts | join('\\n\\n')}] + _norm.rest %}
{%- else %}
    {%- set messages = _norm.rest %}
{%- endif %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}"""

USER_BRANCH_OLD = """    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"""
USER_BRANCH_NEW = """    {%- elif message.role == "user" or message.role == "developer" %}
        {{- '<|im_start|>user\\n' + content + '<|im_end|>' + '\\n' }}"""

# --- 2. optional: tolerate unknown reasoning efforts --------------------------------

EFFORT_OLD = """    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}"""

EFFORT_NEW = """    {%- set _requested_effort = reasoning_effort|default('xhigh') %}{#- PATCH: clamp instead of raising -#}
    {%- if _requested_effort in ('xhigh', 'medium', 'low') %}
        {%- set resolved_reasoning_effort = _requested_effort %}
    {%- elif _requested_effort in ('high', 'max', 'highest') %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- elif _requested_effort in ('minimal', 'none', 'off', 'lowest') %}
        {%- set resolved_reasoning_effort = 'low' %}
    {%- else %}
        {%- set resolved_reasoning_effort = 'medium' %}
    {%- endif %}"""

# --- 3. optional: hardcode thinking off ---------------------------------------------

THINK_GATE_OLD = "{%- if enable_thinking is undefined or enable_thinking is true %}"
THINK_GATE_NEW = "{%- if false %}{#- PATCH: thinking hard-disabled -#}"
GEN_GATE_OLD = "    {%- if enable_thinking is defined and enable_thinking is false %}"
GEN_GATE_NEW = "    {%- if true %}{#- PATCH: always open with a closed, empty think block -#}"


def replace_once(text: str, old: str, new: str, what: str) -> str:
    if old not in text:
        sys.exit(
            "anchor not found: %s\n"
            "The upstream template has changed. Compare it against the strings in this "
            "script rather than editing the template by hand." % what)
    return text.replace(old, new, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="the model's stock chat_template.jinja")
    ap.add_argument("dst", help="where to write the patched template")
    ap.add_argument(
        "--clamp-effort", action="store_true",
        help="map unknown reasoning efforts onto supported ones instead of raising a "
             "Jinja error. Unnecessary if the adapter remaps efforts for you; useful as "
             "a safety net, or for a client the adapter does not sit in front of.")
    ap.add_argument(
        "--disable-thinking", action="store_true",
        help="hardcode thinking off in the template itself. Only for setups without the "
             "adapter -- it cannot be re-enabled per request, and it overrides any "
             "thinking policy you configure.")
    args = ap.parse_args(argv)

    with open(args.src, encoding="utf-8") as fh:
        t = fh.read()

    t = replace_once(t, ANCHOR_NO_MESSAGES, FOLD_LEADING, "no-messages guard")
    t = replace_once(t, USER_BRANCH_OLD, USER_BRANCH_NEW, "user-role branch")

    applied = ["developer-role"]
    if args.clamp_effort:
        t = replace_once(t, EFFORT_OLD, EFFORT_NEW, "reasoning-effort guard")
        applied.append("clamp-effort")
    if args.disable_thinking:
        t = replace_once(t, THINK_GATE_OLD, THINK_GATE_NEW, "enable_thinking gate")
        t = replace_once(t, GEN_GATE_OLD, GEN_GATE_NEW, "generation-prompt gate")
        applied.append("disable-thinking")

    with open(args.dst, "w", encoding="utf-8") as fh:
        fh.write(t)
    print("wrote %s (%d bytes); patches applied: %s"
          % (args.dst, len(t), ", ".join(applied)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
