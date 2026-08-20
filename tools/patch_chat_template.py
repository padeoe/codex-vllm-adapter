#!/usr/bin/env python3
"""Patch the Qwen3.8 chat template so it accepts the OpenAI Responses API 'developer' role.

The stock template raises 'Unexpected message role.' for role=developer (Codex sends one),
and raises 'System message must be at the beginning.' when vLLM turns the Responses
'instructions' field into a leading system message *and* a developer message follows it.

This patch folds every leading system/developer message into one system message.
"""
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/pretrained_models/Qwen3.8-27B-FP8/chat_template.jinja"
DST = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/padeoe/deploy/qwen38/chat_template.codex.jinja"

t = open(SRC, encoding="utf-8").read()

ANCHOR = (
    "{%- if not messages %}\n"
    "    {{- raise_exception('No messages provided.') }}\n"
    "{%- endif %}"
)
if ANCHOR not in t:
    sys.exit("anchor 1 (no-messages guard) not found -- template changed upstream")

NORM = ANCHOR + """
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

t = t.replace(ANCHOR, NORM, 1)

# Clamp unknown reasoning efforts instead of raising. Codex may send levels that belong
# to whichever model it thinks it is talking to (e.g. `high`, `minimal`); the stock
# template hard-fails on anything outside low/medium/xhigh.
EFFORT_OLD = """    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}"""
# THINKING modes:
#   off  (default) - thinking hard-disabled. Fastest, and agentic tool calling is
#                    unaffected: measured 3/3 pass, 0 tool errors, 13s vs 15s for 'low'.
#   low            - thinking ON but pinned to 'low' regardless of what the client sends.
#                    Codex requests xhigh by default and Qwen3.8 then spends its entire
#                    output budget inside <think>, so never leave this unpinned.
#   keep           - honour whatever effort the client sends (stock behaviour).
#
# CORRECTION (2026-08-19): this file previously claimed THINKING=off "BREAKS Codex tool
# calling: the model stops after a sentence instead of invoking tools". That was a
# misattribution. The real cause was Codex's `tool_mode: code_mode_only` on the
# gpt-5.6-* slugs offering only a freeform `exec` tool -- see CODEX.md. Retested after
# that fix: thinking off is fine.
MODE = os.environ.get("THINKING", "off").lower()
if MODE not in ("low", "off", "keep"):
    sys.exit("THINKING must be one of: low, off, keep")

if MODE == "low":
    EFFORT_NEW = """    {%- set resolved_reasoning_effort = 'low' %}{#- PATCH: effort pinned to low -#}"""
else:
    EFFORT_NEW = """    {%- set _requested_effort = reasoning_effort|default('xhigh') %}
    {%- if _requested_effort in ('xhigh', 'medium', 'low') %}
        {%- set resolved_reasoning_effort = _requested_effort %}
    {%- elif _requested_effort in ('high', 'max', 'highest') %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- elif _requested_effort in ('minimal', 'none', 'off', 'lowest') %}
        {%- set resolved_reasoning_effort = 'low' %}
    {%- else %}
        {%- set resolved_reasoning_effort = 'medium' %}
    {%- endif %}"""
if EFFORT_OLD not in t:
    sys.exit("anchor 3 (reasoning effort guard) not found -- template changed upstream")
t = t.replace(EFFORT_OLD, EFFORT_NEW, 1)

# Fallback: a stray non-leading developer message renders as a user turn rather than failing.
OLD = """    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"""
NEW = """    {%- elif message.role == "user" or message.role == "developer" %}
        {{- '<|im_start|>user\\n' + content + '<|im_end|>' + '\\n' }}"""
if OLD not in t:
    sys.exit("anchor 2 (user branch) not found -- template changed upstream")
t = t.replace(OLD, NEW, 1)

# Hard-disable thinking. Codex always sends a reasoning effort (its default for these
# slugs is xhigh), and Qwen3.8 reasons very verbosely under that instruction -- measured
# at 100% of the output budget spent inside <think> with no answer emitted. There is no
# "off" reasoning level to select, and vLLM has no flag to force chat-template kwargs,
# so it is enforced in the template itself.
#
# Set THINKING=keep to build a template that honours enable_thinking/reasoning_effort.
if MODE == "off":
    # 1. Never inject the "Reasoning effort is set to ..." system instruction.
    OLD_GATE = "{%- if enable_thinking is undefined or enable_thinking is true %}"
    NEW_GATE = "{%- if false %}{#- PATCH: thinking hard-disabled -#}"
    if OLD_GATE not in t:
        sys.exit("anchor 4 (enable_thinking gate) not found -- template changed upstream")
    t = t.replace(OLD_GATE, NEW_GATE, 1)

    # 2. Always open the assistant turn with an already-closed, empty think block, so
    #    the model emits its answer directly instead of reasoning first.
    OLD_GEN = "    {%- if enable_thinking is defined and enable_thinking is false %}"
    NEW_GEN = "    {%- if true %}{#- PATCH: always emit a closed empty think block -#}"
    if OLD_GEN not in t:
        sys.exit("anchor 5 (generation prompt gate) not found -- template changed upstream")
    t = t.replace(OLD_GEN, NEW_GEN, 1)

open(DST, "w", encoding="utf-8").write(t)
print("wrote", DST, len(t), "bytes", "(THINKING=%s)" % MODE)
