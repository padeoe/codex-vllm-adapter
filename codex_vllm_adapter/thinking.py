"""Map the client's requested reasoning effort onto "thinking on / thinking off".

Why this exists
---------------
Codex always asks for reasoning. Its default for these threads is a *high* effort, and
under that instruction Qwen3.5/3.8 will happily spend its entire output budget inside
`<think>` and return no answer at all. So a Codex-to-vLLM bridge needs an opinion about
thinking, and users need to be able to change that opinion without editing a Jinja
template or rebuilding anything.

Why it belongs in the adapter rather than in the chat template
--------------------------------------------------------------
vLLM's Responses API accepts a `chat_template_kwargs` object and passes it to the
template, and -- this is the load-bearing detail -- a client-supplied `enable_thinking`
**wins** over the one vLLM would derive from the effort. From
`vllm/entrypoints/openai/responses/protocol.py` (0.27.1)::

    user_kwargs = self.chat_template_kwargs or {}
    if reasoning_effort is not None and "enable_thinking" not in user_kwargs:
        extra_kwargs["enable_thinking"] = reasoning_effort != "none"

Qwen's stock template already honours that variable: the whole reasoning-effort block,
including its `raise_exception` on unknown efforts, sits inside
`{%- if enable_thinking is undefined or enable_thinking is true %}`, and when it is
false the generation prompt opens with an already-closed empty think block.

So thinking is controllable per request, from outside, on an unmodified template. An
earlier version of this project hardcoded thinking-off into a patched template; that was
a build-time answer to a run-time question.

Policy values
-------------
Each effort the client may send maps to one of:

    "off"      disable thinking
    "on"       enable thinking, forward the client's effort unchanged
    "keep"     inject nothing; let vLLM decide from the effort as it normally would
    <effort>   enable thinking, but tell the model to use *this* effort instead
               (e.g. "medium" -- useful because model templates disagree about which
               effort names exist: Qwen3.8 accepts only low/medium/xhigh and raises on
               anything else, while most models expect low/medium/high)
"""

from __future__ import annotations

import json
import os
from typing import Any

# Every effort name the OpenAI Responses schema currently permits, which is the set vLLM
# will accept without a validation error.
KNOWN_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# The shipped default, and the reasoning behind it:
#
# Thinking is off everywhere. On an agentic coding task -- which is what Codex sends --
# it was measured to cost time without buying correctness (see docs/THINKING.md), and at
# the high end it fails outright by exhausting the output budget mid-thought. Codex's
# own default effort lands at the high end, so "honour whatever the client asks for"
# would ship a broken out-of-the-box experience.
#
# `high` and above are left as explicit entries rather than falling through to
# `default`, so that turning them on is a one-word edit in a file you can read.
DEFAULT_POLICY: dict[str, str] = {
    "default": "off",
    "none": "off",
    "minimal": "off",
    "low": "off",
    "medium": "off",
    "high": "off",
    "xhigh": "off",
    "max": "off",
}

# What `--thinking on` means: thinking follows the client, but the effort is clamped to a
# name Qwen's template accepts, because `high` -- which Codex does send -- raises there.
PASSTHROUGH_POLICY: dict[str, str] = {
    "default": "medium",
    "none": "off",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}


class PolicyError(ValueError):
    """Raised for a malformed thinking policy, with a message meant for a CLI user."""


def normalize_policy(raw: Any) -> dict[str, str]:
    """Validate a policy mapping loaded from a file or built by hand."""
    if not isinstance(raw, dict):
        raise PolicyError("thinking policy must be a table/object of effort -> action")

    # Accept a [thinking] section wrapper so a policy can live inside a larger file.
    if "thinking" in raw and isinstance(raw["thinking"], dict):
        raw = raw["thinking"]

    policy: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise PolicyError(
                "thinking policy value for %r must be a string, got %r" % (key, value))
        k = key.strip().lower()
        v = value.strip().lower()
        if k != "default" and k not in KNOWN_EFFORTS:
            raise PolicyError(
                "unknown effort %r in thinking policy (expected one of: %s, or 'default')"
                % (key, ", ".join(KNOWN_EFFORTS)))
        if v not in ("off", "on", "keep") and v not in KNOWN_EFFORTS:
            raise PolicyError(
                "unknown action %r for effort %r (expected off, on, keep, or an effort "
                "name: %s)" % (value, key, ", ".join(KNOWN_EFFORTS)))
        policy[k] = v
    policy.setdefault("default", "off")
    return policy


def load_policy(path: str) -> dict[str, str]:
    """Load a policy from a .json or .toml file.

    TOML needs Python 3.11+ (stdlib `tomllib`); JSON works everywhere. Keeping both
    stdlib-only is deliberate -- this package has no dependencies.
    """
    with open(path, "rb") as fh:
        blob = fh.read()

    if os.path.splitext(path)[1].lower() == ".toml":
        try:
            import tomllib  # noqa: PLC0415 - optional, 3.11+
        except ImportError:
            raise PolicyError(
                "reading a .toml policy needs Python 3.11+; convert it to JSON or "
                "upgrade Python") from None
        try:
            raw = tomllib.loads(blob.decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - surface the parse error verbatim
            raise PolicyError("could not parse %s: %s" % (path, e)) from None
    else:
        try:
            raw = json.loads(blob.decode("utf-8"))
        except ValueError as e:
            raise PolicyError("could not parse %s: %s" % (path, e)) from None

    return normalize_policy(raw)


def apply_thinking_policy(
    body: Any, policy: dict[str, str] | None
) -> tuple[Any, dict[str, int]]:
    """Return (rewritten_body, counts). Never mutates the caller's object.

    `policy=None` disables the feature entirely: the body is returned untouched, which
    is what you want when talking to a backend that has its own opinion.
    """
    if not policy or not isinstance(body, dict):
        return body, {}

    reasoning = body.get("reasoning")
    requested = None
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
        requested = reasoning["effort"].strip().lower()

    action = policy.get(requested or "default", policy.get("default", "off"))
    if action == "keep":
        return body, {}

    stats: dict[str, int] = {}
    body = dict(body)

    if action == "off":
        enable, effort_out = False, None
    elif action == "on":
        enable, effort_out = True, None
    else:
        enable, effort_out = True, action

    # An explicit enable_thinking is what makes this deterministic: without it vLLM
    # derives the value from the effort, so a client that changed its default would
    # silently change the model's behaviour.
    ctk = dict(body.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = enable
    if effort_out is not None and requested is None:
        # No `reasoning` object to rewrite, so hand the effort to the template directly.
        # vLLM drops its own `reasoning_effort=None` when merging, so this survives.
        ctk["reasoning_effort"] = effort_out
    body["chat_template_kwargs"] = ctk

    if effort_out is not None and requested is not None and effort_out != requested:
        # The request's own effort beats chat_template_kwargs in vLLM's merge, so an
        # effort override has to be written here, not into the kwargs.
        body["reasoning"] = dict(reasoning, effort=effort_out)
        stats["rewrote_effort_%s_to_%s" % (requested, effort_out)] = 1

    # Record what the client actually asked for. With `-v` this answers the first
    # question anyone debugging thinking has -- "what effort is my client sending?" --
    # which is otherwise invisible, because clients pick a default without telling you.
    stats["effort_%s" % (requested or "absent")] = 1
    stats["thinking_on" if enable else "thinking_off"] = 1
    return body, stats
