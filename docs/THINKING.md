# Thinking / reasoning effort

Codex assumes it is talking to a model that reasons, and asks for a reasoning effort.
Local models disagree wildly about what that means, and a Qwen served straight out of the
box will either think when you did not want it to or refuse the request outright. This
page explains the mechanism, shows what was measured, and tells you which knob to turn.

**Short version:** thinking is **off at every effort level by default**. Change it with
`--thinking on`, or per level with `--thinking-config`.

## The mechanism

vLLM's Responses API accepts a `chat_template_kwargs` object and hands it to the chat
template. Crucially, a client-supplied `enable_thinking` **wins** over the value vLLM
would otherwise derive from the effort -- from `responses/protocol.py` in 0.27.1:

```python
user_kwargs = self.chat_template_kwargs or {}
if reasoning_effort is not None and "enable_thinking" not in user_kwargs:
    extra_kwargs["enable_thinking"] = reasoning_effort != "none"
```

Qwen3.5/3.8's **stock** template already honours that variable. Its entire reasoning
block -- including the `raise_exception` on unknown efforts -- sits inside
`{%- if enable_thinking is undefined or enable_thinking is true %}`, and when the value is
false the generation prompt opens with an already-closed empty `<think></think>`.

So thinking is controllable per request, from outside, on an unmodified template. The
adapter sets `chat_template_kwargs.enable_thinking` explicitly on every request, which is
what makes the behaviour deterministic rather than dependent on what your client happens
to send this week.

> An earlier version of this project hardcoded thinking-off into a patched template. That
> was a build-time answer to a run-time question, and it is why `--disable-thinking` is
> now merely an option on `tools/patch_chat_template.py` rather than the default.

## Why the default is "off"

Three measured reasons, on Qwen3.8-27B-FP8 / stock `vllm/vllm-openai:v0.27.1`.

**1. Codex sends no effort at all unless you set one -- and "no effort" means maximum
thinking.** Under a real (non-catalog) model id, `codex exec` sends `reasoning: {"summary":
"auto"}` with **no `effort` key**. Nothing then sets `enable_thinking`, and Qwen's template
defaults it to *on* with `reasoning_effort|default('xhigh')`. The out-of-the-box path is
the most expensive one available:

| Request | Result on stock vLLM |
| --- | --- |
| no `effort` key (Codex's default) | thinking **on at xhigh** -- 65 output tokens to answer `17*23` |
| same, through the adapter | thinking off -- **4 output tokens**, 0.1 s |

**2. Two of the five efforts Codex can send are hard errors.** The stock template accepts
only `low`, `medium` and `xhigh`; `model_reasoning_effort = "high"` -- an obvious thing to
set -- raises a Jinja exception that surfaces as a 400.

| `model_reasoning_effort` | What Codex sends | Direct to stock vLLM | Through the adapter |
| --- | --- | --- | --- |
| *(unset)* | *no `effort` key* | thinking on @ xhigh | per policy |
| `minimal` | `minimal` | **400 Unexpected reasoning effort** | ok (`off`, or remapped to `low`) |
| `low` | `low` | thinking on | ok |
| `medium` | `medium` | thinking on | ok |
| `high` | `high` | **400 Unexpected reasoning effort** | ok (`off`, or remapped to `xhigh`) |
| `xhigh` | `xhigh` | thinking on | ok |

Both failures disappear under the default policy without any effort remapping at all,
because a disabled block cannot raise.

**3. On agentic work, thinking bought nothing.** The task: find and fix a two-part bug in
a Python file, run it, verify the output. Three runs per configuration, same server:

| Configuration | Passed | Mean wall time | Tool errors |
| --- | --- | --- | --- |
| thinking off (default policy) | 3/3 | **13 s** | 0 |
| thinking on, effort `medium` | 3/3 | 17 s | 0 |
| thinking on, `high` -> remapped `xhigh` | 3/3 | 14 s | 0 |

Thinking is not *harmful* here -- it costs roughly 10-30% wall time and does not break
tool calling. It simply did not change the outcome, so the cheap setting ships as the
default. On harder or more open-ended work, turn it on and measure your own task.

**What thinking-off does not fix:** on a genuinely hard reasoning prompt (a combinatorial
proof), the model ran to a 16,384-token limit and returned no answer *in both modes*. If
your model rambles, thinking is not the variable to blame.

## Configuring it

The shorthand:

| Flag | Effect |
| --- | --- |
| *(none)* / `--thinking off` | thinking disabled at every level |
| `--thinking on` | thinking follows the client's effort, remapped to a name Qwen accepts (`high`->`xhigh`, `minimal`->`low`); `none` still means off |
| `--thinking keep` | the adapter injects nothing; vLLM behaves exactly as it would without it |

For per-level control, copy [`examples/thinking.toml`](../examples/thinking.toml) (or
`thinking.json` for Python < 3.11, where `tomllib` does not exist) and pass
`--thinking-config`:

```toml
[thinking]
default = "off"   # also covers requests carrying no effort at all -- Codex's default
low     = "off"
medium  = "off"
high    = "xhigh" # thinking ON at these two, so the client's setting becomes a real switch
xhigh   = "xhigh"
```

Each value is one of:

| Value | Meaning |
| --- | --- |
| `off` | disable thinking (`enable_thinking=false`) |
| `on` | enable it, forward the client's effort unchanged |
| `keep` | inject nothing for this level; let vLLM decide |
| an effort name | enable it, but tell the model to use *this* effort instead |

That last column exists because model templates disagree about which effort names exist.
Qwen3.5/3.8 knows `low`/`medium`/`xhigh`; most other models expect `low`/`medium`/`high`.
Set it to match **your** model's template.

## Seeing what your client asks for

Run with `-v` and the adapter logs the effort on every request, which is otherwise
invisible:

```
sanitized /v1/responses: dropped_non_function_tools=2, effort_absent=1, thinking_off=1
sanitized /v1/responses: effort_high=1, rewrote_effort_high_to_xhigh=1, thinking_on=1
```

`effort_absent` means the client sent no `effort` -- see reason 1 above. It is what you
will see from a default Codex install.

## Non-Qwen models

`enable_thinking` is a Qwen-family convention. Templates that do not declare the variable
ignore it harmlessly (vLLM filters unknown kwargs before rendering), so on those models
the `off` policy is a no-op and the effort remapping is the part that matters. See
[MODEL-COMPATIBILITY.md](MODEL-COMPATIBILITY.md).
