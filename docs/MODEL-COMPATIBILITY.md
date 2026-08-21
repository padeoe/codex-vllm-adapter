# Which models does this work with?

**Verified end to end: Qwen3.8-27B-FP8.** Everything measured in this repository ran on
that model, on one H800, under `vllm/vllm-openai:v0.27.1`.

The adapter itself contains no model-specific logic -- it edits a Responses API request
and never looks at the model. But "the adapter is model-agnostic" is not the same claim as
"Codex works with your model", and the gap between those two is worth being precise about,
because most of the ways this fails have nothing to do with the adapter.

## Decide in five minutes

Three checks, in the order they will bite you.

**1. Can vLLM parse tool calls from your model?** Codex is a tool-calling client; a model
that cannot call tools cannot do anything useful in it, no matter how good it is.

```bash
ls /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/   # inside the vLLM image
```

vLLM 0.27.1 ships ~40 of them (`hermes`, `llama_*`, `mistral`, `deepseekv3*`, `glm47_moe`,
`kimi_k2`, `minimax_m2`, `granite*`, `gptoss`, `qwen3_engine`, `seed_oss_engine`,
`step3`, `pythonic`, ...). Serve with `--enable-auto-tool-choice --tool-call-parser NAME`.
If nothing matches your model, stop here.

**2. Does its chat template accept a `developer` role?** The Responses API has no `system`
role. vLLM renders `instructions` into a system message, and Codex additionally sends
`developer` messages.

```bash
grep -c developer /path/to/model/chat_template.jinja
```

Zero means you need `tools/patch_chat_template.py` (written for Qwen's template; the same
two-line idea transfers, the anchors do not). Harmony models such as gpt-oss have
`developer` as a first-class role and need nothing.

**3. Can thinking be switched off?** Only relevant for reasoning models:

```bash
grep -c enable_thinking /path/to/model/chat_template.jinja
```

Non-zero means `--thinking off` works (Qwen-family convention). Zero means the template
ignores the variable -- harmlessly, vLLM filters kwargs a template does not declare -- and
the useful knob is effort *remapping* instead. See [THINKING.md](THINKING.md).

Then run the real test, which is the only one that counts:

```bash
codex exec "fix the bug in calc.py, run it, tell me what it prints"
```

## What each part of the stack requires

| Requirement | Whose problem | If unmet |
| --- | --- | --- |
| vLLM Responses API (`/v1/responses`) | server | Codex ≥ 0.148 needs `wire_api = "responses"` |
| A vLLM tool-call parser for the model | model + server | no tool calls; Codex can talk but not work |
| Template accepts `developer` / leading system | model template | `Unexpected message role.` |
| Effort names the template knows | model template | 400 on `high` / `minimal` / `xhigh` |
| Genuine agentic ability, long context | the model | it "works" and still fails your tasks |
| Request shapes vLLM cannot represent | **the adapter** | the 400s and 500s this project exists for |

Only the last row is ours. That is the honest scope: the adapter removes a fixed class of
protocol incompatibilities that are the same for every model, because they come from
Codex, not from the model.

## Likely to work

Reasoning: the failures the adapter fixes are client-side artefacts, so any model that
clears the three checks should behave like Qwen3.8 did. **This is inference, not
measurement** -- we tested one model per this list, and it is the first one.

* **Qwen3 / Qwen3.5 / Qwen3.8, and Coder variants** -- verified on 3.8; same template
  family, same `enable_thinking` convention, `qwen3_coder` / `qwen3_engine` parsers.
* **DeepSeek-V3 / V3.1 / V3.2**, **GLM-4.x**, **Kimi K2**, **MiniMax-M2** -- large
  tool-trained models with dedicated parsers in 0.27.1. Expect to need the developer-role
  check; thinking control varies by template.
* **Llama 3.x / 4, Mistral Large, Granite 4** -- solid function calling with shipped
  parsers, no thinking machinery to fight.

## Likely not to work

* **Base / non-instruct checkpoints** and any model without a tool-call parser. Hard no.
* **Small models (≲7B).** They will pass every protocol check here and still fail as
  agents: Codex sends a ~20k-character system prompt plus a nine-tool schema before your
  task even starts, and holding that while planning multi-step edits is where small models
  come apart. Protocol compatibility is not capability.
* **Always-reasoning models with no "off" switch** (DeepSeek-R1 and its distills) -- usable,
  but every turn pays full reasoning cost and `--thinking off` cannot help; the model has
  no non-thinking mode to select.
* **Short-context models.** Codex's own preamble is large; below ~32k you will be
  compacting constantly.

## The one other family we actually tried: gpt-oss-20b

Run as a spot check on the same box (vLLM 0.27.1, `--gpu-memory-utilization 0.24`,
32k context). It is the useful counter-example, and it fails for a reason that is not the
adapter's:

| Step | Result |
| --- | --- |
| Chat template | **No patch needed** -- `developer` is a first-class Harmony role |
| `enable_thinking` | **Ignored** -- reasoning-only model, no off switch. Harmless no-op |
| Effort names | Accepts `low`/`medium`/`high` only. **`xhigh`, `minimal` and `none` are 400s** |
| Codex, direct to vLLM | **Fails**: `400 tool type namespace not supported` |
| Codex, through the adapter | 400 gone (the `namespace` and `web_search` tools are dropped)... |
| ...then | **500 `HarmonyError: Unexpected token 12606 while expecting start token 200006`** |

We isolated that last one rather than guessing:

| Probe | Result |
| --- | --- |
| Plain question, no tools | OK |
| Long prompt (6.8k chars), no tools | OK, coherent |
| Tool declared, prompt that needs no tool | OK |
| **Prompt that induces a tool call, tool declared** | **500 HarmonyError** |

The crash arrives exactly when the model emits a tool call, with or without Codex in the
picture, streaming or not. It is inside vLLM's harmony parser, upstream of anything the
adapter touches. **Conclusion: gpt-oss on vLLM 0.27.1 is not usable with Codex in this
environment**, and no request rewriting can change that.

Two things transfer from it regardless:

* the adapter's tool filtering was **necessary** for a second, unrelated model family --
  Codex declares a `namespace`-typed tool that vLLM rejects outright;
* on Harmony models the thinking policy must remap efforts rather than disable them,
  because `none` is a 400 there. See
  [`examples/thinking-gptoss.toml`](../examples/thinking-gptoss.toml).

## Please report what you find

If you run this against another model, an issue saying which model, which vLLM version,
and which of the three checks it failed is genuinely useful -- this page should be a table
of measurements, and right now it is mostly a table of reasoning.
