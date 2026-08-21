"""Tests for the thinking policy. No GPU, no vLLM, no network."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codex_vllm_adapter.thinking import (  # noqa: E402
    DEFAULT_POLICY,
    PASSTHROUGH_POLICY,
    PolicyError,
    apply_thinking_policy,
    load_policy,
    normalize_policy,
)


def req(effort=None, **extra):
    body = {"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}]}
    if effort is not None:
        body["reasoning"] = {"effort": effort, "summary": "auto"}
    body.update(extra)
    return body


class TestApply(unittest.TestCase):
    def test_default_policy_disables_thinking_at_every_level(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            new, stats = apply_thinking_policy(req(effort), DEFAULT_POLICY)
            self.assertIs(new["chat_template_kwargs"]["enable_thinking"], False, effort)
            self.assertEqual(stats.get("thinking_off"), 1, effort)
            self.assertEqual(stats.get("effort_%s" % effort), 1, effort)

    def test_effort_is_left_alone_when_thinking_is_off(self):
        # The template skips the whole reasoning block, so there is nothing to clamp,
        # and forwarding the client's real effort keeps the request honest.
        new, _ = apply_thinking_policy(req("xhigh"), DEFAULT_POLICY)
        self.assertEqual(new["reasoning"]["effort"], "xhigh")

    def test_passthrough_policy_remaps_high_to_a_supported_level(self):
        # Qwen3.5/3.8 raises a Jinja error on 'high'; it accepts only low/medium/xhigh.
        new, stats = apply_thinking_policy(req("high"), PASSTHROUGH_POLICY)
        self.assertIs(new["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(new["reasoning"]["effort"], "xhigh")
        self.assertEqual(stats.get("rewrote_effort_high_to_xhigh"), 1)

    def test_remap_preserves_other_reasoning_fields(self):
        new, _ = apply_thinking_policy(req("high"), PASSTHROUGH_POLICY)
        self.assertEqual(new["reasoning"]["summary"], "auto")

    def test_stats_report_the_effort_the_client_asked_for(self):
        _, stats = apply_thinking_policy(req(), DEFAULT_POLICY)
        self.assertEqual(stats.get("effort_absent"), 1)

    def test_keep_touches_nothing(self):
        body = req("high")
        new, stats = apply_thinking_policy(body, {"default": "keep", "high": "keep"})
        self.assertIs(new, body)
        self.assertEqual(stats, {})

    def test_policy_none_is_a_full_bypass(self):
        body = req("high")
        new, stats = apply_thinking_policy(body, None)
        self.assertIs(new, body)
        self.assertEqual(stats, {})

    def test_request_without_reasoning_uses_the_default_entry(self):
        new, _ = apply_thinking_policy(req(), {"default": "medium"})
        self.assertIs(new["chat_template_kwargs"]["enable_thinking"], True)
        # No `reasoning` object to rewrite, so the effort goes to the template directly.
        # vLLM drops its own reasoning_effort=None when merging, so this one survives.
        self.assertEqual(new["chat_template_kwargs"]["reasoning_effort"], "medium")
        self.assertNotIn("reasoning", new)

    def test_unlisted_effort_falls_back_to_default(self):
        new, _ = apply_thinking_policy(req("max"), {"default": "off"})
        self.assertIs(new["chat_template_kwargs"]["enable_thinking"], False)

    def test_effort_matching_is_case_insensitive(self):
        new, _ = apply_thinking_policy(req("HIGH"), {"default": "off", "high": "on"})
        self.assertIs(new["chat_template_kwargs"]["enable_thinking"], True)

    def test_existing_chat_template_kwargs_are_preserved(self):
        new, _ = apply_thinking_policy(
            req("low", chat_template_kwargs={"custom": 1}), DEFAULT_POLICY)
        self.assertEqual(new["chat_template_kwargs"]["custom"], 1)
        self.assertIs(new["chat_template_kwargs"]["enable_thinking"], False)

    def test_does_not_mutate_caller_input(self):
        body = req("high", chat_template_kwargs={"custom": 1})
        before = json.dumps(body, sort_keys=True)
        apply_thinking_policy(body, PASSTHROUGH_POLICY)
        self.assertEqual(json.dumps(body, sort_keys=True), before)

    def test_non_dict_body_is_returned_untouched(self):
        self.assertEqual(apply_thinking_policy([1, 2], DEFAULT_POLICY), ([1, 2], {}))


class TestPolicyParsing(unittest.TestCase):
    def test_section_wrapper_is_optional(self):
        self.assertEqual(
            normalize_policy({"thinking": {"low": "off"}}),
            normalize_policy({"low": "off"}))

    def test_default_is_supplied_when_missing(self):
        self.assertEqual(normalize_policy({"low": "on"})["default"], "off")

    def test_unknown_effort_name_is_rejected(self):
        with self.assertRaises(PolicyError) as cm:
            normalize_policy({"enormous": "off"})
        self.assertIn("enormous", str(cm.exception))

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(PolicyError) as cm:
            normalize_policy({"low": "maybe"})
        self.assertIn("maybe", str(cm.exception))

    def test_non_string_action_is_rejected(self):
        with self.assertRaises(PolicyError):
            normalize_policy({"low": True})

    def test_loads_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"thinking": {"default": "off", "high": "xhigh"}}, fh)
        try:
            self.assertEqual(load_policy(fh.name)["high"], "xhigh")
        finally:
            os.unlink(fh.name)

    @unittest.skipIf(sys.version_info < (3, 11), "tomllib is 3.11+")
    def test_loads_toml(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write('[thinking]\ndefault = "off"\nhigh = "xhigh"\n')
        try:
            self.assertEqual(load_policy(fh.name)["high"], "xhigh")
        finally:
            os.unlink(fh.name)

    def test_shipped_examples_are_valid(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        loaded = load_policy(os.path.join(here, "examples", "thinking.json"))
        self.assertEqual(loaded["default"], "off")
        if sys.version_info >= (3, 11):
            self.assertEqual(
                load_policy(os.path.join(here, "examples", "thinking.toml")), loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
