"""Unit tests for the request rewriting.

These need no GPU, no vLLM and no network -- which is the practical argument for doing
this in an adapter rather than as a patch against vLLM's source. Patch-based fixes can
only be tested by booting a 27B model on an 80GB card; these run in milliseconds in CI.

Run: python -m pytest tests/ -q     (or: python tests/test_sanitize.py)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_vllm_adapter.sanitize import sanitize_request  # noqa: E402

USER = {"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "hi"}]}


class TestSanitize(unittest.TestCase):

    def test_passthrough_is_byte_identical(self):
        body = {"model": "m", "input": [USER]}
        new, stats = sanitize_request(body)
        self.assertEqual(stats, {})
        self.assertIs(new, body, "an untouched request must not be rebuilt")

    def test_drops_additional_tools_item(self):
        body = {"model": "m", "input": [
            {"type": "additional_tools", "role": "developer", "tools": [{"name": "x"}]},
            USER]}
        new, stats = sanitize_request(body)
        self.assertEqual(new["input"], [USER])
        self.assertEqual(stats["dropped_additional_tools"], 1)

    def test_drops_compaction_item(self):
        body = {"model": "m", "input": [
            {"type": "compaction", "id": "cp_1", "encrypted_content": "opaque"}, USER]}
        new, stats = sanitize_request(body)
        self.assertEqual(new["input"], [USER])
        self.assertEqual(stats["dropped_compaction"], 1)

    def test_drops_custom_tool_call_records(self):
        # These dominate real recorded history and each one is a 500 on the way in.
        body = {"model": "m", "input": [
            {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch",
             "input": "*** Begin Patch"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
            USER]}
        new, stats = sanitize_request(body)
        self.assertEqual(new["input"], [USER])
        self.assertEqual(stats["dropped_custom_tool_call"], 1)
        self.assertEqual(stats["dropped_custom_tool_call_output"], 1)

    def test_keeps_ordinary_function_call_history(self):
        # The counterpart to the test above: ordinary function calls must survive, or
        # resume would lose all real tool history.
        fc = {"type": "function_call", "call_id": "c1", "name": "get_weather",
              "arguments": "{}"}
        fo = {"type": "function_call_output", "call_id": "c1", "output": "sunny"}
        body = {"model": "m", "input": [fc, fo, USER]}
        new, stats = sanitize_request(body)
        self.assertEqual(stats, {})
        self.assertIs(new, body)

    def test_strips_encrypted_content_but_keeps_readable_reasoning(self):
        body = {"model": "m", "input": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
             "summary": [{"type": "summary_text", "text": "thought about it"}]},
            USER]}
        new, stats = sanitize_request(body)
        kept = new["input"][0]
        self.assertNotIn("encrypted_content", kept)
        self.assertEqual(kept["summary"][0]["text"], "thought about it")
        self.assertEqual(stats["stripped_encrypted_reasoning"], 1)

    def test_drops_reasoning_that_is_only_ciphertext(self):
        body = {"model": "m", "input": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
             "summary": []},
            USER]}
        new, stats = sanitize_request(body)
        self.assertEqual(new["input"], [USER])
        self.assertEqual(stats["dropped_empty_reasoning"], 1)

    def test_drops_non_function_tools_only(self):
        body = {"model": "m", "input": [USER], "tools": [
            {"type": "custom", "name": "apply_patch"},
            {"type": "web_search"},
            {"type": "function", "name": "get_weather", "parameters": {}},
        ]}
        new, stats = sanitize_request(body)
        self.assertEqual([t["type"] for t in new["tools"]], ["function"])
        self.assertEqual(stats["dropped_non_function_tools"], 2)

    def test_tool_without_type_is_kept(self):
        # Some clients omit "type" and mean a function tool. Dropping those would break
        # ordinary tool calling, so absence of the key must not be treated as non-function.
        body = {"model": "m", "input": [USER],
                "tools": [{"name": "get_weather", "parameters": {}}]}
        new, stats = sanitize_request(body)
        self.assertEqual(len(new["tools"]), 1)
        self.assertEqual(stats, {})

    def test_does_not_mutate_caller_input(self):
        original = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
                    "summary": [{"type": "summary_text", "text": "t"}]}
        body = {"model": "m", "input": [original, USER]}
        sanitize_request(body)
        self.assertIn("encrypted_content", original,
                      "sanitizing must not mutate the caller's objects in place")

    def test_non_dict_body_is_returned_untouched(self):
        for junk in ([], "string", None, 7):
            new, stats = sanitize_request(junk)
            self.assertIs(new, junk)
            self.assertEqual(stats, {})

    def test_unknown_item_types_are_preserved(self):
        # Only the two known-unrepresentable types are removed. Anything else is left
        # alone so a future Codex item type is not silently swallowed.
        body = {"model": "m", "input": [{"type": "some_future_thing", "role": "user"}]}
        new, stats = sanitize_request(body)
        self.assertEqual(stats, {})
        self.assertIs(new, body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
