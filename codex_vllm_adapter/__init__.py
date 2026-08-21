"""Make stock vLLM accept OpenAI Codex CLI traffic, without patching vLLM."""

from .sanitize import sanitize_request
from .thinking import apply_thinking_policy

__all__ = ["sanitize_request", "apply_thinking_policy"]
__version__ = "0.2.0"
