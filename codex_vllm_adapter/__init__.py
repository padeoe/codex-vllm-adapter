"""Make stock vLLM accept OpenAI Codex CLI traffic, without patching vLLM."""

from .sanitize import sanitize_request

__all__ = ["sanitize_request"]
__version__ = "0.1.0"
