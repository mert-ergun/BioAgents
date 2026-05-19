"""Adapters for integrating external libraries with BioAgents."""

import hashlib
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from smolagents import ChatMessage, MessageRole, Model

logger = logging.getLogger(__name__)

# smolagents CodeAgent passes the markdown closing fence as a stop sequence. That halts
# generation at the *first* "```" in the stream — including inside Python docstrings or
# strings — truncating code and causing parse/retry loops and huge malformed outputs.
_CODE_AGENT_FENCE_STOPS = frozenset({"```", "\n```", "```\n"})

_MAX_DUPLICATE_CODE = 2
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _sanitize_codeagent_stop_sequences(stop_sequences: list[str] | None) -> list[str] | None:
    if not stop_sequences:
        return stop_sequences
    filtered = [s for s in stop_sequences if s not in _CODE_AGENT_FENCE_STOPS]
    return filtered if filtered else None


def _is_nonproductive(content: str | None) -> bool:
    """Detect truly empty LLM responses.

    Only catches None, whitespace-only, or completely empty content.
    Non-empty text without a code block is treated as a valid summary
    or final answer from the model (e.g. after completing its task).
    """
    if content is None:
        return True
    if not isinstance(content, str):
        return False
    return not content.strip()


def _fix_code_fences(content: str | None) -> str | None:
    """Fix LLM output missing proper ```python``` code fences.

    Some models (notably Gemini) output raw code starting with ``python\\n``
    without the triple-backtick fences that smolagents' parser requires.
    This wraps such content in proper fences so the parser can extract it.
    """
    if content is None or not isinstance(content, str):
        return content
    stripped = content.strip()
    # Already has proper fences — nothing to do
    if "```python" in stripped:
        return content
    # Gemini sometimes outputs: python\n<code here>\n```
    # or just: python\n<code here> (no closing fence either)
    if stripped.startswith("python\n") or stripped.startswith("python\r\n"):
        code = stripped[len("python\n") :]
        return f"Thoughts: Executing the generated code.\n```python\n{code}\n```"
    return content


class LangChainModelAdapter(Model):
    """Adapter to use LangChain models with smolagents."""

    def __init__(self, langchain_model: BaseChatModel, **kwargs):
        super().__init__(**kwargs)
        self.langchain_model = langchain_model
        self._last_code_hash: str | None = None
        self._duplicate_code_count = 0

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatMessage:
        langchain_messages = self._convert_messages(messages)

        stop_sequences = _sanitize_codeagent_stop_sequences(stop_sequences)

        if stop_sequences:
            try:
                model = self.langchain_model.bind(stop=stop_sequences)
                response = model.invoke(langchain_messages, **kwargs)
            except Exception:
                response = self.langchain_model.invoke(langchain_messages, **kwargs)
        else:
            response = self.langchain_model.invoke(langchain_messages, **kwargs)

        content: str | list[str | dict] | None = response.content
        if isinstance(content, list):
            text_content = ""
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_content += part["text"]
                elif isinstance(part, str):
                    text_content += part
            content = text_content

        content = _fix_code_fences(content)

        # Log raw LLM output for debugging empty-response loops
        logger.debug(
            "LangChainModelAdapter.generate raw output: "
            "content_type=%s, content_len=%d, "
            "content_preview=%r, "
            "has_code_block=%s",
            type(content).__name__,
            len(content) if content else 0,
            content[:500] if content else None,
            "```python" in content if content else False,
        )

        # Detect repeated identical code generation (the LLM keeps generating the
        # same broken code without learning from execution errors).  Break the loop
        # by injecting a final_answer after N consecutive duplicates.
        code_match = _CODE_BLOCK_RE.search(content) if content else None
        if code_match and "final_answer" not in code_match.group(1):
            code_hash = hashlib.md5(
                code_match.group(1).strip().encode(), usedforsecurity=False
            ).hexdigest()
            if code_hash == self._last_code_hash:
                self._duplicate_code_count += 1
            else:
                self._last_code_hash = code_hash
                self._duplicate_code_count = 0

            if self._duplicate_code_count >= _MAX_DUPLICATE_CODE:
                content = (
                    "Thoughts: The same code was generated repeatedly and keeps "
                    "failing. Stopping retries to avoid an infinite loop.\n"
                    "```python\n"
                    'final_answer("Code execution repeatedly failed with the same '
                    'error. A different approach is needed.")\n'
                    "```"
                )
                self._duplicate_code_count = 0
                self._last_code_hash = None

        if _is_nonproductive(content):
            # Truly empty response — force termination
            content = (
                "Thoughts: The model returned an empty response. Finishing.\n"
                "```python\n"
                'final_answer("Task completed.")\n'
                "```"
            )
        elif content is not None and "```python" not in content:
            # Non-empty text without a code block: the model is giving a summary
            # or final answer after completing its task. Wrap it in final_answer()
            # so smolagents recognizes it as task completion.
            answer = content.replace('"', '\\"').strip()
            content = (
                "Thoughts: Task completed, providing final answer.\n"
                "```python\n"
                f'final_answer("{answer}")\n'
                "```"
            )

        return ChatMessage(role=MessageRole.ASSISTANT, content=content, raw=response)

    def _convert_messages(self, messages: list[ChatMessage]) -> list[BaseMessage]:
        lc_messages: list[BaseMessage] = []
        # Check if model is Gemini (ChatGoogleGenerativeAI) - it may not support SystemMessage
        is_gemini = (
            "google" in str(type(self.langchain_model)).lower()
            or "gemini" in str(type(self.langchain_model)).lower()
        )

        for msg in messages:
            content = msg.content
            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=content))
            elif msg.role == MessageRole.SYSTEM:
                # Gemini may not support SystemMessage, convert to HumanMessage
                if is_gemini:
                    # Prepend system message as a user message for Gemini
                    lc_messages.append(HumanMessage(content=f"System: {content}"))
                else:
                    lc_messages.append(SystemMessage(content=content))
        return lc_messages
