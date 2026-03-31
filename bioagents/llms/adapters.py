"""Adapters for integrating external libraries with BioAgents."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from smolagents import ChatMessage, MessageRole, Model

# smolagents CodeAgent passes the markdown closing fence as a stop sequence. That halts
# generation at the *first* "```" in the stream — including inside Python docstrings or
# strings — truncating code and causing parse/retry loops and huge malformed outputs.
_CODE_AGENT_FENCE_STOPS = frozenset({"```", "\n```", "```\n"})


def _sanitize_codeagent_stop_sequences(stop_sequences: list[str] | None) -> list[str] | None:
    if not stop_sequences:
        return stop_sequences
    filtered = [s for s in stop_sequences if s not in _CODE_AGENT_FENCE_STOPS]
    return filtered if filtered else None


_MAX_NONPRODUCTIVE = 3


def _is_nonproductive(content: str | None) -> bool:
    """Detect empty or essentially-empty LLM responses.

    Catches truly empty content, whitespace-only, and responses that are just
    echoed fallback text without a valid python code block.
    """
    if content is None:
        return True
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return True
    if "```python" in stripped:
        return False
    # Short responses without a code block are non-productive
    if len(stripped) < 200:
        return True
    return False


class LangChainModelAdapter(Model):
    """Adapter to use LangChain models with smolagents."""

    def __init__(self, langchain_model: BaseChatModel, **kwargs):
        super().__init__(**kwargs)
        self.langchain_model = langchain_model
        self._nonproductive_count = 0

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

        content = response.content
        if isinstance(content, list):
            text_content = ""
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_content += part["text"]
                elif isinstance(part, str):
                    text_content += part
            content = text_content

        if _is_nonproductive(content):
            self._nonproductive_count += 1
            if self._nonproductive_count >= _MAX_NONPRODUCTIVE:
                content = (
                    "Thoughts: The model returned empty/non-productive content "
                    f"{self._nonproductive_count} times. Finishing with available results.\n"
                    "```python\n"
                    'final_answer("Task could not be completed: '
                    'the model returned empty responses repeatedly.")\n'
                    "```"
                )
            else:
                content = (
                    "Thoughts: The previous model call returned empty content. "
                    "I will retry with a simpler approach.\n"
                    "```python\n"
                    'print("Previous LLM call returned empty. Reassessing the task.")\n'
                    "```"
                )
        else:
            self._nonproductive_count = 0

        return ChatMessage(role=MessageRole.ASSISTANT, content=content, raw=response)

    def _convert_messages(self, messages: list[ChatMessage]) -> list[BaseMessage]:
        lc_messages = []
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
