"""Gemini-specific adapter for ACE framework.

Since Gemini API is not OpenAI-compatible, we need a custom adapter
that wraps LangChain's ChatGoogleGenerativeAI to work with ACE.
"""

from langchain_core.language_models import BaseChatModel


class GeminiACEAdapter:
    """
    Adapter to use Gemini (LangChain) with ACE framework.

    This adapter wraps LangChain's ChatGoogleGenerativeAI to provide
    an OpenAI-compatible interface for ACE's timed_llm_call function.
    """

    def __init__(self, langchain_llm: BaseChatModel):
        """
        Initialize Gemini adapter.

        Args:
            langchain_llm: LangChain LLM instance (ChatGoogleGenerativeAI)
        """
        self.llm = langchain_llm

    class Chat:
        """Mock chat object for OpenAI-compatible interface."""

        def __init__(self, adapter):
            self.adapter = adapter
            self.completions = self.Completions(adapter)

        class Completions:
            """Mock completions object for OpenAI-compatible interface."""

            def __init__(self, adapter):
                self.adapter = adapter

            def create(
                self,
                model: str,  # noqa: ARG002
                messages: list,
                temperature: float = 0.0,  # noqa: ARG002
                max_tokens: int | None = None,  # noqa: ARG002
                max_completion_tokens: int | None = None,  # noqa: ARG002
                response_format: dict | None = None,
                **kwargs,  # noqa: ARG002
            ) -> "MockResponse":
                """
                Create a completion using Gemini via LangChain.

                Args:
                    model: Model name (ignored, uses adapter's LLM)
                    messages: List of message dicts with 'role' and 'content'
                    temperature: Temperature for generation
                    max_tokens: Max tokens (ignored for Gemini)
                    max_completion_tokens: Max completion tokens
                    response_format: JSON mode format (if needed)
                    **kwargs: Additional parameters

                Returns:
                    MockResponse object with OpenAI-compatible structure
                """
                from langchain_core.messages import AIMessage, HumanMessage

                # Convert messages to LangChain format
                lc_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    elif role == "system":
                        lc_messages.append(HumanMessage(content=f"System: {content}"))

                # Invoke LangChain LLM
                response = self.adapter.llm.invoke(lc_messages)

                # Extract content
                content = response.content if hasattr(response, "content") else str(response)

                if response_format and response_format.get("type") == "json_object":
                    import json
                    import re

                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        try:
                            json.loads(json_match.group(0))
                            content = json_match.group(0)
                        except json.JSONDecodeError:
                            pass

                # Create mock response
                return MockResponse(content)

    @property
    def chat(self):
        """Return a chat object with completions for OpenAI-compatible interface."""
        return self.Chat(self)


class MockResponse:
    """Mock OpenAI response object for compatibility."""

    def __init__(self, content: str):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(len(content) // 4, len(content) // 4)


class MockChoice:
    """Mock OpenAI choice object."""

    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock OpenAI message object."""

    def __init__(self, content: str):
        self.content = content


class MockUsage:
    """Mock OpenAI usage object."""

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


def create_gemini_ace_client(langchain_llm: BaseChatModel):
    """
    Create Gemini adapter for ACE framework.

    Args:
        langchain_llm: LangChain LLM instance (ChatGoogleGenerativeAI)

    Returns:
        GeminiACEAdapter instance that provides OpenAI-compatible interface
    """
    return GeminiACEAdapter(langchain_llm)
