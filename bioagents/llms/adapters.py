"""Adapters for integrating external libraries with BioAgents."""

from typing import List, Optional
from smolagents import Model, ChatMessage, MessageRole
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel


class LangChainModelAdapter(Model):
    """Adapter to use LangChain models with smolagents."""
    
    def __init__(self, langchain_model: BaseChatModel, **kwargs):
        super().__init__(**kwargs)
        self.langchain_model = langchain_model

    def generate(
        self,
        messages: List[ChatMessage],
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatMessage:
        langchain_messages = self._convert_messages(messages)
        
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
                if isinstance(part, dict) and 'text' in part:
                    text_content += part['text']
                elif isinstance(part, str):
                    text_content += part
            content = text_content

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            raw=response
        )

    def _convert_messages(self, messages: List[ChatMessage]) -> List[BaseMessage]:
        lc_messages = []
        # Check if model is Gemini (ChatGoogleGenerativeAI) - it may not support SystemMessage
        is_gemini = "google" in str(type(self.langchain_model)).lower() or "gemini" in str(type(self.langchain_model)).lower()
        
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

