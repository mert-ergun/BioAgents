"""
XML-based prompt system for BioAgents.

This module provides structured, maintainable prompts for all agents
in the multi-agent system.
"""

from bioagents.prompts.prompt_loader import (
    PromptLoader,
    get_prompt_llm_model,
    get_prompt_llm_models,
    load_prompt,
)

__all__ = ["PromptLoader", "get_prompt_llm_model", "get_prompt_llm_models", "load_prompt"]
