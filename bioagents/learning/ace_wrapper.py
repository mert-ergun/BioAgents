"""ACE wrapper for BioAgents - enables self-evolving capabilities."""

import os
import sys
from pathlib import Path

from bioagents.learning.ace_adapters import create_ace_client_from_bioagents_llm
from bioagents.learning.execution_tracker import ExecutionTracker
from bioagents.learning.playbook_converter import (
    load_xml_prompt,
    xml_to_playbook,
)
from bioagents.learning.playbook_manager import (
    load_playbook,
    save_playbook,
    update_bullet_counts,
)


def is_ace_enabled() -> bool:
    """
    Check if ACE is enabled via environment variable.

    Returns:
        True if ACE_ENABLED is set to 'true', '1', or 'yes'
    """
    enabled = os.getenv("ACE_ENABLED", "false").lower()
    return enabled in ("true", "1", "yes")


def _setup_ace_path() -> None:
    """Add ACE directory to sys.path for imports."""
    # Find ACE directory relative to BioAgents root
    # __file__ = bioagents/learning/ace_wrapper.py
    # .parent = bioagents/learning
    # .parent.parent = bioagents
    # .parent.parent.parent = BioAgents (root)
    bioagents_root = Path(__file__).parent.parent.parent
    ace_path = bioagents_root / "ace"

    if ace_path.exists() and str(ace_path) not in sys.path:
        sys.path.insert(0, str(ace_path))


class BioAgentsACE:
    """
    ACE wrapper for BioAgents agents.

    Enables self-evolving capabilities through Generator-Reflector-Curator cycle.
    """

    def __init__(
        self,
        agent_name: str,
        llm_provider,
        max_tokens: int = 4096,
        curator_frequency: int = 5,
        use_bulletpoint_analyzer: bool = False,
        initial_playbook: str | None = None,
    ):
        """
        Initialize ACE wrapper for an agent.

        Args:
            agent_name: Name of the agent (e.g., 'supervisor', 'coder')
            llm_provider: BioAgents LLM instance (LangChain BaseChatModel)
            max_tokens: Maximum tokens for LLM calls
            curator_frequency: How often to run curator (every N executions)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            initial_playbook: Initial playbook content (optional, loads from file if not provided)
        """
        if not is_ace_enabled():
            self._enabled = False
            return

        self._enabled = True
        self.agent_name = agent_name
        self.curator_frequency = curator_frequency
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.execution_count = 0

        # Setup ACE path and import components
        _setup_ace_path()

        try:
            # Lazy import ACE components (after path setup)
            from ace.core.curator import Curator
            from ace.core.generator import Generator
            from ace.core.reflector import Reflector

            # Create ACE clients from BioAgents LLM
            generator_client, api_provider, model_name = create_ace_client_from_bioagents_llm(
                llm_provider
            )

            # Initialize ACE components
            self.generator = Generator(generator_client, api_provider, model_name, max_tokens)
            self.reflector = Reflector(generator_client, api_provider, model_name, max_tokens)
            self.curator = Curator(generator_client, api_provider, model_name, max_tokens)

            # Initialize bulletpoint analyzer if requested
            self.bulletpoint_analyzer = None
            if use_bulletpoint_analyzer:
                try:
                    from ace.core.bulletpoint_analyzer import BulletpointAnalyzer

                    self.bulletpoint_analyzer = BulletpointAnalyzer(
                        generator_client, model_name, max_tokens
                    )
                except ImportError:
                    print("Warning: BulletpointAnalyzer not available")

            # Initialize execution tracker
            self.execution_tracker = ExecutionTracker()

            # Load or initialize playbook
            if initial_playbook:
                self.playbook = initial_playbook
            else:
                self.playbook = load_playbook(agent_name)
                if not self.playbook.strip():
                    # Convert XML prompt to playbook if playbook is empty
                    xml_prompt = load_xml_prompt(agent_name)
                    if xml_prompt:
                        self.playbook = xml_to_playbook(xml_prompt, agent_name)
                        save_playbook(agent_name, self.playbook)

            self.next_global_id = self._get_next_global_id()

        except ImportError as e:
            print(f"Warning: Failed to import ACE components: {e}")
            print("ACE functionality will be disabled.")
            self._enabled = False

    def is_enabled(self) -> bool:
        """Check if ACE is enabled."""
        return getattr(self, "_enabled", False)

    def _get_next_global_id(self) -> int:
        """Get next global ID for playbook bullets."""
        import re

        max_id = 0
        for line in self.playbook.split("\n"):
            match = re.search(r"\[([^\]]+)\]", line)
            if match:
                id_str = match.group(1)
                num_match = re.search(r"-(\d+)$", id_str)
                if num_match:
                    num = int(num_match.group(1))
                    max_id = max(max_id, num)

        return max_id + 1

    def track_execution(
        self,
        task: str,
        output: str,
        success: bool,
        error_message: str | None = None,
        instructions_used: list[str] | None = None,
    ) -> None:
        """
        Track an agent execution.

        Args:
            task: The task that was executed
            output: The output from the agent
            success: Whether the execution was successful
            error_message: Error message if execution failed
            instructions_used: List of instruction IDs used (if available)
        """
        if not self.is_enabled():
            return

        self.execution_tracker.track_execution(
            agent_name=self.agent_name,
            task=task,
            output=output,
            success=success,
            error_message=error_message,
            instructions_used=instructions_used or [],
        )

        self.execution_count += 1

    def evolve_cycle(
        self,
        task: str,
        agent_output: str,
        success: bool,
        ground_truth: str | None = None,
        use_json_mode: bool = False,
    ) -> None:
        """
        Run one Generator-Reflector-Curator cycle.

        Args:
            task: The task that was executed
            agent_output: The output from the agent
            success: Whether the execution was successful
            ground_truth: Ground truth answer (if available)
            use_json_mode: Whether to use JSON mode for LLM calls
        """
        if not self.is_enabled():
            return

        # Extract bullets used (simplified - would need actual tracking)
        bullets_used = self._extract_bullets_used()

        # REFLECTOR: Tag instructions as helpful/harmful
        reflection_content, bullet_tags, _ = self.reflector.reflect(
            question=task,
            reasoning_trace=agent_output,
            predicted_answer=agent_output,
            ground_truth=ground_truth,
            environment_feedback="Correct" if success else "Incorrect",
            bullets_used=bullets_used,
            use_ground_truth=ground_truth is not None,
            use_json_mode=use_json_mode,
            call_id=f"{self.agent_name}_reflect_{self.execution_count}",
        )

        # Update bullet counts
        if bullet_tags:
            self.playbook = update_bullet_counts(self.playbook, bullet_tags)
            save_playbook(self.agent_name, self.playbook)

        # CURATOR: Add new instructions (every N executions)
        if self.should_curate():
            self._curate_playbook(reflection_content, task, use_json_mode)

    def _extract_bullets_used(self) -> str:
        """Extract bullets that were used (simplified version)."""
        # In a full implementation, this would track which bullets were actually used
        # For now, return a subset of the playbook
        lines = self.playbook.split("\n")
        bullet_lines = [line for line in lines if line.strip().startswith("[")]
        return "\n".join(bullet_lines[:10])  # Return first 10 bullets as example

    def should_curate(self) -> bool:
        """
        Determine if curator should run.

        Returns:
            True if curator should run (every N executions)
        """
        if not self.is_enabled():
            return False

        return self.execution_count % self.curator_frequency == 0

    def _curate_playbook(
        self, reflection_content: str, question_context: str, use_json_mode: bool
    ) -> None:
        """Run curator to add new instructions to playbook."""
        if not self.is_enabled():
            return

        # Get playbook stats
        playbook_stats = {
            "num_bullets": len(
                [line for line in self.playbook.split("\n") if line.strip().startswith("[")]
            ),
            "num_sections": len(
                [line for line in self.playbook.split("\n") if line.strip().startswith("##")]
            ),
        }

        token_budget = int(os.getenv("ACE_PLAYBOOK_TOKEN_BUDGET", "80000"))

        # Run curator
        updated_playbook, next_id, _operations, _ = self.curator.curate(
            current_playbook=self.playbook,
            recent_reflection=reflection_content,
            question_context=question_context,
            current_step=self.execution_count,
            total_samples=1000,  # Placeholder
            token_budget=token_budget,
            playbook_stats=playbook_stats,
            use_ground_truth=True,
            use_json_mode=use_json_mode,
            call_id=f"{self.agent_name}_curate_{self.execution_count}",
            next_global_id=self.next_global_id,
        )

        self.playbook = updated_playbook
        self.next_global_id = next_id

        # Run bulletpoint analyzer if enabled
        if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
            self.playbook = self.bulletpoint_analyzer.analyze(
                playbook=self.playbook,
                threshold=0.90,
                merge=True,
            )

        # Save updated playbook
        save_playbook(self.agent_name, self.playbook)


def create_ace_wrapper_if_enabled(
    agent_name: str,
    llm_provider,
    max_tokens: int = 4096,
    curator_frequency: int | None = None,
    use_bulletpoint_analyzer: bool = False,
    initial_playbook: str | None = None,
) -> BioAgentsACE | None:
    """
    Create ACE wrapper only if enabled.

    Args:
        agent_name: Name of the agent
        llm_provider: BioAgents LLM instance
        max_tokens: Maximum tokens for LLM calls
        curator_frequency: How often to run curator (defaults to env var or 5)
        use_bulletpoint_analyzer: Whether to use bulletpoint analyzer
        initial_playbook: Initial playbook content (optional)

    Returns:
        BioAgentsACE instance if enabled, None otherwise
    """
    if not is_ace_enabled():
        return None

    if curator_frequency is None:
        curator_frequency = int(os.getenv("ACE_CURATOR_FREQUENCY", "5"))

    return BioAgentsACE(
        agent_name=agent_name,
        llm_provider=llm_provider,
        max_tokens=max_tokens,
        curator_frequency=curator_frequency,
        use_bulletpoint_analyzer=use_bulletpoint_analyzer,
        initial_playbook=initial_playbook,
    )
