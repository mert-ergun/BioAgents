"""Tools for the Tool Builder Agent for discovering, creating, and managing custom tools."""

import inspect
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import time
from datetime import datetime

from langchain_core.tools import tool

from bioagents.limits import (
    TOOL_WRAPPER_MAX_ATTEMPTS,
    TOOL_WRAPPER_RETRY_BASE_DELAY_SEC,
    TOOL_WRAPPER_RETRY_MAX_DELAY_SEC,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.llms.timeout_llm import remaining_workflow_seconds
from bioagents.sandbox.dependency_events import (
    record_blocked_install,
    record_dependency_event,
    run_install_command,
    track_dependencies,
)
from bioagents.sandbox.pip_guard import filter_install_requests
from bioagents.tools.code_api_check import check_generated_code
from bioagents.tools.dependency_inference import extract_system_dependencies_from_code
from bioagents.tools.result_status import tool_result_status
from bioagents.tools.tool_registry import (
    ToolDefinition,
    ToolParameter,
    get_registry,
)

logger = logging.getLogger(__name__)


class _EmptyWrapperResponse(ValueError):
    pass


def _invoke_wrapper(llm, prompt):
    attempts = TOOL_WRAPPER_MAX_ATTEMPTS
    for attempt in range(attempts):
        remaining = remaining_workflow_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Workflow deadline reached before wrapper generation")
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if not isinstance(content, str) or not content.strip():
                raise _EmptyWrapperResponse("Wrapper generation returned empty content")
            return content
        except Exception as exc:
            retryable = isinstance(
                exc, (TimeoutError, ConnectionError, _EmptyWrapperResponse)
            ) or getattr(exc, "status_code", None) in {429, 500, 502, 503, 504}
            remaining = remaining_workflow_seconds()
            if (
                not retryable
                or attempt + 1 == attempts
                or (remaining is not None and remaining < 1)
            ):
                raise
            delay = min(
                TOOL_WRAPPER_RETRY_BASE_DELAY_SEC * 2**attempt, TOOL_WRAPPER_RETRY_MAX_DELAY_SEC
            )
            if remaining is not None:
                delay = min(delay, max(0.0, remaining - 1))
            logger.warning(
                "Retrying wrapper generation after %s (%d/%d)",
                type(exc).__name__,
                attempt + 1,
                attempts,
            )
            time.sleep(delay)


def _run_pip_install_streaming(pip_cmd, timeout=180):
    """Run pip with a real bounded communicate deadline and audited outcome."""
    result = run_install_command(
        pip_cmd, reason="missing_python_dependency", capture_output=True, text=True, timeout=timeout
    )
    for line in (result.stdout or "").splitlines():
        logger.info("pip: %s", line)
    return result


_IMPORT_ALIASES = {
    "scikit-learn": "sklearn",
    "biopython": "Bio",
    "pillow": "PIL",
    "python-levenshtein": "Levenshtein",
    "pyyaml": "yaml",
    "openbabel-wheel": "openbabel",
}


_EXECUTABLE_NAME = re.compile(r"[a-z0-9][a-z0-9_.+-]*")
_INSTALL_FAILED = "install_failed"


def _screen_system_dependencies(
    dependencies: list, previous: dict[str, str] | None = None
) -> tuple[list[str], dict[str, str]]:
    """Keep plausible executables; reject prose such as "Directory" from error text.

    Lowercase executable-like names are kept even when missing so they can be
    installed. Any other spelling is kept only if it already resolves on PATH
    (e.g. ``Rscript``), because installers cannot be expected to provide it.
    A recorded install failure is preserved so it is not retried on every call.
    """
    previous = previous or {}
    kept: list[str] = []
    status: dict[str, str] = {}
    for dep in dependencies:
        name = dep.strip() if isinstance(dep, str) else ""
        if not name or name in status:
            continue
        if shutil.which(name):
            status[name] = "available"
        elif not _EXECUTABLE_NAME.fullmatch(name):
            status[name] = "rejected: not an executable-like name and not found on PATH"
            continue
        elif previous.get(name) == _INSTALL_FAILED:
            status[name] = _INSTALL_FAILED
        else:
            status[name] = "missing"
        kept.append(name)
    return kept, status


def _persist_tool(registry, tool_def: ToolDefinition) -> None:
    try:
        registry.register_tool(tool_def, overwrite=True)
    except Exception as exc:
        logger.warning("Could not persist registry entry for %s: %s", tool_def.name, exc)


def _failure(phase: str, reasons: list[str]) -> dict[str, str]:
    return {
        "phase": phase,
        "reason": "; ".join(reasons)[:2000],
        "at": datetime.now().isoformat(),
    }


def _with_static_check(payload: str) -> str:
    """Annotate generated wrapper JSON with import/API drift found in its code."""
    try:
        data = json.loads(payload)
    except ValueError:
        return payload
    if not isinstance(data, dict) or not isinstance(data.get("code"), str):
        return payload
    problems = check_generated_code(data["code"])
    if problems:
        data["static_check_errors"] = problems
    return json.dumps(data)


def _usable_system_dependencies(registry, tool_def: ToolDefinition) -> list[str]:
    """Sanitize persisted dependencies (older registries may contain prose words)."""
    kept, status = _screen_system_dependencies(
        tool_def.system_dependencies, tool_def.system_dependency_status
    )
    status = {
        **{
            k: v
            for k, v in tool_def.system_dependency_status.items()
            if v.startswith("rejected") and k not in status
        },
        **status,
    }
    if kept != tool_def.system_dependencies or status != tool_def.system_dependency_status:
        rejected = sorted(set(tool_def.system_dependencies) - set(kept))
        if rejected:
            logger.info(
                "Dropping implausible system dependencies for %s: %s", tool_def.name, rejected
            )
        tool_def.system_dependencies = kept
        tool_def.system_dependency_status = status
        _persist_tool(registry, tool_def)
    return kept


def _missing_python_dependencies(dependencies):
    missing = []
    for requirement in dependencies:
        name = re.split(r"[\[<>=!~ @]", requirement, maxsplit=1)[0]
        normalized = name.lower().replace("_", "-")
        module = _IMPORT_ALIASES.get(normalized, normalized.replace("-", "_"))
        try:
            __import__(module)
        except ImportError:
            missing.append(requirement)
    return missing


@tool
def extract_tools_from_text(text: str, _section_type: str = "methods") -> str:
    """Extract bioinformatics tool mentions from scientific text.

    Analyzes Methods sections of papers to identify:
    - Software tools being used
    - Tasks they perform
    - Associated databases
    - GitHub/documentation URLs

    Args:
        text: The scientific text to analyze (ideally Methods section)
        _section_type: Type of section (methods, results, introduction)

    Returns:
        JSON string with extracted tool information
    """
    # get_llm() raises without an API key, so it is called inside the guarded block below.
    prompt = f"""Analyze this scientific text and extract all bioinformatics tools, software, and databases mentioned.

For each tool, identify:
1. Name: The exact name of the tool/software
2. Task: What the tool is used for in this context
3. URL: Any GitHub, PyPI, or documentation URL mentioned
4. Database: Any associated database the tool accesses

Text to analyze:
{text}

Return a JSON object with a "tools" array. Each tool should have: name, task, url (or null), database (or null).
Only include actual bioinformatics tools, not general Python libraries like pandas or numpy.
Focus on domain-specific tools like: samtools, BLAST, DESeq2, Scanpy, CellTypist, etc.
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group()

        return json.dumps({"tools": [], "error": "Could not extract structured data"})

    except Exception as e:
        logger.error(f"Tool extraction failed: {e}")
        return json.dumps({"tools": [], "error": str(e)})


@tool
def research_tool_documentation(tool_name: str, tool_url: str | None = None) -> str:
    """Research a tool to gather documentation and usage information.

    Searches for:
    - GitHub README and documentation
    - PyPI package information
    - API documentation
    - Usage examples

    Args:
        tool_name: Name of the tool to research
        tool_url: Optional URL to start research from

    Returns:
        JSON string with gathered documentation
    """
    # Placeholder: returns a template for the LLM instead of doing a real web search.

    result = {
        "tool_name": tool_name,
        "url": tool_url,
        "research_notes": f"Research needed for {tool_name}",
        "suggested_actions": [
            f"Search GitHub for '{tool_name}'",
            f"Check PyPI for '{tool_name}' package",
            "Look for official documentation",
            "Find usage examples in tutorials",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def generate_tool_wrapper(
    tool_name: str,
    tool_description: str,
    tool_category: str,
    parameters_json: str,
    return_type: str,
    return_description: str,
    implementation_notes: str,
) -> str:
    """Generate a Python wrapper for a bioinformatics tool.

    Creates a complete Python function that wraps the tool's functionality,
    handling input/output, error handling, and integration with BioAgents.

    Args:
        tool_name: Name for the wrapper function
        tool_description: Description of what the tool does
        tool_category: Category (genomics, proteomics, structural, ml, etc.)
        parameters_json: JSON string describing parameters
        return_type: Return type (string, dict, DataFrame)
        return_description: Description of what's returned
        implementation_notes: Notes on how to implement (CLI wrapper, API call, etc.)

    Returns:
        JSON string with generated code and metadata
    """
    llm = get_llm()

    prompt = f"""Generate a Python wrapper function for this bioinformatics tool:

Tool Name: {tool_name}
Description: {tool_description}
Category: {tool_category}
Parameters: {parameters_json}
Return Type: {return_type}
Return Description: {return_description}
Implementation Notes: {implementation_notes}

Requirements:
1. Create a well-documented Python function
2. Include proper type hints
3. Add comprehensive docstring
4. Handle errors gracefully
5. Return data in a consistent format
6. Make it work as a standalone function

Generate the complete Python code for this wrapper. Include:
- All necessary imports at the top
- The main function implementation
- A simple test case at the bottom (in if __name__ == "__main__" block)

Return a JSON object with:
- "code": The complete Python code
- "dependencies": List of pip packages needed
- "test_code": A simple test invocation
"""

    try:
        content = _invoke_wrapper(llm, prompt)

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return _with_static_check(json_match.group())

        code_match = re.search(r"```python\n([\s\S]*?)```", content)
        if code_match:
            return _with_static_check(
                json.dumps(
                    {
                        "code": code_match.group(1),
                        "dependencies": [],
                        "test_code": None,
                    }
                )
            )

        return json.dumps(
            {
                "error": "Could not generate wrapper",
                "raw_response": content[:500],
            }
        )

    except Exception as e:
        logger.error(f"Wrapper generation failed: {e}")
        return json.dumps({"error": str(e)})


@tool
@track_dependencies(component="tool_builder", phase="registration")
def register_custom_tool(
    name: str,
    description: str,
    category: str,
    code: str,
    parameters_json: str,
    return_type: str,
    return_description: str,
    dependencies_json: str = "[]",
    system_dependencies_json: str = "[]",
    source_paper: str | None = None,
    source_url: str | None = None,
    source_software: str | None = None,
) -> str:
    """Register a custom tool in the BioAgents tool registry.

    Saves the tool for future use by all agents in the system.

    Args:
        name: Unique name for the tool
        description: What the tool does
        category: Tool category
        code: Python code implementing the tool
        parameters_json: JSON array of parameter definitions
        return_type: What the tool returns
        return_description: Description of return value
        dependencies_json: JSON array of pip dependencies
        system_dependencies_json: JSON array of system dependencies (e.g., ["samtools", "blast"])
        source_paper: DOI or bioRxiv ID if from literature
        source_url: GitHub or documentation URL
        source_software: Name of the wrapped software

    Returns:
        Status message
    """
    try:
        params_data = json.loads(parameters_json)
        parameters = [
            ToolParameter(
                name=p.get("name", "arg"),
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
            )
            for p in params_data
        ]

        dependencies = json.loads(dependencies_json)
        system_dependencies = json.loads(system_dependencies_json)

        if not system_dependencies:
            extracted_deps = extract_system_dependencies_from_code(code)
            if extracted_deps:
                system_dependencies = extracted_deps
                logger.info(f"Auto-extracted system dependencies from code: {extracted_deps}")
        if not isinstance(system_dependencies, list):
            system_dependencies = []
        system_dependencies, system_dependency_status = _screen_system_dependencies(
            system_dependencies
        )
        rejected = {k: v for k, v in system_dependency_status.items() if v.startswith("rejected")}
        if rejected:
            logger.info(f"Rejected implausible system dependencies for {name}: {sorted(rejected)}")

        definition = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            return_type=return_type,
            return_description=return_description,
            code=code,
            dependencies=dependencies,
            system_dependencies=system_dependencies,
            system_dependency_status=system_dependency_status,
            source_paper=source_paper,
            source_url=source_url,
            source_software=source_software,
            static_check_errors=check_generated_code(code),
        )
        if definition.static_check_errors:
            definition.last_failure = _failure("static_check", definition.static_check_errors)

        registry = get_registry()
        success = registry.register_tool(definition, overwrite=True)

        if success:
            response: dict[str, object] = {
                "status": "success",
                "message": f"Tool '{name}' registered successfully",
                "tool_name": name,
                "category": category,
            }
            if definition.static_check_errors:
                response["static_check_errors"] = definition.static_check_errors
                response["warning"] = (
                    "The code does not match the installed environment; fix these "
                    "problems and re-register before validating."
                )
            return json.dumps(response)
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to register tool '{name}'",
                }
            )

    except Exception as e:
        logger.error(f"Tool registration failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
@track_dependencies(component="tool_builder", phase="validation")
def validate_custom_tool(tool_name: str, test_args_json: str = "") -> str:
    """Validate a custom tool by loading and optionally testing it.

    Args:
        tool_name: Name of the tool to validate
        test_args_json: Optional JSON object with test arguments. Omit to load without execution.

    Returns:
        Validation result
    """
    try:
        registry = get_registry()
        test_args = json.loads(test_args_json) if test_args_json else None

        tool = registry.get_tool(tool_name)
        if tool is not None:
            tool.static_check_errors = check_generated_code(tool.code)
            if tool.static_check_errors:
                tool.validated = False
                tool.validation_error = "Static check failed: " + "; ".join(
                    tool.static_check_errors
                )
                tool.last_failure = _failure("static_check", tool.static_check_errors)
                _persist_tool(registry, tool)
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Validation failed: {tool.validation_error}",
                        "static_check_errors": tool.static_check_errors,
                    }
                )

        success = registry.validate_tool(tool_name, test_args)

        if success:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Tool '{tool_name}' validated successfully",
                    "tool_name": tool_name,  # Add tool_name for success detection
                }
            )
        else:
            tool = registry.get_tool(tool_name)
            error = tool.validation_error if tool else "Tool not found"
            if tool:
                tool.last_failure = _failure("validation", [str(error)])
                _persist_tool(registry, tool)
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Validation failed: {error}",
                }
            )

    except Exception as e:
        logger.error(f"Tool validation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def list_custom_tools(category: str | None = None, validated_only: bool = False) -> str:
    """List all custom tools in the registry.

    Args:
        category: Filter by category (genomics, proteomics, etc.)
        validated_only: Only show validated tools

    Returns:
        JSON array of tool information
    """
    registry = get_registry()
    tools = registry.list_tools(category=category, validated_only=validated_only)

    return json.dumps(
        {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "validated": t.validated,
                    "usage_count": t.usage_count,
                    "source_software": t.source_software,
                    "parameters": [p.__dict__ for p in t.parameters],
                    "schema": t.get_schema(),
                    "execution_tool": "execute_custom_tool",
                }
                for t in tools
            ],
            "total": len(tools),
        },
        indent=2,
    )


@tool
def search_custom_tools(query: str, limit: int = 5) -> str:
    """Search custom tools by description.

    Args:
        query: Search query describing needed functionality
        limit: Maximum results to return

    Returns:
        JSON array of matching tools with scores
    """
    registry = get_registry()
    results = registry.search_tools(query, limit=limit)

    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "score": score,
                    "validated": tool.validated,
                    "parameters": [p.__dict__ for p in tool.parameters],
                    "schema": tool.get_schema(),
                    "execution_tool": "execute_custom_tool",
                }
                for tool, score in results
            ],
        },
        indent=2,
    )


@tool
def get_tool_code(tool_name: str) -> str:
    """Get the Python code for a custom tool.

    Args:
        tool_name: Name of the tool

    Returns:
        The tool's Python code or error message
    """
    registry = get_registry()
    tool = registry.get_tool(tool_name)

    if tool is None:
        return json.dumps({"error": f"Tool '{tool_name}' not found"})

    return json.dumps(
        {
            "name": tool.name,
            "code": tool.code,
            "dependencies": tool.dependencies,
            "parameters": [p.__dict__ for p in tool.parameters],
            "schema": tool.get_schema(),
            "execution_tool": "execute_custom_tool",
        },
        indent=2,
    )


def _argument_error(tool_def: ToolDefinition, message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "tool": tool_def.name,
            "message": f"Invalid tool arguments: {message}",
            "schema": tool_def.get_schema(),
        }
    )


@tool
@track_dependencies(component="tool_builder", phase="execution")
def execute_custom_tool(tool_name: str, arguments_json: str = "{}") -> str:
    """Execute a custom tool from the registry.

    Automatically checks and installs missing system dependencies before execution.

    Args:
        tool_name: Name of the tool to execute
        arguments_json: JSON object with tool arguments

    Returns:
        Tool execution result
    """

    try:
        registry = get_registry()
        tool_def = registry.get_tool(tool_name)

        if tool_def is None:
            return json.dumps(
                {
                    "status": "error",
                    "tool": tool_name,
                    "message": f"Tool '{tool_name}' not found in registry",
                }
            )

        args = json.loads(arguments_json)
        if not isinstance(args, dict):
            return _argument_error(tool_def, "arguments_json must contain a JSON object")
        unknown = set(args) - {p.name for p in tool_def.parameters}
        missing = {p.name for p in tool_def.parameters if p.required} - set(args)
        if unknown or missing:
            return _argument_error(
                tool_def,
                f"Unknown parameters: {sorted(unknown)}; missing required parameters: {sorted(missing)}",
            )

        # Canonicalise path-like arguments against the workspace root (issue #68).
        from bioagents.sandbox.workspace import WorkspacePathError, resolve_path_arguments

        try:
            args, resolved_paths = resolve_path_arguments(args)
        except WorkspacePathError as exc:
            return _argument_error(tool_def, str(exc))

        if _usable_system_dependencies(registry, tool_def):
            missing_deps = []
            for dep in tool_def.system_dependencies:
                if not shutil.which(dep):
                    missing_deps.append(dep)
            not_retried = [
                d
                for d in missing_deps
                if tool_def.system_dependency_status.get(d) == _INSTALL_FAILED
            ]
            if not_retried:
                logger.info(
                    f"Not retrying installs that previously failed for {tool_name}: {not_retried}"
                )
                missing_deps = [d for d in missing_deps if d not in not_retried]

            if missing_deps:
                logger.info(
                    f"Missing system dependencies for {tool_name}: {missing_deps}. "
                    "Attempting to install..."
                )

                # Try mamba first (faster, better dependency resolution)
                mamba_path = shutil.which("mamba")
                if mamba_path:
                    try:
                        install_cmd = [
                            "mamba",
                            "install",
                            "-y",
                            "--quiet",
                            "-c",
                            "bioconda",
                            "-c",
                            "conda-forge",
                            *missing_deps,
                        ]
                        result = run_install_command(
                            install_cmd,
                            reason="missing_system_executable",
                            capture_output=True,
                            text=True,
                            timeout=60,  # 1 minute timeout (shorter to avoid workflow timeout)
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Successfully installed dependencies via mamba: {missing_deps}"
                            )
                            missing_deps = [d for d in missing_deps if not shutil.which(d)]
                        else:
                            logger.warning(f"Mamba install failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Mamba dependency installation timed out after 60s for: {missing_deps}. "
                            "Skipping automatic installation."
                        )
                    except Exception as e:
                        logger.warning(f"Mamba install failed: {e}")

                conda_path = shutil.which("conda")
                if conda_path and missing_deps:
                    try:
                        install_cmd = [
                            "conda",
                            "install",
                            "-y",
                            "--quiet",
                            "-c",
                            "bioconda",
                            "-c",
                            "conda-forge",
                            *missing_deps,
                        ]
                        result = run_install_command(
                            install_cmd,
                            reason="missing_system_executable",
                            capture_output=True,
                            text=True,
                            timeout=60,  # 1 minute timeout (shorter to avoid workflow timeout)
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Successfully installed dependencies via conda: {missing_deps}"
                            )
                            missing_deps = [d for d in missing_deps if not shutil.which(d)]
                        else:
                            logger.warning(f"Conda install failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Conda dependency installation timed out after 60s for: {missing_deps}. "
                            "Skipping automatic installation."
                        )
                    except Exception as e:
                        logger.warning(f"Conda install failed: {e}")

                # apt-get needs sudo, so only log manual instructions.
                if missing_deps and os.name != "nt" and shutil.which("apt-get"):
                    record_blocked_install(
                        ["sudo", "apt-get", "install", "-y", *missing_deps],
                        reason="missing_system_executable",
                        detail="requires_sudo",
                    )
                    logger.warning(
                        f"apt-get installation requires sudo privileges. "
                        f"Please install manually: sudo apt-get install {' '.join(missing_deps)}"
                    )

                still_missing = [d for d in missing_deps if not shutil.which(d)]
                if still_missing:
                    tool_def.system_dependency_status = {
                        **tool_def.system_dependency_status,
                        **dict.fromkeys(still_missing, _INSTALL_FAILED),
                    }
                    _persist_tool(registry, tool_def)
                    install_instructions = {
                        "mamba": f"mamba install -c bioconda -c conda-forge {' '.join(still_missing)}",
                        "conda": f"conda install -c bioconda -c conda-forge {' '.join(still_missing)}",
                    }
                    if os.name != "nt":
                        install_instructions["apt"] = (
                            f"sudo apt-get install {' '.join(still_missing)}"
                        )

                    # Run anyway: the tool may degrade gracefully without the dependency.
                    logger.warning(
                        f"Missing system dependencies: {', '.join(still_missing)}. "
                        f"Attempting to run tool anyway. Install with: {install_instructions}"
                    )

        if tool_def.dependencies:
            missing_pip_deps = _missing_python_dependencies(tool_def.dependencies)
            missing_pip_deps, guarded_pip_deps = filter_install_requests(missing_pip_deps)
            if guarded_pip_deps:
                record_blocked_install(
                    [sys.executable, "-m", "pip", "install", *guarded_pip_deps],
                    reason="missing_python_dependency",
                    detail="pip_guard",
                )
                logger.warning(
                    f"Not installing pinned/project packages for {tool_name}: {guarded_pip_deps}"
                )

            if missing_pip_deps:
                logger.info(
                    f"Missing Python dependencies for {tool_name}: {missing_pip_deps}. "
                    "Attempting to install via pip..."
                )

                try:
                    pip_cmd = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        *missing_pip_deps,
                    ]

                    logger.info(f"Installing packages: {', '.join(missing_pip_deps)}")
                    logger.info("Installation progress (real-time):")
                    result = _run_pip_install_streaming(pip_cmd, timeout=180)

                    if result.returncode != 0 and "No module named pip" in result.stdout:
                        logger.info(
                            "Pip module not found. Attempting to install pip via ensurepip..."
                        )
                        ensurepip_cmd = [sys.executable, "-m", "ensurepip", "--upgrade"]
                        ensurepip_installed = False

                        try:
                            ensurepip_result = run_install_command(
                                ensurepip_cmd,
                                reason="pip_bootstrap",
                                capture_output=True,
                                text=True,
                                timeout=180,
                            )
                            if ensurepip_result.returncode == 0:
                                logger.info(
                                    "Pip installed successfully via ensurepip. Verifying installation..."
                                )
                                verify_cmd = [sys.executable, "-m", "pip", "--version"]
                                verify_result = subprocess.run(  # nosec B603
                                    verify_cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=10,
                                )
                                if verify_result.returncode == 0:
                                    logger.info(f"Pip verified: {verify_result.stdout.strip()}")
                                    ensurepip_installed = True
                                else:
                                    logger.warning(
                                        f"Pip installation verification failed: {verify_result.stderr}"
                                    )
                            else:
                                logger.warning(
                                    f"Failed to install pip via ensurepip (returncode {ensurepip_result.returncode}): "
                                    f"{ensurepip_result.stderr or ensurepip_result.stdout}. "
                                    "Trying alternative methods..."
                                )
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                "ensurepip timed out after 180s. This may indicate network issues or "
                                "a corrupted Python installation. Trying alternative methods..."
                            )
                        except Exception as e:
                            logger.warning(
                                f"ensurepip failed with exception: {e}. Trying alternative methods..."
                            )

                        if not ensurepip_installed:
                            try:
                                logger.info("Trying ensurepip with --user flag...")
                                ensurepip_user_cmd = [
                                    sys.executable,
                                    "-m",
                                    "ensurepip",
                                    "--upgrade",
                                    "--user",
                                ]
                                ensurepip_user_result = run_install_command(
                                    ensurepip_user_cmd,
                                    reason="pip_bootstrap",
                                    capture_output=True,
                                    text=True,
                                    timeout=180,
                                )
                                if ensurepip_user_result.returncode == 0:
                                    logger.info("Pip installed successfully via ensurepip --user")
                                    ensurepip_installed = True
                            except Exception as e:
                                logger.warning(f"ensurepip --user failed: {e}")

                            if not ensurepip_installed:
                                direct_pip = shutil.which("pip")
                                if direct_pip:
                                    logger.info(
                                        f"Found pip in PATH: {direct_pip}. Using it directly..."
                                    )
                                    direct_pip_cmd = [
                                        direct_pip,
                                        "install",
                                        "--disable-pip-version-check",
                                        *missing_pip_deps,
                                    ]
                                    try:
                                        logger.info("Installation progress (real-time):")
                                        result = _run_pip_install_streaming(
                                            direct_pip_cmd, timeout=180
                                        )
                                        if result.returncode == 0:
                                            ensurepip_installed = True
                                    except Exception as e:
                                        logger.warning(f"Direct pip install failed: {e}")
                                else:
                                    logger.warning(
                                        "Pip not available via ensurepip, ensurepip --user, or PATH. "
                                        "Please install pip manually or use a proper virtual environment. "
                                        "You can try: python -m ensurepip --upgrade"
                                    )

                        if ensurepip_installed:
                            logger.info("Retrying package installation with newly installed pip...")
                            logger.info("Installation progress (real-time):")
                            result = _run_pip_install_streaming(pip_cmd, timeout=180)

                    if result.returncode == 0:
                        logger.info(
                            f"Successfully installed Python dependencies via pip: {missing_pip_deps}"
                        )
                        still_missing_pip = _missing_python_dependencies(missing_pip_deps)
                        if still_missing_pip:
                            record_dependency_event(
                                "install_unverified",
                                packages=still_missing_pip,
                                reason="missing_python_dependency",
                            )
                            logger.warning(
                                f"Some Python dependencies could not be imported after installation: {still_missing_pip}"
                            )
                    else:
                        logger.warning(f"Pip install failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Pip dependency installation timed out after 180s for: {missing_pip_deps}"
                    )
                except Exception as e:
                    logger.warning(f"Pip install failed: {e}")

        still_missing_before_exec = []
        if tool_def.system_dependencies:
            still_missing_before_exec = [
                d for d in tool_def.system_dependencies if not shutil.which(d)
            ]

        func = registry.load_tool_function(tool_name)

        if func is None:
            reason = tool_def.validation_error or "unknown load error"
            tool_def.validated = False
            tool_def.last_failure = _failure("load", [reason])
            _persist_tool(registry, tool_def)
            return json.dumps(
                {
                    "status": "error",
                    "tool": tool_name,
                    "message": f"Failed to load tool '{tool_name}': {reason}",
                }
            )

        try:
            inspect.signature(func).bind(**args)
        except TypeError as exc:
            return _argument_error(tool_def, str(exc))
        result = func(**args)

        # Some tools return an error dict naming the missing dependency.
        if isinstance(result, dict):
            error_msg = result.get("message", "") or result.get("error", "")
            if still_missing_before_exec and any(
                dep in error_msg.lower() for dep in still_missing_before_exec
            ):
                install_instructions = {
                    "mamba": f"mamba install -c bioconda -c conda-forge {' '.join(still_missing_before_exec)}",
                    "conda": f"conda install -c bioconda -c conda-forge {' '.join(still_missing_before_exec)}",
                }
                if os.name != "nt":
                    install_instructions["apt"] = (
                        f"sudo apt-get install {' '.join(still_missing_before_exec)}"
                    )

                if result.get("status") == "error":
                    result["install_instructions"] = install_instructions
                    result["note"] = "Please install the missing dependencies and try again."

        registry.record_usage(tool_name)

        return json.dumps(
            {
                "status": tool_result_status(result),
                "tool": tool_name,
                "result": result,
                "resolved_paths": resolved_paths,
            },
            default=str,
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        error_msg = str(e)

        try:
            registry = get_registry()
            tool_def = registry.get_tool(tool_name)
            if tool_def and tool_def.system_dependencies:
                still_missing = [d for d in tool_def.system_dependencies if not shutil.which(d)]
                if still_missing and any(dep in error_msg.lower() for dep in still_missing):
                    install_instructions = {
                        "mamba": f"mamba install -c bioconda -c conda-forge {' '.join(still_missing)}",
                        "conda": f"conda install -c bioconda -c conda-forge {' '.join(still_missing)}",
                    }
                    if os.name != "nt":
                        install_instructions["apt"] = (
                            f"sudo apt-get install {' '.join(still_missing)}"
                        )

                    return json.dumps(
                        {
                            "status": "error",
                            "tool": tool_name,
                            "message": error_msg,
                            "install_instructions": install_instructions,
                            "note": "Please install the missing dependencies and try again.",
                        },
                        indent=2,
                    )
        except Exception:  # nosec B110
            # If we can't get tool_def, just return the error
            pass

        return json.dumps(
            {
                "status": "error",
                "tool": tool_name,
                "message": error_msg,
            }
        )


def get_tool_builder_tools() -> list:
    """Get all tools available to the Tool Builder Agent."""
    return [
        extract_tools_from_text,
        research_tool_documentation,
        generate_tool_wrapper,
        register_custom_tool,
        validate_custom_tool,
        list_custom_tools,
        search_custom_tools,
        get_tool_code,
        execute_custom_tool,
    ]
