"""Tools for the Tool Builder Agent for discovering, creating, and managing custom tools."""

import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import sys

from langchain_core.tools import tool

from bioagents.llms.llm_provider import get_llm
from bioagents.tools.tool_registry import (
    ToolDefinition,
    ToolParameter,
    get_registry,
)
from bioagents.tools.tool_universe import tool_universe_call_tool, tool_universe_find_tools

logger = logging.getLogger(__name__)


def _run_pip_install_streaming(pip_cmd, timeout=180):
    """
    Run pip install command with real-time output streaming.

    Args:
        pip_cmd: List of command arguments for pip install
        timeout: Timeout in seconds

    Returns:
        Result object with returncode, stdout, and stderr attributes
    """
    # Use Popen to stream output in real-time
    process = subprocess.Popen(  # nosec B603
        pip_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered for real-time output
    )

    # Stream output line by line in real-time
    output_lines = []
    if process.stdout is not None:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"pip: {line}")
                output_lines.append(line)

    # Wait for process to complete (with timeout)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        logger.warning(f"Pip install timed out after {timeout}s")
        returncode = 1

    # Create result object for compatibility
    class Result:
        def __init__(self, returncode, output_lines):
            self.returncode = returncode
            self.stdout = "\n".join(output_lines)
            self.stderr = ""

    return Result(returncode, output_lines)


def extract_system_dependencies_from_code(code: str) -> list[str]:
    """Extract system dependency names from tool code.

    Looks for patterns like:
    - shutil.which("tool_name")
    - subprocess.run(["tool_name", ...])
    - "tool_name" in error messages about missing tools

    Args:
        code: Python code to analyze

    Returns:
        List of potential system dependency names
    """
    dependencies = []

    # Pattern 1: shutil.which("tool_name")
    pattern1 = r'shutil\.which\(["\']([^"\']+)["\']\)'
    matches = re.findall(pattern1, code)
    dependencies.extend(matches)

    # Pattern 2: subprocess.run(["tool_name", ...])
    pattern2 = r'subprocess\.run\(\[["\']([^"\']+)["\']'
    matches = re.findall(pattern2, code)
    dependencies.extend(matches)

    # Pattern 3: "tool_name not found" or similar error messages
    pattern3 = r'["\']([a-z_][a-z0-9_-]*)\s+not\s+found'
    matches = re.findall(pattern3, code, re.IGNORECASE)
    dependencies.extend(matches)

    # Pattern 4: Check for common bioinformatics tools in comments/docstrings
    bioinfo_tools = [
        "samtools",
        "blast",
        "blastn",
        "blastp",
        "blastx",
        "tblastn",
        "tblastx",
        "bowtie",
        "bowtie2",
        "bwa",
        "gatk",
        "picard",
        "bedtools",
        "vcftools",
        "bcftools",
        "htslib",
    ]
    for bioinfo_tool in bioinfo_tools:
        if (
            bioinfo_tool in code.lower()
            and bioinfo_tool not in dependencies
            and any(
                pattern in code.lower()
                for pattern in [
                    f"{bioinfo_tool} command",
                    f"{bioinfo_tool} not found",
                    f"which {bioinfo_tool}",
                    f"check.*{bioinfo_tool}",
                ]
            )
        ):
            dependencies.append(bioinfo_tool)

    # Remove duplicates and common false positives
    dependencies = list(set(dependencies))
    false_positives = [
        "python",
        "pip",
        "conda",
        "sh",
        "bash",
        "echo",
        "cat",
        "grep",
        "ls",
        "cd",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "tar",
        "zip",
        "unzip",
    ]
    dependencies = [d for d in dependencies if d not in false_positives]

    return sorted(dependencies)


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
    llm = get_llm()

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
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from the response
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
    # This would integrate with web search in production
    # For now, return a structured template for the LLM to work with

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
- "system_dependencies": List of system binaries/tools needed (e.g., ["samtools", "blast"])
- "test_code": A simple test invocation
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Extract JSON or code from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                result = json.loads(json_match.group())
                # If system_dependencies not in response, try to extract from code
                if "code" in result and "system_dependencies" not in result:
                    code = result["code"]
                    extracted_deps = extract_system_dependencies_from_code(code)
                    if extracted_deps:
                        result["system_dependencies"] = extracted_deps
                        logger.info(f"Extracted system dependencies from code: {extracted_deps}")
                return json.dumps(result)
            except json.JSONDecodeError:
                pass

        # If no JSON, wrap the code in a JSON structure
        code_match = re.search(r"```python\n([\s\S]*?)```", content)
        if code_match:
            code = code_match.group(1)
            extracted_deps = extract_system_dependencies_from_code(code)
            return json.dumps(
                {
                    "code": code,
                    "dependencies": [],
                    "system_dependencies": extracted_deps,
                    "test_code": None,
                }
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
        # Parse parameters
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

        # If system_dependencies is empty, try to extract from code
        if not system_dependencies:
            extracted_deps = extract_system_dependencies_from_code(code)
            if extracted_deps:
                system_dependencies = extracted_deps
                logger.info(f"Auto-extracted system dependencies from code: {extracted_deps}")

        # Create tool definition
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
            source_paper=source_paper,
            source_url=source_url,
            source_software=source_software,
        )

        # Register in the registry
        registry = get_registry()
        success = registry.register_tool(definition, overwrite=True)

        if success:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Tool '{name}' registered successfully",
                    "tool_name": name,
                    "category": category,
                }
            )
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
def validate_custom_tool(tool_name: str, test_args_json: str = "{}") -> str:
    """Validate a custom tool by loading and optionally testing it.

    Args:
        tool_name: Name of the tool to validate
        test_args_json: Optional JSON object with test arguments

    Returns:
        Validation result
    """
    try:
        registry = get_registry()
        test_args = json.loads(test_args_json) if test_args_json else None

        success = registry.validate_tool(tool_name, test_args)

        if success:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Tool '{tool_name}' validated successfully",
                }
            )
        else:
            tool = registry.get_tool(tool_name)
            error = tool.validation_error if tool else "Tool not found"
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
            "parameters": [
                {"name": p.name, "type": p.type, "description": p.description}
                for p in tool.parameters
            ],
        },
        indent=2,
    )


@tool
def execute_custom_tool(tool_name: str, arguments_json: str = "{}") -> str:
    """Execute a custom tool from the registry.

    Automatically checks and installs missing system dependencies before execution.

    Args:
        tool_name: Name of the tool to execute
        arguments_json: JSON object with tool arguments

    Returns:
        Tool execution result
    """
    import subprocess  # nosec B404

    try:
        registry = get_registry()
        tool_def = registry.get_tool(tool_name)

        if tool_def is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool '{tool_name}' not found in registry",
                }
            )

        # Check and install system dependencies BEFORE executing
        if tool_def.system_dependencies:
            missing_deps = []
            for dep in tool_def.system_dependencies:
                if not shutil.which(dep):
                    missing_deps.append(dep)

            if missing_deps:
                logger.info(
                    f"Missing system dependencies for {tool_name}: {missing_deps}. "
                    "Attempting to install..."
                )

                # Try mamba first (faster, better dependency resolution)
                mamba_path = shutil.which("mamba")
                if mamba_path:
                    try:
                        # Install via mamba with bioconda and conda-forge channels
                        # Use shorter timeout to avoid blocking workflow
                        install_cmd = [
                            "mamba",
                            "install",
                            "-y",
                            "--quiet",  # Reduce output
                            "-c",
                            "bioconda",
                            "-c",
                            "conda-forge",
                            *missing_deps,
                        ]
                        result = subprocess.run(  # nosec B603
                            install_cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,  # 1 minute timeout (shorter to avoid workflow timeout)
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Successfully installed dependencies via mamba: {missing_deps}"
                            )
                            # Re-check which dependencies are still missing
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

                # If mamba failed or not available, try conda (fallback)
                conda_path = shutil.which("conda")
                if conda_path and missing_deps:
                    try:
                        # Install via conda with bioconda and conda-forge channels
                        # Use shorter timeout to avoid blocking workflow
                        install_cmd = [
                            "conda",
                            "install",
                            "-y",
                            "--quiet",  # Reduce output
                            "-c",
                            "bioconda",
                            "-c",
                            "conda-forge",
                            *missing_deps,
                        ]
                        result = subprocess.run(  # nosec B603
                            install_cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,  # 1 minute timeout (shorter to avoid workflow timeout)
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Successfully installed dependencies via conda: {missing_deps}"
                            )
                            # Re-check which dependencies are still missing
                            missing_deps = [d for d in missing_deps if not shutil.which(d)]
                        else:
                            logger.warning(f"Conda install failed: {result.stderr}")
                            # Skip --no-deps attempt to avoid further delays
                            # Tool will be executed anyway, and may handle missing deps gracefully
                            logger.info("Skipping --no-deps attempt. Tool will be executed anyway.")
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Conda dependency installation timed out after 60s for: {missing_deps}. "
                            "Skipping automatic installation."
                        )
                    except Exception as e:
                        logger.warning(f"Conda install failed: {e}")

                # Skip apt-get (requires sudo, not suitable for automatic installation)
                # Just log a warning with instructions
                if missing_deps and os.name != "nt" and shutil.which("apt-get"):
                    logger.warning(
                        f"apt-get installation requires sudo privileges. "
                        f"Please install manually: sudo apt-get install {' '.join(missing_deps)}"
                    )

                # Check what's still missing after all attempts
                still_missing = [d for d in missing_deps if not shutil.which(d)]
                if still_missing:
                    # Build comprehensive install instructions
                    install_instructions = {
                        "mamba": f"mamba install -c bioconda -c conda-forge {' '.join(still_missing)}",
                        "conda": f"conda install -c bioconda -c conda-forge {' '.join(still_missing)}",
                    }
                    if os.name != "nt":
                        install_instructions["apt"] = (
                            f"sudo apt-get install {' '.join(still_missing)}"
                        )

                    # Log warning but try to run tool anyway (graceful degradation)
                    # The tool itself may handle missing dependencies gracefully
                    logger.warning(
                        f"Missing system dependencies: {', '.join(still_missing)}. "
                        f"Attempting to run tool anyway. Install with: {install_instructions}"
                    )

                    # Don't return error yet - try to execute the tool
                    # The tool's own error handling will provide better feedback
                    # If tool execution fails, it will return a proper error message

        # Check and install Python pip dependencies BEFORE executing
        if tool_def.dependencies:
            missing_pip_deps = []
            for dep in tool_def.dependencies:
                # Try to import the package
                # Handle package name variations (e.g., "scikit-learn" -> "sklearn")
                module_name = dep.replace("-", "_")
                try:
                    __import__(module_name)
                except ImportError:
                    # Also try the original name in case it's different
                    try:
                        __import__(dep)
                    except ImportError:
                        missing_pip_deps.append(dep)

            if missing_pip_deps:
                logger.info(
                    f"Missing Python dependencies for {tool_name}: {missing_pip_deps}. "
                    "Attempting to install via pip..."
                )

                try:
                    # First, try to install pip if it's missing
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

                    # If pip module is missing, try to install it first
                    if result.returncode != 0 and "No module named pip" in result.stdout:
                        logger.info(
                            "Pip module not found. Attempting to install pip via ensurepip..."
                        )
                        ensurepip_cmd = [sys.executable, "-m", "ensurepip", "--upgrade"]
                        ensurepip_installed = False

                        try:
                            ensurepip_result = subprocess.run(  # nosec B603
                                ensurepip_cmd,
                                capture_output=True,
                                text=True,
                                timeout=180,  # Increased timeout to 3 minutes
                            )
                            if ensurepip_result.returncode == 0:
                                logger.info(
                                    "Pip installed successfully via ensurepip. Verifying installation..."
                                )
                                # Verify pip is actually available
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

                        # If ensurepip didn't work, try alternative methods
                        if not ensurepip_installed:
                            # Try ensurepip with --user flag
                            try:
                                logger.info("Trying ensurepip with --user flag...")
                                ensurepip_user_cmd = [
                                    sys.executable,
                                    "-m",
                                    "ensurepip",
                                    "--upgrade",
                                    "--user",
                                ]
                                ensurepip_user_result = subprocess.run(  # nosec B603
                                    ensurepip_user_cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=180,
                                )
                                if ensurepip_user_result.returncode == 0:
                                    logger.info("Pip installed successfully via ensurepip --user")
                                    ensurepip_installed = True
                            except Exception as e:
                                logger.warning(f"ensurepip --user failed: {e}")

                            # Try direct pip command if available in PATH
                            if not ensurepip_installed:
                                direct_pip = shutil.which("pip")
                                if direct_pip:
                                    logger.info(
                                        f"Found pip in PATH: {direct_pip}. Using it directly..."
                                    )
                                    direct_pip_cmd = [
                                        direct_pip,
                                        "install",
                                        # Removed --quiet to show progress
                                        "--disable-pip-version-check",
                                        *missing_pip_deps,
                                    ]
                                    try:
                                        logger.info("Installation progress (real-time):")
                                        result = _run_pip_install_streaming(
                                            direct_pip_cmd, timeout=180
                                        )
                                        if result.returncode == 0:
                                            ensurepip_installed = True  # Mark as success
                                    except Exception as e:
                                        logger.warning(f"Direct pip install failed: {e}")
                                else:
                                    logger.warning(
                                        "Pip not available via ensurepip, ensurepip --user, or PATH. "
                                        "Please install pip manually or use a proper virtual environment. "
                                        "You can try: python -m ensurepip --upgrade"
                                    )

                        # Retry pip install if ensurepip succeeded
                        if ensurepip_installed:
                            logger.info("Retrying package installation with newly installed pip...")
                            logger.info("Installation progress (real-time):")
                            result = _run_pip_install_streaming(pip_cmd, timeout=180)

                    # Check result after initial install or retry
                    if result.returncode == 0:
                        logger.info(
                            f"Successfully installed Python dependencies via pip: {missing_pip_deps}"
                        )
                        # Re-check which dependencies are still missing
                        still_missing_pip = []
                        for dep in missing_pip_deps:
                            module_name = dep.replace("-", "_")
                            try:
                                __import__(module_name)
                            except ImportError:
                                try:
                                    __import__(dep)
                                except ImportError:
                                    still_missing_pip.append(dep)
                        if still_missing_pip:
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

        # Check if there are still missing dependencies before executing
        still_missing_before_exec = []
        if tool_def.system_dependencies:
            still_missing_before_exec = [
                d for d in tool_def.system_dependencies if not shutil.which(d)
            ]

        # Now execute the tool
        func = registry.load_tool_function(tool_name)

        if func is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to load tool '{tool_name}'",
                }
            )

        args = json.loads(arguments_json)
        result = func(**args)

        # Check if result indicates missing dependencies
        # (some tools return error dicts with dependency messages)
        if isinstance(result, dict):
            error_msg = result.get("message", "") or result.get("error", "")
            if still_missing_before_exec and any(
                dep in error_msg.lower() for dep in still_missing_before_exec
            ):
                # Build install instructions for the error response
                install_instructions = {
                    "mamba": f"mamba install -c bioconda -c conda-forge {' '.join(still_missing_before_exec)}",
                    "conda": f"conda install -c bioconda -c conda-forge {' '.join(still_missing_before_exec)}",
                }
                if os.name != "nt":
                    install_instructions["apt"] = (
                        f"sudo apt-get install {' '.join(still_missing_before_exec)}"
                    )

                # Enhance error message with install instructions
                if result.get("status") == "error":
                    result["install_instructions"] = install_instructions
                    result["note"] = "Please install the missing dependencies and try again."

        # Record usage
        registry.record_usage(tool_name)

        return json.dumps(
            {
                "status": "success",
                "tool": tool_name,
                "result": result,
            },
            default=str,
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        error_msg = str(e)

        # Check if error is related to missing dependencies
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
                "message": error_msg,
            }
        )


def get_tool_builder_tools() -> list:
    """Get all tools available to the Tool Builder Agent."""
    return [
        # Tool discovery: First check ToolUniverse, then custom registry
        tool_universe_find_tools,
        tool_universe_call_tool,
        search_custom_tools,
        list_custom_tools,
        # Tool creation and management
        extract_tools_from_text,
        research_tool_documentation,
        generate_tool_wrapper,
        register_custom_tool,
        validate_custom_tool,
        get_tool_code,
        execute_custom_tool,
    ]
