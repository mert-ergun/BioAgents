# Prompt Engineering with XML

## Overview

BioAgents uses **XML-formatted system prompts** for all agents in the multi-agent system. This approach provides superior structure, maintainability, and effectiveness compared to traditional string-based prompts.

## Why XML for Prompts?

### 1. **Structure and Hierarchy**

XML provides a natural way to organize complex prompts into logical sections:

```xml
<prompt>
  <role>Core responsibility</role>
  <capabilities>What the agent can do</capabilities>
  <instructions>How to behave</instructions>
  <examples>Concrete use cases</examples>
</prompt>
```

### 2. **Maintainability**

- **Easy Updates**: Modify specific sections without touching others
- **Version Control**: XML diffs clearly show what changed
- **Self-Documenting**: Structure makes the prompt's organization obvious
- **Modular**: Sections can be independently updated

### 3. **Consistency**

- Standardized format across all agents
- Common sections with similar structure
- Easy to apply prompt engineering patterns

### 4. **Flexibility**

- Priority levels for instructions
- Conditional content sections
- Metadata for tracking and versioning
- Extensible without breaking existing structure

## Prompt Structure

### Standard Sections

#### Metadata
```xml
<metadata>
  <name>Agent Name</name>
  <version>1.0</version>
  <description>Brief description</description>
</metadata>
```

**Purpose**: Track prompt versions and provide context

#### Role
```xml
<role>
  You are a specialized [Agent Type] in a bioinformatics team.
  Your primary responsibility is [core function].
</role>
```

**Purpose**: Define the agent's identity and core responsibility

#### Capabilities
```xml
<capabilities>
  <capability>
    <name>Capability Name</name>
    <description>What it does</description>
    <tool>tool_name</tool>
  </capability>
</capabilities>
```

**Purpose**: Document available tools and abilities

#### Instructions
```xml
<instructions>
  <instruction priority="high">Critical instruction</instruction>
  <instruction priority="medium">Important guideline</instruction>
  <instruction priority="low">Optional suggestion</instruction>
</instructions>
```

**Purpose**: Provide behavioral guidelines with priority levels

#### Examples
```xml
<examples>
  <example>
    <scenario>User asks for X</scenario>
    <action>Agent does Y</action>
    <response>Agent responds with Z</response>
  </example>
</examples>
```

**Purpose**: Show concrete examples of expected behavior

## Agent-Specific Sections

### Supervisor Agent

```xml
<team>
  <agent name="research">
    <description>What this agent does</description>
    <use_when>When to route to this agent</use_when>
  </agent>
</team>

<decision_guidelines>
  <guideline>
    <condition>When X happens</condition>
    <action>Route to Y</action>
  </guideline>
</decision_guidelines>
```

### Analysis Agent

```xml
<interpretation_guidelines>
  <guideline property="molecular_weight">
    <context>What this property means</context>
    <insights>
      <insight>How to interpret low values</insight>
      <insight>How to interpret high values</insight>
    </insights>
  </guideline>
</interpretation_guidelines>

<biochemical_context>
  <property name="hydrophobicity">
    <amino_acids>A, V, I, L, M, F, W, P</amino_acids>
    <significance>Biological importance</significance>
  </property>
</biochemical_context>
```

### Report Agent

```xml
<report_structure>
  <section name="Summary" priority="high">
    <description>What goes in this section</description>
    <content>
      <item>Specific item to include</item>
    </content>
  </section>
</report_structure>

<synthesis_guidelines>
  <guideline>
    <principle>How to synthesize information</principle>
    <example>Concrete example</example>
  </guideline>
</synthesis_guidelines>
```

## Prompt Engineering Best Practices

### 1. **Be Specific About Role**

- **Bad**: "You are an AI assistant"

- **Good**: "You are a specialized Research Agent in a bioinformatics team. Your primary responsibility is to fetch and retrieve biological data from databases."

### 2. **Prioritize Instructions**

```xml
<instructions>
  <instruction priority="high">
    ! Use available tools to retrieve requested data
  </instruction>
  <instruction priority="medium">
    • Return data in a clear format
  </instruction>
  <instruction priority="low">
    Be concise in your responses
  </instruction>
</instructions>
```

### 3. **Provide Context**

Include domain knowledge that helps the agent understand its purpose:

```xml
<biochemical_context>
  <property name="isoelectric_point">
    <context>pI affects protein solubility and behavior at different pH</context>
    <insights>
      <insight>Low pI (&lt;5): Acidic protein</insight>
      <insight>High pI (&gt;8): Basic protein</insight>
    </insights>
  </property>
</biochemical_context>
```

### 4. **Use Concrete Examples**

Abstract instructions are less effective than concrete examples:

```xml
<example>
  <request>Fetch the sequence for P53_HUMAN</request>
  <action>Call fetch_uniprot_fasta with protein_id="P53_HUMAN"</action>
  <response>Return the FASTA sequence without additional commentary</response>
</example>
```

### 5. **Define Boundaries**

Explicitly state what the agent should NOT do:

```xml
<instruction priority="high">
  Do NOT analyze or interpret the data - that's the Analysis Agent's job
</instruction>
```

### 6. **Include Error Handling**

Guide the agent on how to handle errors:

```xml
<error_handling>
  <error type="protein_not_found">
    <response>Clearly state that the protein identifier was not found</response>
    <suggestion>Suggest checking the identifier format</suggestion>
  </error>
</error_handling>
```

## Comparison: String vs XML Prompts

### Traditional String Prompt

```python
AGENT_PROMPT = """You are an analysis agent.
Analyze protein sequences.
Use tools to calculate properties.
Be scientific."""
```

**Problems**:
- Flat structure, hard to organize
- Difficult to update specific parts
- No metadata or versioning
- Limited expressiveness
- Hard to maintain consistency

### XML-Based Prompt

```xml
<prompt>
  <metadata>
    <name>Analysis Agent</name>
    <version>1.0</version>
  </metadata>

  <role>
    You are a specialized Analysis Agent in a bioinformatics team.
    Your expertise lies in analyzing biological sequences.
  </role>

  <capabilities>
    <capability>
      <name>Molecular Weight Calculation</name>
      <tool>calculate_molecular_weight</tool>
    </capability>
  </capabilities>

  <instructions>
    <instruction priority="high">
      Use appropriate analysis tools
    </instruction>
    <instruction priority="medium">
      Interpret results scientifically
    </instruction>
  </instructions>
</prompt>
```

**Benefits**:
- Clear hierarchical structure
- Easy to update sections independently
- Version tracking with metadata
- Rich expressiveness with nested elements
- Consistent format across agents

## Loading and Using Prompts

### Basic Usage

```python
from bioagents.prompts.prompt_loader import load_prompt

# Load a prompt
supervisor_prompt = load_prompt("supervisor")

# Use in agent
messages = [SystemMessage(content=supervisor_prompt)] + messages
```

### Custom Prompt Loader

```python
from bioagents.prompts.prompt_loader import PromptLoader
from pathlib import Path

# Create a custom loader
custom_loader = PromptLoader(prompts_dir=Path("custom/prompts"))

# Load from custom directory
prompt = custom_loader.load_prompt("my_agent")
```

## Updating Prompts

### Step-by-Step Process

1. **Edit the XML file** in `bioagents/prompts/`
2. **Update version number** in metadata
3. **Validate XML structure**:
   ```bash
   python examples/test_xml_prompts.py
   ```
4. **Test with the agent**:
   ```bash
   python -m bioagents.main
   ```
5. **Document significant changes** in comments

### Example Update

```xml
<!-- Before -->
<instruction priority="medium">
  Analyze amino acid composition
</instruction>

<!-- After -->
<instruction priority="high">
  Analyze amino acid composition and highlight unusual patterns
</instruction>
```

Update version:
```xml
<metadata>
  <name>Analysis Agent</name>
  <version>1.1</version>  <!-- Incremented -->
  <description>Enhanced with pattern highlighting</description>
</metadata>
```

## Validation

### Automated Testing

```bash
# Validate all XML prompts
python examples/test_xml_prompts.py

# Run basic workflow test
python examples/test_basic_workflow.py
```

### Manual Validation

```python
import xml.etree.ElementTree as ET

# Validate XML is well-formed
tree = ET.parse('bioagents/prompts/supervisor.xml')
root = tree.getroot()

# Check required sections
assert root.find('role') is not None
assert root.find('instructions') is not None
```

## Advanced Techniques

### Priority-Based Instructions

Use priority levels to emphasize critical instructions:

```xml
<instructions>
  <instruction priority="high">Never skip this</instruction>
  <instruction priority="medium">Important guideline</instruction>
  <instruction priority="low">Optional suggestion</instruction>
</instructions>
```

The loader adds visual markers for high-priority items.

### Conditional Sections

Include sections that apply only in certain contexts:

```xml
<section name="Sequence Analysis" priority="high" condition="if_analysis_performed">
  <!-- Only relevant when analysis is done -->
</section>
```

### Nested Context

Provide hierarchical context:

```xml
<interpretation_guidelines>
  <guideline property="molecular_weight">
    <context>Overall importance</context>
    <insights>
      <insight>Specific interpretation 1</insight>
      <insight>Specific interpretation 2</insight>
    </insights>
  </guideline>
</interpretation_guidelines>
```

## Future Enhancements

### Parameterized Prompts

```xml
<role>
  You are a ${AGENT_TYPE} agent specialized in ${DOMAIN}.
</role>
```

### Multi-Language Support

```xml
<prompt language="en">
  <role>You are a research agent...</role>
</prompt>

<prompt language="es">
  <role>Eres un agente de investigación...</role>
</prompt>
```

### A/B Testing

```xml
<metadata>
  <version>1.0</version>
  <variant>A</variant>  <!-- or B for testing -->
</metadata>
```

## Resources

- **XML Prompts**: `bioagents/prompts/*.xml`
- **Prompt Loader**: `bioagents/prompts/prompt_loader.py`
- **Testing**: `examples/test_xml_prompts.py`
- **Documentation**: `bioagents/prompts/README.md`

## Summary

XML-formatted prompts provide:

- **Better Structure** - Clear hierarchy and organization
- **Easier Maintenance** - Update sections independently
- **Version Control** - Track changes effectively
- **Consistency** - Standardized format across agents
- **Flexibility** - Rich expressiveness for complex prompts
- **Documentation** - Self-documenting with metadata
- **Testing** - Validate structure automatically

This approach scales well as the system grows and makes prompt engineering more systematic and effective.
