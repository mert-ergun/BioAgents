# ACE Configuration Guide

ACE (Agent-Curator-Environment) enables self-evolving capabilities for BioAgents. This guide explains how to configure ACE.

## Environment Variables

Add these variables to your `.env` file:

### Basic Configuration

```bash
# Enable/disable ACE wrapper for self-evolving agent capabilities
# Set to 'true', '1', or 'yes' to enable (default: false)
ACE_ENABLED=false

# How often to run curator (every N executions)
# Lower values = more frequent updates, higher values = less frequent
ACE_CURATOR_FREQUENCY=5

# Playbook directory (relative to BioAgents root)
ACE_PLAYBOOK_DIR=bioagents/playbooks

# Playbook token budget (optional, default: 80000)
ACE_PLAYBOOK_TOKEN_BUDGET=80000
```

### Example .env Configuration

```bash
# ACE Configuration
ACE_ENABLED=true
ACE_CURATOR_FREQUENCY=5
ACE_PLAYBOOK_DIR=bioagents/playbooks
ACE_PLAYBOOK_TOKEN_BUDGET=80000
```

## How It Works

1. **Enable ACE**: Set `ACE_ENABLED=true` in your `.env` file
2. **Run Agents**: Use BioAgents normally - ACE automatically tracks executions
3. **Playbooks**: Playbooks are created/updated in `bioagents/playbooks/` directory
4. **Playbook Integration**: When ACE is enabled, learned best practices from playbooks are **directly appended to agent prompts** (following ACE's original approach). Only bullets with `helpful > harmful` are included.
5. **Evolution**: Every N executions (defined by `ACE_CURATOR_FREQUENCY`), the curator adds new instructions based on learned patterns

## Disabling ACE

To disable ACE, simply set:
```bash
ACE_ENABLED=false
```

When disabled, BioAgents works exactly as before with zero overhead.

## Playbook Files

Playbooks are stored as YAML files in the `bioagents/playbooks/` directory:
- `supervisor_playbook.yaml`
- `coder_playbook.yaml`
- `research_playbook.yaml`
- `ml_playbook.yaml`
- `dl_playbook.yaml`

These files are automatically created and updated as agents learn from executions.

## Testing

To test ACE integration:

```bash
# Enable ACE
export ACE_ENABLED=true
export ACE_CURATOR_FREQUENCY=2  # Run curator every 2 executions for testing

# Run demo script
python examples/ace_integration_demo.py
```

## Troubleshooting

### ACE not working?

1. Check that `ACE_ENABLED=true` is set in your `.env` file
2. Verify that the `ace/` directory exists in the BioAgents repository root (not parent directory)
3. Check that playbooks directory is writable: `bioagents/playbooks/`
4. Ensure ACE framework files are present: `ace/ace/core/generator.py`, `ace/ace/core/reflector.py`, `ace/ace/core/curator.py`

### Playbooks not updating?

- Check that `ACE_CURATOR_FREQUENCY` is set appropriately (lower = more frequent)
- Verify that agents are actually executing (check logs)
- Ensure playbooks directory exists and is writable
