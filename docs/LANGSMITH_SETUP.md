# LangSmith Monitoring Setup Guide

This guide explains how to set up and use LangSmith monitoring for reliable token usage tracking in BioAgents.

## What is LangSmith?

LangSmith is LangChain's observability platform that provides:
- **Real-time monitoring** of all LLM calls
- **Token usage tracking** with automatic cost calculation
- **Execution traces** showing complete agent workflows
- **Performance metrics** and debugging tools

## Quick Start

### 1. Get Your API Key

1. Sign up at [https://smith.langchain.com](https://smith.langchain.com)
2. Navigate to **Settings → API Keys**
3. Create a new API key or copy an existing one

### 2. Configure Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`):

```bash
# Enable LangSmith tracing
LANGCHAIN_TRACING_V2=true

# Your LangSmith API key (required)
LANGCHAIN_API_KEY=your_api_key_here

# Optional: Project name to organize runs
LANGCHAIN_PROJECT=bioagents

# Optional: Custom endpoint (defaults to https://api.smith.langchain.com)
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

**Note:** You can also use `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` - both naming conventions work.

### 3. Run Your Agents

Simply run your agents as normal:

```bash
python -m bioagents.main
```

LangSmith will automatically:
- Track all LLM calls
- Record token usage
- Monitor agent execution flow
- Log tool calls and responses

### 4. View Traces

1. Visit [https://smith.langchain.com](https://smith.langchain.com)
2. Navigate to **Projects → bioagents** (or your project name)
3. View all runs with:
   - Token usage breakdown
   - Cost estimates
   - Execution traces
   - Agent decision paths

## Token Usage Tracking

### Dual Tracking System

BioAgents uses a **dual tracking system** for maximum reliability:

1. **Local Tracking** (always enabled)
   - Tracks tokens in memory
   - Provides immediate feedback
   - Works offline
   - Printed summary after each run

2. **LangSmith Tracking** (when enabled)
   - Cloud-based tracking
   - Historical data retention
   - Advanced analytics
   - Cost breakdowns by agent/model

### Viewing Token Usage

#### Local Summary

After running agents, you'll see a summary like:

```
============================================================
TOKEN USAGE SUMMARY
============================================================
Total LLM Calls: 5
Total Tokens: 2,345
  - Prompt Tokens: 1,234
  - Completion Tokens: 1,111
Estimated Cost: $0.0023 USD

By Provider:
  openai: 5 calls, 2,345 tokens, $0.0023

By Agent:
  Supervisor: 2 calls, 456 tokens, $0.0005
  Research: 1 calls, 789 tokens, $0.0008
  Analysis: 1 calls, 567 tokens, $0.0006
  Report: 1 calls, 533 tokens, $0.0005
============================================================
```

#### Programmatic Access

```python
from bioagents.llms.token_tracker import get_token_tracker

tracker = get_token_tracker()

# Get statistics
stats = tracker.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")

# Print formatted summary
tracker.print_summary()

# Export to JSON
tracker.export_json("token_usage.json")
```

#### LangSmith Dashboard

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Select your project
3. Click on any run to see:
   - Token usage per LLM call
   - Cost breakdown
   - Execution timeline
   - Agent routing decisions

## Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable/disable LangSmith tracing |
| `LANGCHAIN_API_KEY` | Yes* | - | Your LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `bioagents` | Project name for organizing runs |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | Custom LangSmith endpoint |
| `ENABLE_TOKEN_TRACKING` | No | `true` | Enable/disable local token tracking |

\* Required only if `LANGCHAIN_TRACING_V2=true`

### Alternative Variable Names

LangChain supports both naming conventions:

- `LANGCHAIN_API_KEY` or `LANGSMITH_API_KEY`
- `LANGCHAIN_PROJECT` or `LANGSMITH_PROJECT`
- `LANGCHAIN_ENDPOINT` or `LANGSMITH_ENDPOINT`

## Troubleshooting

### LangSmith Not Tracking

**Problem:** Runs don't appear in LangSmith dashboard.

**Solutions:**
1. Verify `LANGCHAIN_TRACING_V2=true` in your `.env` file
2. Check that `LANGCHAIN_API_KEY` is set correctly
3. Ensure you're logged into the correct LangSmith account
4. Check network connectivity (LangSmith requires internet)

### Token Usage Not Showing

**Problem:** Token counts are zero or missing.

**Solutions:**
1. Ensure `ENABLE_TOKEN_TRACKING=true` (default)
2. Check that your LLM provider returns token usage in responses
3. For Gemini, ensure `google-generativeai` package is installed for fallback counting
4. Check logs for token tracking warnings

### Cost Estimates Incorrect

**Problem:** Cost calculations seem wrong.

**Solutions:**
1. Verify model name matches pricing table in `token_tracker.py`
2. Check that you're using the correct model (some models have similar names)
3. Pricing is per 1M tokens - verify calculations match provider pricing

## Best Practices

1. **Use Projects:** Organize runs by project name (e.g., `bioagents-dev`, `bioagents-prod`)
2. **Monitor Costs:** Regularly check token usage to avoid unexpected costs
3. **Set Rate Limits:** Use `*_RATE_LIMIT` environment variables to prevent API overuse
4. **Export Data:** Periodically export token usage JSON for backup/analysis
5. **Review Traces:** Use LangSmith dashboard to debug agent routing and tool usage

## Disabling Monitoring

To disable LangSmith (but keep local token tracking):

```bash
# In .env file
LANGCHAIN_TRACING_V2=false
```

To disable all token tracking:

```bash
# In .env file
ENABLE_TOKEN_TRACKING=false
LANGCHAIN_TRACING_V2=false
```

## Advanced Usage

### Custom Project Names

Use different project names for different environments:

```bash
# Development
LANGCHAIN_PROJECT=bioagents-dev

# Production
LANGCHAIN_PROJECT=bioagents-prod

# Testing
LANGCHAIN_PROJECT=bioagents-test
```

### Programmatic Configuration

```python
from bioagents.llms.langsmith_config import (
    setup_langsmith_environment,
    get_langsmith_config,
    print_langsmith_status,
)

# Set up environment (call after load_dotenv())
setup_langsmith_environment()

# Get config for graph execution
config = get_langsmith_config()
graph.invoke(state, config=config)

# Check status
print_langsmith_status()
```

## Support

For issues or questions:
- LangSmith Documentation: [https://docs.smith.langchain.com](https://docs.smith.langchain.com)
- LangChain Community: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

