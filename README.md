# BioAgents

[![Project Status: In Development](https://img.shields.io/badge/status-in%20development-blueviolet.svg)](https://shields.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for our final year BBM479 project, `BioAgents`. The project aims to develop a Multi Agent System for Bioinformatics.

---

## 🚧 Project Status

**This project is currently in the planning and setup phase.** The core features are being defined, and the technology stack is being decided. The primary focus at this stage is to establish a robust and professional development workflow.

---

## 🛠 Technology Stack

-   **Agent Framework:** LangGraph + LangChain
-   **LLM Providers:** OpenAI (GPT-4o-mini) / Ollama (local models) / Google Gemini (gemini-2.0-flash-exp)
-   **Tools:** BioPython, UniProt API
-   **Language:** Python 3.12+
-   **Package Manager:** uv

---

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mert-ergun/BioAgents.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd BioAgents
    ```

3. **Install the dependencies:**
    #### uv
    ```bash
    uv sync
    ```

    #### pip
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables:**
    ```bash
    cp .env.example .env
    # Edit .env and add your API keys:
    # - OPENAI_API_KEY (for OpenAI)
    # - GEMINI_API_KEY (for Google Gemini)
    # - LLM_PROVIDER (set to 'openai', 'ollama', or 'gemini')
    # - LANGCHAIN_TRACING_V2 (set to 'true' to enable LangSmith monitoring)
    # - LANGCHAIN_API_KEY (your LangSmith API key for monitoring)
    ```

5. **Configure Pre-Commits for Development (Optional)**
    ```bash
    # After cloning/pulling your changes
    uv sync --all-groups          # Install dependencies including pre-commit
    uv run pre-commit install     # Install the git hooks locally
    ```

---

## ✨ Contribution Guidelines

To ensure a smooth and collaborative development process, we will adhere to the following guidelines. All team members are expected to follow this workflow.

### 1. The Main Branches

-   `main`: This branch is for stable, production-ready code only. **Direct pushes to `main` are disabled.** Merges to `main` will only be done from the `develop` branch after major milestones.
-   `develop`: This is the main development branch. All feature branches are merged into `develop` after a successful code review. This branch should always be in a runnable state.

### 2. General Workflow

1.  **Create an Issue:** All new features, bugs, or tasks must have a corresponding issue in the "Issues" tab.
2.  **Assign the Issue:** Assign the issue to yourself so everyone knows who is working on what.
3.  **Create a New Branch:** Create a new branch from the `develop` branch. **Never work directly on `main` or `develop`.**
    ```bash
    # Make sure you are on the develop branch and have the latest changes
    git checkout develop
    git pull

    # Create your new feature branch
    git checkout -b <branch-type>/<short-description>
    ```
4.  **Commit Your Changes:** Make small, logical commits using the **Conventional Commits** format.
5.  **Create a Pull Request:** Once your work is complete, push your branch to the remote repository and open a Pull Request (PR) to merge into `develop`.
6.  **Code Review:** At least **one** other team member must review and approve the PR.
7.  **Merge:** Once approved, the PR can be merged into `develop`.

### 3. Branch Naming Convention

Use the following prefixes for your branches to clearly identify their purpose:

-   `feature/`: For new features (e.g., `feature/agent-workflow`)
-   `fix/`: For bug fixes (e.g., `fix/research-tool-issue`)
-   `docs/`: For changes to documentation (e.g., `docs/update-readme`)
-   `chore/`: For maintenance tasks, like updating dependencies (e.g., `chore/update-python-version`)

### 4. Conventional Commits

We use the Conventional Commits specification for our commit messages. This creates an explicit and readable commit history.

The format is: `<type>: <subject>`

**Common Types:**

-   **`feat`**: A new feature.
-   **`fix`**: A bug fix.
-   **`docs`**: Changes to documentation only.
-   **`style`**: Code style changes that do not affect the meaning of the code (e.g., formatting, semi-colons).
-   **`refactor`**: A code change that neither fixes a bug nor adds a feature.
-   **`test`**: Adding missing tests or correcting existing tests.
-   **`chore`**: Changes to the build process or auxiliary tools and libraries.

**Example Commit:**

```bash
git commit -m "feat: Add user authentication endpoint"
git commit -m "fix: Resolve validation error on registration form"
```

### 5. Pull Requests (PRs)

-   A PR is the only way to get code into the `develop` branch.
-   Use a clear and descriptive title.
-   In the description, link the issue that your PR resolves by writing `Closes #[issue_number]`.
-   If there are visual changes, include screenshots or GIFs.
-   Request a review from your teammates by using the "Reviewers" panel on the right.
-   **Do not merge your own PR** until it has been approved by at least one other person.

---

## 📊 Monitoring & Observability

### LangSmith Integration

BioAgents includes robust LangSmith integration for monitoring agent execution, tracking token usage, and debugging workflows. LangSmith provides:

- **Real-time monitoring** of all agent interactions
- **Token usage tracking** with cost estimation
- **Execution traces** showing the complete workflow
- **Performance metrics** for each agent and tool call

#### Setup

1. **Get your LangSmith API key:**
   - Sign up at [https://smith.langchain.com](https://smith.langchain.com)
   - Navigate to Settings → API Keys
   - Copy your API key

2. **Configure environment variables:**
   ```bash
   # In your .env file
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=bioagents  # Optional: organize runs by project
   ```

3. **Run your agents:**
   ```bash
   python -m bioagents.main
   ```

4. **View traces:**
   - Visit [https://smith.langchain.com](https://smith.langchain.com)
   - All agent runs will appear in your dashboard
   - Filter by project, agent, or time range

#### Token Usage Tracking

Token usage is automatically tracked in LangSmith:

- **LangSmith tracking:** Available in the dashboard with detailed breakdowns
- **Cost estimation:** Automatic cost calculation based on provider pricing
> **Note:** Local token tracking is not yet available. This feature is planned for a future release.

#### Disabling Monitoring

To disable LangSmith monitoring:
```bash
# In your .env file
LANGCHAIN_TRACING_V2=false
```

Token tracking will still work locally even if LangSmith is disabled.

---

## 📚 Documentation

- [Prompt Engineering](docs/PROMPT_ENGINEERING.md) - XML-based prompt system
- [Development Guide](docs/DEVELOPMENT.md) - Contributing to the project

---

## 👥 Team

| Name                 | GitHub Profile                                |
| -------------------- | --------------------------------------------- |
| Mert ERGUN           | [mert-ergun](https://github.com/mert-ergun)   |
| Arya Zeynep Mete     | [aryazeynep](https://github.com/aryazeynep)   |
| Arın Dalahmetoğlu    | [arindalahmetoglu](https://github.com/arindalahmetoglu)|

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
