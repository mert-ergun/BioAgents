# BioAgents

[![Project Status: In Development](https://img.shields.io/badge/status-in%20development-blueviolet.svg)](https://shields.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for our final year BBM479 project, `BioAgents`. The project aims to develop a Multi Agent System for Bioinformatics.

---

## 🚧 Project Status

**This project is currently in the planning and setup phase.** The core features are being defined, and the technology stack is being decided. The primary focus at this stage is to establish a robust and professional development workflow.

---

## 🛠 Technology Stack (To Be Determined)

The technologies for this project have not been finalized yet. They will be listed here once decided.

-   **Frontend:** (TBD)
-   **Backend:** (TBD)
-   **Database:** (TBD)
-   **Deployment:** (TBD)

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

## 👥 Team

| Name                 | GitHub Profile                                |
| -------------------- | --------------------------------------------- |
| Mert ERGUN           | [mert-ergun](https://github.com/mert-ergun)   |
| Arya Zeynep Mete     | [aryazeynep](https://github.com/aryazeynep)   |
| Arın Dal Ahmetoğlu   | [arindalahmetoglu](https://github.com/arindalahmetoglu)|

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.