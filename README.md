---
title: ChangelogEnv
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - rl-environment
  - changelog
---

# ChangelogEnv

A Release Notes Generation Environment for RL Agents.

## Overview

ChangelogEnv is a real-world OpenEnv-compliant reinforcement learning environment where an AI agent learns to transform raw Git commit data into professional, structured release notes.

The environment models a genuinely multi-step decision process:
1. Classify commits by type (feature/bugfix/breaking/internal/chore/docs)
2. Write changelog bullets for user-facing changes
3. Set the correct semver bump (patch/minor/major)
4. Submit the final changelog

## Installation

```bash
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/changelog-env
```

## Usage

```python
from changelog_env import ChangelogEnv
from models import ChangelogAction

with ChangelogEnv(base_url='https://YOUR_USERNAME-changelog-env.hf.space').sync() as env:
    # Reset with a specific task
    result = env.reset(task_id='task_easy')

    # Classify a commit
    result = env.step(ChangelogAction(
        action_type='classify_commit',
        commit_hash='a1b2c3d',
        label='feature',
    ))

    # Add a bullet point
    result = env.step(ChangelogAction(
        action_type='add_bullet',
        section='Features',
        content='Added new authentication system',
    ))

    # Set version bump
    result = env.step(ChangelogAction(
        action_type='set_version',
        version_bump='minor',
    ))

    # Submit when done
    result = env.step(ChangelogAction(action_type='submit'))

    # Get final state
    state = env.state()
```

## Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| task_easy | Easy | Single PR Summary - 5 commits from a single PR |
| task_medium | Medium | Sprint Release - 18 commits with mixed signals |
| task_hard | Hard | Multi-Version Audit - 52 commits, infer 3 version boundaries |

## Action Space

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `classify_commit` | Classify a commit | `commit_hash`, `label` |
| `add_bullet` | Add bullet to section | `section`, `content` |
| `remove_bullet` | Remove bullet by index | `section`, `bullet_index` |
| `set_version` | Set semver bump | `version_bump` |
| `reorder_sections` | Reorder sections | `content` |
| `submit` | End episode | - |
| `noop` | No operation | - |

## Valid Classification Labels

- `feature` - New user-facing functionality
- `bugfix` - Fixes a bug
- `breaking` - Removes or changes existing API contract
- `internal` - Refactor, perf improvement, infra change
- `chore` - Dependency bumps, CI config, tooling
- `docs` - Documentation-only changes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Primary transport |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger API docs |
| `/web` | GET | Web UI |

## Running Baseline Inference

```bash
export MODEL_NAME=your-model-name
export HF_TOKEN=your-api-key
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

## Development

```bash
# Run tests
pytest tests/

# Build Docker image
docker build -t changelog-env .

# Run locally
docker run -p 7860:7860 changelog-env
```

## License

MIT
