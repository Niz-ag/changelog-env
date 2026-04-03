# ChangelogEnv Setup Guide

## Overview

ChangelogEnv is an OpenEnv-compliant RL environment for training LLMs to generate changelogs from git commits.

---

## Quick Start

```bash
cd changelog-env
pip install -r requirements.txt
```

---

## Part 1: Local Development with Uvicorn

The fastest iteration loop:

```bash
# Install dependencies
pip install -r requirements.txt

# Run server with auto-reload
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

Connect from Python:
```python
from client import ChangelogEnv
from models import ChangelogAction

with ChangelogEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id='task_easy')
    print(f"Commits: {len(result.observation.commits)}")
```

---

## Part 2: Docker Deployment

### Build from source:
```bash
cd changelog-env
docker build -t changelog-env:latest -f Dockerfile .
docker run -d -p 8000:8000 changelog-env:latest
```

### With environment variables:
```bash
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    changelog-env:latest
```

---

## Part 3: Deploy to HF Spaces

### Using `openenv push` (recommended):

```bash
# Install openenv CLI
pip install openenv-core

# Push to HF Spaces
cd changelog-env
openenv push --repo-id YOUR_USERNAME/changelog-env
```

Your environment is now live:
- **API endpoint:** `https://YOUR_USERNAME-changelog-env.hf.space`
- **Web UI:** `https://YOUR_USERNAME-changelog-env.hf.space/web`
- **API docs:** `https://YOUR_USERNAME-changelog-env.hf.space/docs`
- **Health check:** `https://YOUR_USERNAME-changelog-env.hf.space/health`

### Manual Git Push:

```bash
cd changelog-env
git init
git add .
git commit -m "Initial ChangelogEnv"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/changelog-env
git push -u origin main
```

### Install as pip package from Space:

```bash
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/changelog-env
```

---

## Part 4: Baseline Inference (Optional)

Test with an LLM before training:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_huggingface_token"
export ENV_BASE_URL="http://localhost:8000"

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference
python inference.py
```

---

## Part 5: GRPO Training with TRL (Module 5)

Train your own model using environment rewards.

### Install Training Dependencies

```bash
pip install -r requirements-training.txt
```

### Start the Environment

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Training

```bash
python train.py
```

Training will:
1. Connect to the environment
2. Load Qwen3-1.7B model
3. Generate completions and interact with the environment
4. Update model weights based on rewards
5. Save the trained model to `./changelog-grpo-Qwen3-1.7B`

### Evaluate the Trained Model

```bash
python evaluate.py --model_path ./changelog-grpo-Qwen3-1.7B
```

### Training Configuration

Edit `train.py` to customize:

```python
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Base model
OUTPUT_DIR = "changelog-grpo-Qwen3-1.7B"

grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    num_generations=4,  # Group size
    max_completion_length=512,
    vllm_gpu_memory_utilization=0.3,
    ...
)
```

### Hardware Requirements

| GPU | Training Time | Memory |
|-----|---------------|--------|
| A100 40GB | ~60-90 min | ~30GB |
| A10G 24GB | ~2-3 hours | ~22GB |

See [TRAINING.md](TRAINING.md) for detailed training documentation.

---

## File Structure

```
changelog-env/
├── models.py              # Pydantic models
├── client.py              # OpenEnv client
├── inference.py           # Baseline inference (OpenAI API)
├── train.py               # GRPO training script (Module 5)
├── evaluate.py            # Model evaluation
├── rewards.py             # Reward functions for training
├── server/
│   ├── environment.py     # ChangelogEnvironment
│   ├── app.py             # FastAPI app
│   ├── tasks.py           # Task definitions
│   ├── graders.py         # Scoring functions
│   └── reward.py          # Reward functions
├── tests/                 # Unit tests
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile             # Container definition
├── requirements.txt       # Core dependencies
├── requirements-training.txt  # Training dependencies
└── TRAINING.md            # Training documentation
```

---

## The 3-Component Pattern (Module 4)

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **models.py** | Type definitions | `ChangelogAction`, `ChangelogObservation`, `ChangelogState` |
| **client.py** | HTTP/WebSocket client | `ChangelogEnv(EnvClient)` |
| **server/environment.py** | Game logic | `ChangelogEnvironment(Environment)` |

All models extend `openenv.core.env_server` base classes.

---

## Troubleshooting

### Import Errors
```bash
cd changelog-env
export PYTHONPATH=.
```

### GPU Memory Issues (Training)
```python
grpo_config.vllm_gpu_memory_utilization = 0.2
grpo_config.gradient_accumulation_steps = 64
```

### Environment Connection Failed
```bash
# Check server
curl http://localhost:8000/health

# Check URL
echo $ENV_BASE_URL
```

---

## Resources

- [TRAINING.md](TRAINING.md) — Complete GRPO training guide
- [Module 4 Course](../help/openenv-course/module-4/README.md) — Building environments
- [Module 5 Course](../help/openenv-course/module-5/README.md) — GRPO training
- [OpenEnv Documentation](https://huggingface.co/docs/openenv)
- [TRL GRPO Documentation](https://huggingface.co/docs/trl/grpo_trainer)
