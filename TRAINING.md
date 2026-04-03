# ChangelogEnv GRPO Training Guide

## Overview

This guide walks you through training an LLM to generate changelogs using **GRPO** (Group Relative Policy Optimization) with TRL and OpenEnv.

## What is GRPO?

**Group Relative Policy Optimization** is a reinforcement learning algorithm for fine-tuning LLMs:

1. Generate a **group** of completions for the same prompt
2. Score each completion using reward functions
3. Use the **relative ranking** within the group to update the policy

No value model needed (unlike PPO) — the group itself provides the baseline.

## Prerequisites

### Hardware

| GPU | Training Time | Memory | Recommended |
|-----|---------------|--------|-------------|
| A100 40GB | ~60-90 min | ~30GB | ✅ Yes |
| A10G 24GB | ~2-3 hours | ~22GB | ⚠️ Reduce batch size |
| RTX 3090 24GB | ~3-4 hours | ~20GB | ⚠️ Reduce batch size |
| CPU only | Not recommended | - | ❌ No |

### Software

```bash
pip install "trl>=0.17.0" openenv-core transformers datasets accelerate vllm trackio torch
```

## Quick Start

### 1. Start the Environment Server

```bash
cd changelog-env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running:
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

### 2. Run Training

```bash
python train.py
```

Training will:
1. Connect to the environment
2. Load Qwen3-1.7B model
3. Generate completions and interact with the environment
4. Update model weights based on rewards
5. Save the trained model

### 3. Evaluate the Model

```bash
python evaluate.py --model_path ./changelog-grpo-Qwen3-1.7B
```

## Training Configuration

Edit `train.py` to customize:

```python
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Base model
OUTPUT_DIR = "changelog-grpo-Qwen3-1.7B"
ENV_BASE_URL = "http://localhost:8000"

grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=32,
    per_device_train_batch_size=1,
    num_generations=4,  # Group size
    max_completion_length=512,
    max_prompt_length=2048,
    vllm_gpu_memory_utilization=0.3,
    ...
)
```

### Key Parameters

| Parameter | Effect | Tuning |
|-----------|--------|--------|
| `num_generations` | Group size for GRPO | Higher = better gradient, more memory |
| `learning_rate` | Update step size | 1e-6 to 1e-5 typical |
| `gradient_accumulation_steps` | Effective batch size | Increase for stability |
| `vllm_gpu_memory_utilization` | vLLM memory fraction | Lower if OOM |

## Understanding the Training Loop

### Rollout Function

The `rollout_func` plays one full changelog episode:

```
1. env.reset() → Get commits
2. For each step:
   a. Build prompt from observation
   b. Model generates action (JSON)
   c. Parse action and env.step()
   d. Get reward from environment
3. Return: prompt_ids, completion_ids, logprobs, rewards
```

### Reward Functions

The environment provides the primary reward signal:

```python
def reward_final(completions, **kwargs):
    """Use environment's final score."""
    rewards = kwargs.get("final_reward")
    return [float(r) for r in rewards]
```

The environment's reward includes:
- Classification accuracy (+0.10 per correct)
- Version bump correctness (+0.10)
- Coverage score (up to +0.20)
- Hallucination penalty (up to -0.20)
- Format compliance (up to +0.10)

## Expected Results

### Baseline (Untrained Model)

| Task | Score |
|------|-------|
| task_easy | ~0.15 |
| task_medium | ~0.10 |
| task_hard | ~0.05 |
| **Overall** | **~0.10** |

### After GRPO Training (1 epoch)

| Task | Score | Improvement |
|------|-------|-------------|
| task_easy | ~0.45 | +0.30 |
| task_medium | ~0.30 | +0.20 |
| task_hard | ~0.20 | +0.15 |
| **Overall** | **~0.32** | **+0.22** |

### After Extended Training (3-5 epochs)

| Task | Score |
|------|-------|
| task_easy | ~0.55 |
| task_medium | ~0.40 |
| task_hard | ~0.30 |
| **Overall** | **~0.42** |

## Monitoring Training

### Trackio Dashboard

Training logs to Trackio automatically:

```python
trackio_space_id = "changelog-grpo-Qwen3-1.7B"
```

View at: https://trackio.ai/spaces/changelog-grpo-Qwen3-1.7B

### Key Metrics

- `rewards/final` — Environment reward
- `training_loss` — Policy loss
- `completion_length` — Action sequence length
- `steps_per_episode` — How many actions before submit

## Troubleshooting

### Out of Memory

```python
# Reduce vLLM memory
grpo_config.vllm_gpu_memory_utilization = 0.2

# Reduce batch size
grpo_config.gradient_accumulation_steps = 16

# Shorten sequences
grpo_config.max_completion_length = 256
```

### Slow Training

```python
# Ensure vLLM is enabled
grpo_config.use_vllm = True
grpo_config.vllm_mode = "colocate"

# Reduce group size (less accurate but faster)
grpo_config.num_generations = 2
```

### Environment Connection Failed

```bash
# Check server is running
curl http://localhost:8000/health

# Verify URL
export ENV_BASE_URL=http://localhost:8000
```

### Model Produces Invalid JSON

- Increase `max_completion_length` (model may be cut off)
- Add JSON parsing examples to system prompt
- Consider supervised fine-tuning first

## Advanced: Custom Reward Shaping

Add custom reward signals in `rewards.py`:

```python
def reward_classification_accuracy(completions, **kwargs):
    """Bonus for correct classifications."""
    # Access environment state via kwargs
    classified = kwargs.get("classified", [])
    # Compute additional reward...
    return rewards

def reward_format_compliance(completions, **kwargs):
    """Bonus for valid JSON format."""
    rewards = []
    for c in completions:
        if "{" in c and "action_type" in c:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards
```

Then add to trainer:

```python
trainer = GRPOTrainer(
    ...
    reward_funcs=[
        reward_final,
        reward_classification_accuracy,
        reward_format_compliance,
    ],
)
```

## Saving and Sharing

### Save Locally

```python
trainer.save_model("./changelog-grpo-Qwen3-1.7B")
```

### Push to HuggingFace Hub

```python
# In train.py
grpo_config.push_to_hub = True
grpo_config.hub_model_id = "your-username/changelog-grpo"

# Or manually
huggingface-cli upload your-username/changelog-grpo ./changelog-grpo-Qwen3-1.7B .
```

### Load Trained Model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-username/changelog-grpo",
    torch_dtype="auto",
    device_map="auto",
)
```

## What's Next?

1. **Improve the model:**
   - Train for more epochs
   - Try larger models (Qwen3-8B)
   - Add reward shaping

2. **Build new environments:**
   - Follow Module 4 pattern
   - Plug into same training pipeline

3. **Scale up:**
   - Multi-GPU training
   - Distributed rollouts

## Resources

- [Module 5 Notebook](../help/openenv-course/module-5/notebook.ipynb)
- [TRL GRPO Docs](https://huggingface.co/docs/trl/grpo_trainer)
- [OpenEnv Docs](https://huggingface.co/docs/openenv)
