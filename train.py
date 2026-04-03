#!/usr/bin/env python3
"""
Module 5: Train a Changelog Generator with GRPO

Fine-tune Qwen3-1.7B to generate changelogs from git commits using GRPO
(Group Relative Policy Optimization) via TRL and OpenEnv.

Time: ~90 min (training) · Difficulty: Advanced · GPU: A100 required

Usage:
    python train.py

Requirements:
    pip install "trl>=0.17.0" openenv-core transformers datasets accelerate vllm trackio
"""

import sys
import os
from collections import defaultdict

# Setup path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from trl.experimental.openenv import generate_rollout_completions

from client import ChangelogEnv
from models import ChangelogAction, ChangelogObservation

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "changelog-grpo-Qwen3-1.7B"
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# Task prompts for training
TASK_PROMPTS = {
    "task_easy": "Generate a changelog summary for this single PR.",
    "task_medium": "Generate release notes for this sprint with mixed commits.",
    "task_hard": "Generate a multi-version changelog audit from these commits.",
}

# =============================================================================
# 1. Initialize Environment
# =============================================================================

print("=" * 60)
print("Step 1: Connecting to ChangelogEnv")
print("=" * 60)

# Create persistent sync client for training
env = ChangelogEnv(base_url=ENV_BASE_URL)
sync_env = env.sync()
sync_env.connect()

# Verify connection
try:
    result = sync_env.reset()
    print(f"Connected to: {ENV_BASE_URL}")
    print(f"Task: {result.observation.task_id}")
    print(f"Commits available: {len(result.observation.commits)}")
except Exception as e:
    print(f"ERROR: Could not connect to environment: {e}")
    print("Make sure the server is running: uvicorn server.app:app --host 0.0.0.0 --port 8000")
    sys.exit(1)

# =============================================================================
# 2. Initialize Tokenizer
# =============================================================================

print("\n" + "=" * 60)
print("Step 2: Loading Model and Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"Model: {MODEL_NAME}")
print(f"Pad token: {tokenizer.pad_token}")

# =============================================================================
# 3. Define System Prompt
# =============================================================================

print("\n" + "=" * 60)
print("Step 3: System Prompt")
print("=" * 60)

SYSTEM_PROMPT = """You are an expert changelog generator for software projects.

## YOUR TASK

Transform raw git commit data into professional, structured release notes.

## ACTION SPACE

You have 7 action types available:

1. **classify_commit** - Label a commit by type
   - feature: New user-facing functionality
   - bugfix: Fixes a bug
   - breaking: Removes or changes API contract
   - internal: Refactor/perf change (no user impact)
   - chore: Dependencies, CI, tooling
   - docs: Documentation only

2. **add_bullet** - Add a bullet point to a section
   - Sections: Features, Bug Fixes, Breaking Changes

3. **remove_bullet** - Remove a bullet by index

4. **set_version** - Set semver bump (patch/minor/major)
   - patch: Only bugfixes
   - minor: New features (backwards compatible)
   - major: Breaking changes

5. **reorder_sections** - Set section order

6. **submit** - Finalize and submit changelog

7. **noop** - No operation (avoid using)

## RESPONSE FORMAT

Respond with a JSON action object:
{
    "action_type": "classify_commit",
    "commit_hash": "abc123",
    "label": "feature"
}

## STRATEGY

1. First, classify all commits to understand the changes
2. Filter out merge commits, internal changes, and chores
3. Write clear, user-facing bullets for features and bugfixes
4. Identify any breaking changes
5. Set the correct version bump
6. Submit when done (max 20 steps)

## EXAMPLE ACTIONS

Classify a commit:
{"action_type": "classify_commit", "commit_hash": "a1b2c3d", "label": "feature"}

Add a bullet:
{"action_type": "add_bullet", "section": "Features", "content": "Added user authentication"}

Set version:
{"action_type": "set_version", "version_bump": "minor"}

Submit:
{"action_type": "submit"}
"""

print("System prompt defined for changelog generation.")

# =============================================================================
# 4. Helper Functions
# =============================================================================

print("\n" + "=" * 60)
print("Step 4: Helper Functions")
print("=" * 60)


def format_observation(obs: ChangelogObservation) -> str:
    """Format observation as text for the model."""
    lines = []

    lines.append(f"Task: {obs.task_id}")
    lines.append(f"Step: {obs.score_so_far:.2f}")
    lines.append("")

    # Commits summary
    lines.append(f"Total commits: {len(obs.commits)}")
    lines.append("Commits to classify:")
    for commit in obs.commits:
        status = obs.classified.get(commit.hash, "UNCLASSIFIED")
        lines.append(f"  [{commit.hash[:7]}] {commit.message[:60]}... -> {status}")

    lines.append("")
    lines.append(f"Current draft: {obs.draft}")
    lines.append(f"Version bump: {obs.version_bump or 'Not set'}")
    lines.append(f"Last action: {obs.last_action_result}")

    return "\n".join(lines)


def extract_action(text: str) -> ChangelogAction:
    """Extract JSON action from model output."""
    import json
    import re

    # Try to find JSON in the response
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            action_dict = json.loads(json_match.group())
            return ChangelogAction(**action_dict)
        except (json.JSONDecodeError, Exception):
            pass

    # Fallback to noop
    return ChangelogAction(action_type="noop")


def make_user_prompt(task_prompt: str, obs: ChangelogObservation) -> str:
    """Build structured prompt from task and observation."""
    obs_text = format_observation(obs)
    return f"{task_prompt}\n\n{obs_text}\n\nRespond with your next action as JSON."


print("Helper functions defined.")

# =============================================================================
# 5. Rollout Function
# =============================================================================

print("\n" + "=" * 60)
print("Step 5: Rollout Function")
print("=" * 60)


def rollout_once(trainer, sync_env, tokenizer, task_prompt, system_prompt, max_steps=20):
    """
    Execute one full changelog generation episode.

    Returns dict with prompt_ids, completion_ids, logprobs, and rewards.
    """
    # Reset environment with a random task
    import random
    task_id = random.choice(list(TASK_PROMPTS.keys()))
    result = sync_env.reset(task_id=task_id)
    observation = result.observation

    prompt_ids_list = []
    completion_ids_list = []
    logprobs_list = []
    rewards = []

    actions_taken = []

    for step in range(max_steps):
        if result.done:
            break

        # Build prompt
        user_prompt = make_user_prompt(task_prompt, observation)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate action from model
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids_list.append(rollout_outputs["prompt_ids"])
        completion_ids_list.append(rollout_outputs["completion_ids"])
        logprobs_list.append(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse and execute action
        action = extract_action(completion_text)
        actions_taken.append(action.action_type)

        result = sync_env.step(action)
        observation = result.observation

        # Store step reward
        if result.reward is not None:
            rewards.append(result.reward)

    # Compute final reward (episode return)
    final_reward = observation.score_so_far

    # Flatten all prompt/completion ids
    all_prompt_ids = [pid for sublist in prompt_ids_list for pid in sublist]
    all_completion_ids = [cid for sublist in completion_ids_list for cid in sublist]
    all_logprobs = [lp for sublist in logprobs_list for lp in sublist]

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "final_reward": final_reward,
        "steps_taken": len(actions_taken),
        "actions": actions_taken,
    }


def rollout_func(prompts, trainer=None):
    """
    Rollout function called by GRPOTrainer.

    Args:
        prompts: List of prompt strings (one per generation in the batch)
        trainer: GRPOTrainer instance

    Returns:
        Dict with lists of prompt_ids, completion_ids, logprobs, and rewards
    """
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    final_rewards = []

    for prompt_text in prompts:
        episode = rollout_once(
            trainer=trainer,
            sync_env=sync_env,
            tokenizer=tokenizer,
            task_prompt=prompt_text,
            system_prompt=SYSTEM_PROMPT,
            max_steps=20,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        final_rewards.append(episode["final_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "final_reward": final_rewards,
    }


print("Rollout functions defined.")

# =============================================================================
# 6. Define Reward Functions
# =============================================================================

print("\n" + "=" * 60)
print("Step 6: Reward Functions")
print("=" * 60)


def reward_final(completions, **kwargs):
    """Use the final episode reward from the environment."""
    rewards = kwargs.get("final_reward")
    if rewards:
        return [float(r) for r in rewards]
    return [0.0] * len(completions)


def reward_classification(completions, **kwargs):
    """
    Bonus reward for correct classifications.
    This is a shaping reward to encourage proper commit classification.
    """
    # The environment already rewards classifications in step rewards
    # which contribute to final_reward, so this is optional additional shaping
    return [0.0] * len(completions)  # Placeholder for custom shaping


print("Reward functions defined:")
print("  - reward_final: Uses environment's final score")
print("  - reward_classification: Optional shaping reward")

# =============================================================================
# 7. Create Dataset
# =============================================================================

print("\n" + "=" * 60)
print("Step 7: Creating Dataset")
print("=" * 60)

# Create dataset with task prompts
dataset_size = 500  # Adjust based on training needs
dataset = Dataset.from_dict({
    "prompt": [
        "Generate a changelog from git commits."
    ] * dataset_size
})

print(f"Dataset created: {len(dataset)} prompts")
print(f"Sample prompt: {dataset[0]['prompt'][:50]}...")

# =============================================================================
# 8. Configure GRPO Training
# =============================================================================

print("\n" + "=" * 60)
print("Step 8: GRPO Configuration")
print("=" * 60)

grpo_config = GRPOConfig(
    # Training
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=32,
    per_device_train_batch_size=1,
    warmup_steps=10,

    # GRPO specific
    num_generations=4,  # Group size for GRPO
    beta=0.04,  # KL penalty

    # Sequence lengths
    max_completion_length=512,  # Changelog actions are longer than Wordle guesses
    max_prompt_length=2048,     # Commit data can be lengthy

    # vLLM for fast generation
    use_vllm=True,
    vllm_mode="colocate",  # Generation + training on same GPU
    vllm_gpu_memory_utilization=0.3,

    # Output
    output_dir=OUTPUT_DIR,
    report_to="trackio",
    trackio_space_id=OUTPUT_DIR,

    # Logging and saving
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,

    # Optimization
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Hub
    push_to_hub=False,  # Set True to push to HF Hub
    hub_model_id=None,
)

print(f"Output directory: {OUTPUT_DIR}")
print(f"vLLM mode: {grpo_config.vllm_mode}")
print(f"Num generations (group size): {grpo_config.num_generations}")
print(f"Max completion length: {grpo_config.max_completion_length}")

# =============================================================================
# 9. Create Trainer
# =============================================================================

print("\n" + "=" * 60)
print("Step 9: Creating GRPOTrainer")
print("=" * 60)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    processing_class=tokenizer,
    reward_funcs=[
        reward_final,
        reward_classification,
    ],
    train_dataset=dataset,
    args=grpo_config,
    rollout_func=rollout_func,
)

print(f"Trainer created with model: {MODEL_NAME}")

# =============================================================================
# 10. Check GPU and Start Training
# =============================================================================

print("\n" + "=" * 60)
print("Step 10: GPU Check")
print("=" * 60)

if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}")
    print(f"Total memory: {max_memory} GB")
    print(f"Reserved memory: {start_gpu_memory} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")
    print("Consider using Colab Pro or a cloud GPU instance.")

# =============================================================================
# 11. Train
# =============================================================================

print("\n" + "=" * 60)
print("Step 11: Starting Training")
print("=" * 60)
print("This will take approximately 60-90 minutes on an A100 GPU.")
print("Press Ctrl+C to stop early.")
print("=" * 60)

try:
    trainer_stats = trainer.train()

    # Memory stats after training
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_for_training = round(used_memory - start_gpu_memory, 3)
        print(f"\nPeak memory: {used_memory} GB")
        print(f"Memory for training: {used_for_training} GB")

    print(f"\nTraining time: {round(trainer_stats.metrics['train_runtime'] / 60, 1)} minutes")
    print(f"Training completed successfully!")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# 12. Save Model
# =============================================================================

print("\n" + "=" * 60)
print("Step 12: Saving Model")
print("=" * 60)

# Close environment connection
sync_env.close()

trainer.save_model(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")

if grpo_config.push_to_hub and grpo_config.hub_model_id:
    trainer.push_to_hub()
    print(f"Model pushed to Hub: {grpo_config.hub_model_id}")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"""
Next steps:
1. Evaluate the model: python evaluate.py
2. Load and use:
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("{OUTPUT_DIR}")
3. Push to Hub (optional):
   huggingface-cli upload {grpo_config.hub_model_id or 'your-repo'} {OUTPUT_DIR} .
""")
