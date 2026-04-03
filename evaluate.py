#!/usr/bin/env python3
"""
Evaluate a trained ChangelogEnv model.

Usage:
    python evaluate.py --model_path ./changelog-grpo-Qwen3-1.7B
    python evaluate.py --model_path Qwen/Qwen3-1.7B  # Base model for comparison
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from client import ChangelogEnv
from models import ChangelogAction

# =============================================================================
# Configuration
# =============================================================================

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MAX_STEPS = 20

SYSTEM_PROMPT = """You are an expert changelog generator for software projects.

## YOUR TASK

Transform raw git commit data into professional, structured release notes.

## ACTION SPACE

1. classify_commit - Label commit (feature/bugfix/breaking/internal/chore/docs)
2. add_bullet - Add bullet to section (Features, Bug Fixes, Breaking Changes)
3. set_version - Set semver bump (patch/minor/major)
4. submit - Finalize changelog

## RESPONSE FORMAT

Respond with JSON only:
{"action_type": "classify_commit", "commit_hash": "abc", "label": "feature"}
"""

# =============================================================================
# Helper Functions
# =============================================================================


def extract_action(text: str) -> ChangelogAction:
    """Extract JSON action from model output."""
    import re

    # Try to find JSON in response
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            action_dict = json.loads(json_match.group())
            return ChangelogAction(**action_dict)
        except (json.JSONDecodeError, Exception):
            pass

    return ChangelogAction(action_type="noop")


def format_observation(obs) -> str:
    """Format observation as text."""
    lines = []
    lines.append(f"Task: {obs.task_id}")
    lines.append(f"Commits: {len(obs.commits)}")
    lines.append(f"Classified: {obs.classified}")
    lines.append(f"Draft: {obs.draft}")
    lines.append(f"Version: {obs.version_bump or 'Not set'}")
    lines.append(f"Last: {obs.last_action_result}")
    return "\n".join(lines)


def run_episode(model, tokenizer, sync_env, task_id: str, verbose: bool = True) -> dict:
    """Run one evaluation episode."""

    result = sync_env.reset(task_id=task_id)
    observation = result.observation

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    actions_taken = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        if result.done:
            break

        # Build prompt
        obs_text = format_observation(observation)
        user_prompt = f"Generate changelog.\n\n{obs_text}\n\nRespond with JSON action."

        messages.append({"role": "user", "content": user_prompt})

        # Generate
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,  # Greedy for evaluation
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Parse and execute action
        action = extract_action(generated_text)
        actions_taken.append(action.action_type)

        result = sync_env.step(action)
        observation = result.observation

        if result.reward is not None:
            total_reward += result.reward

        if verbose:
            print(f"  Step {step + 1}: {action.action_type} -> reward={result.reward}")

    if verbose:
        print(f"\n  Final score: {observation.score_so_far:.2f}")
        print(f"  Actions: {actions_taken}")
        print(f"  Draft: {observation.draft}")

    return {
        "task_id": task_id,
        "final_score": observation.score_so_far,
        "total_reward": total_reward,
        "steps_taken": len(actions_taken),
        "actions": actions_taken,
        "draft": observation.draft,
        "version_bump": observation.version_bump,
    }


def evaluate_model(model, tokenizer, env, tasks: list = None, num_episodes: int = 3) -> dict:
    """Evaluate model on all tasks."""

    if tasks is None:
        tasks = ["task_easy", "task_medium", "task_hard"]

    results = {}

    with env.sync() as sync_env:
        for task_id in tasks:
            print(f"\n{'='*60}")
            print(f"Evaluating {task_id}")
            print(f"{'='*60}")

            task_results = []
            for episode in range(num_episodes):
                print(f"\n  Episode {episode + 1}/{num_episodes}")
                result = run_episode(
                    model, tokenizer, sync_env, task_id, verbose=True
                )
                task_results.append(result)

            # Aggregate
            avg_score = sum(r["final_score"] for r in task_results) / len(task_results)
            results[task_id] = {
                "avg_score": avg_score,
                "episodes": task_results,
            }

            print(f"\n  {task_id} Average Score: {avg_score:.2f}")

    # Overall summary
    all_scores = [r["avg_score"] for r in results.values()]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    results["overall"] = {"avg_score": overall_avg}

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChangelogEnv model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./changelog-grpo-Qwen3-1.7B",
        help="Path to trained model or model name",
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default=ENV_BASE_URL,
        help="Environment URL",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ChangelogEnv Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Environment: {args.env_url}")
    print(f"Episodes per task: {args.episodes}")

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Model loaded: {args.model_path}")

    # Initialize environment
    env = ChangelogEnv(base_url=args.env_url)

    # Evaluate
    results = evaluate_model(
        model, tokenizer, env,
        num_episodes=args.episodes,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        if task_id in results:
            print(f"  {task_id}: {results[task_id]['avg_score']:.2f}")

    print(f"\n  Overall Average: {results['overall']['avg_score']:.2f}")

    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
