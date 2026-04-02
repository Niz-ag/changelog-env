"""Baseline inference script for ChangelogEnv - ReAct-style agent using OpenAI API.

This script runs a baseline agent against all 3 tasks and prints scores.

Environment variables:
    API_BASE_URL: LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME: Model identifier (required)
    HF_TOKEN: API key (falls back to API_KEY)
"""

import json
import os
import sys
import time
from typing import Optional

from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChangelogAction, ChangelogObservation, ChangelogState

# Configuration from environment
API_BASE_URL = os.environ.get('API_BASE_URL', 'https://router.huggingface.co/v1')
MODEL_NAME = os.environ.get('MODEL_NAME')
HF_TOKEN = os.environ.get('HF_TOKEN', os.environ.get('API_KEY', ''))

if not MODEL_NAME:
    print("ERROR: MODEL_NAME environment variable is required")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Environment base URL (for HF Space deployment)
ENV_BASE_URL = os.environ.get('ENV_BASE_URL', 'http://localhost:7860')

# Maximum steps per episode
MAX_STEPS = 20

# Task IDs to run
TASK_IDS = ['task_easy', 'task_medium', 'task_hard']

# System prompt explaining the action space
SYSTEM_PROMPT = """You are an AI agent that generates changelogs from git commit data.

Your goal is to transform raw commit information into professional, structured release notes.

AVAILABLE ACTIONS:
1. classify_commit - Classify a commit by type
   Required fields: commit_hash (str), label (one of: feature, bugfix, breaking, internal, chore, docs)
   - feature: New user-facing functionality
   - bugfix: Fixes a bug
   - breaking: Removes or changes existing API contract
   - internal: Refactor, perf improvement, infra change (no user impact)
   - chore: Dependency bumps, CI config, tooling
   - docs: Documentation-only changes

2. add_bullet - Add a bullet point to a section
   Required fields: section (str), content (str)
   Sections should be: Features, Bug Fixes, Breaking Changes, etc.

3. remove_bullet - Remove a bullet point by index
   Required fields: section (str), bullet_index (int)

4. set_version - Set the semantic version bump
   Required fields: version_bump (one of: patch, minor, major)
   - patch: Only bug fixes
   - minor: New features (backwards compatible)
   - major: Breaking changes

5. reorder_sections - Set the order of sections in final doc
   Required fields: content (comma-separated section names)

6. submit - End the episode and submit your changelog
   No required fields

7. noop - No operation (debugging only, penalized)

RESPONSE FORMAT:
You must respond with a JSON object representing a ChangelogAction:
{
    "action_type": "classify_commit" | "add_bullet" | "remove_bullet" | "set_version" | "reorder_sections" | "submit" | "noop",
    "commit_hash": string | null,
    "label": string | null,
    "section": string | null,
    "content": string | null,
    "bullet_index": integer | null,
    "version_bump": string | null
}

STRATEGY:
1. First, classify all commits to understand what you're working with
2. Filter out merge commits, internal changes, and chores
3. Write clear, user-facing bullets for features and bug fixes
4. Identify any breaking changes
5. Set the correct version bump based on your analysis
6. Submit when done

Remember: You have a maximum of 20 steps per episode."""


def parse_action(response_text: str) -> Optional[ChangelogAction]:
    """Parse LLM response into a ChangelogAction."""
    try:
        # Try to extract JSON from response
        response_text = response_text.strip()

        # Handle markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith('```json') or line.startswith('```'):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            response_text = '\n'.join(json_lines)

        action_dict = json.loads(response_text)

        # Validate and create action
        return ChangelogAction(**action_dict)

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Failed to parse action: {e}")
        # Fallback to noop on parse failure
        return ChangelogAction(action_type='noop')


def run_episode(task_id: str, verbose: bool = True) -> float:
    """
    Run a single episode for the given task.

    Returns the final score.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {task_id}")
        print(f"{'='*60}")

    # Import client for environment interaction
    from client import ChangelogEnv

    try:
        with ChangelogEnv(base_url=ENV_BASE_URL).sync() as env:
            # Reset environment
            result = env.reset(task_id=task_id)
            observation: ChangelogObservation = result.observation

            if verbose:
                print(f"  Task: {observation.task_id}")
                print(f"  Commits to classify: {len(observation.commits)}")

            step = 0
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            while not result.done and step < MAX_STEPS:
                step += 1

                # Build observation for LLM
                obs_json = json.dumps({
                    'task_id': observation.task_id,
                    'commits': [
                        {
                            'hash': c.hash,
                            'message': c.message,
                            'author': c.author,
                            'files_changed': c.files_changed,
                            'diff_summary': c.diff_summary,
                        }
                        for c in observation.commits
                    ],
                    'draft': observation.draft,
                    'classified': observation.classified,
                    'version_bump': observation.version_bump,
                    'last_action_result': observation.last_action_result,
                    'score_so_far': observation.score_so_far,
                    'step': step,
                }, indent=2)

                # Add observation to messages
                messages.append({
                    "role": "user",
                    "content": f"Current observation (step {step}):\n{obs_json}"
                })

                # Call LLM
                start_time = time.time()
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                )
                elapsed = time.time() - start_time

                action_text = response.choices[0].message.content

                if verbose:
                    print(f"  Step {step}: {action_text[:100]}... ({elapsed:.2f}s)")

                # Parse action
                action = parse_action(action_text)

                # Add assistant response to messages
                messages.append({
                    "role": "assistant",
                    "content": action_text,
                })

                # Take step in environment
                result = env.step(action)
                observation = result.observation

            # Episode complete
            final_score = observation.score_so_far

            if verbose:
                print(f"\n  Episode complete!")
                print(f"  Final score: {final_score:.2f}")
                print(f"  Last action result: {observation.last_action_result[:200]}")

            return final_score

    except Exception as e:
        print(f"  ERROR running episode: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    """Run all tasks and print summary scores."""
    print("="*60)
    print("ChangelogEnv Baseline Inference")
    print("="*60)
    print(f"Environment: {ENV_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print()

    scores = {}
    total_time = time.time()

    for task_id in TASK_IDS:
        task_start = time.time()
        score = run_episode(task_id, verbose=True)
        task_time = time.time() - task_start

        scores[task_id] = score
        print(f"  Time: {task_time:.1f}s")

    total_time = time.time() - total_time

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.2f}")

    # Weighted average (25% easy, 35% medium, 40% hard)
    weighted_avg = (
        scores.get('task_easy', 0) * 0.25 +
        scores.get('task_medium', 0) * 0.35 +
        scores.get('task_hard', 0) * 0.40
    )
    print(f"\n  Weighted Average: {weighted_avg:.2f}")
    print(f"  Total Time: {total_time:.1f}s")

    # Check runtime constraint
    if total_time > 20 * 60:
        print("\n  WARNING: Exceeded 20 minute runtime limit!")

    return scores


if __name__ == '__main__':
    main()
