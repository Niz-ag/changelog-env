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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChangelogAction, ChangelogObservation

# Configuration from environment
API_BASE_URL = os.environ.get('API_BASE_URL', 'https://router.huggingface.co/v1')
MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
HF_TOKEN = os.environ.get('HF_TOKEN', os.environ.get('API_KEY', ''))
ENV_BASE_URL = os.environ.get('ENV_BASE_URL', 'http://localhost:7860')
MAX_STEPS = 20
TASK_IDS = ['task_easy', 'task_medium', 'task_hard']

SYSTEM_PROMPT = """You are an AI agent that generates changelogs from git commit data.

Your goal is to transform raw commit information into professional, structured release notes.

AVAILABLE ACTIONS:
1. classify_commit - Classify a commit by type
   Required fields: commit_hash (str), label (one of: feature, bugfix, breaking, internal, chore, docs)

2. add_bullet - Add a bullet point to a section
   Required fields: section (str), content (str)
   Sections: Features, Bug Fixes, Breaking Changes

3. remove_bullet - Remove a bullet point by index
   Required fields: section (str), bullet_index (int)

4. set_version - Set the semantic version bump
   Required fields: version_bump (one of: patch, minor, major)

5. reorder_sections - Set the order of sections
   Required fields: content (comma-separated section names)

6. submit - End the episode and submit your changelog

7. noop - No operation (penalized)

RESPONSE FORMAT:
{
    "action_type": "classify_commit",
    "commit_hash": "abc123",
    "label": "feature"
}

STRATEGY:
1. Classify all commits first
2. Filter out merge commits, internal changes, chores
3. Write clear user-facing bullets for features and bugfixes
4. Set correct version bump
5. Submit when done (max 20 steps)"""


def parse_action(response_text: str) -> ChangelogAction:
    """Parse LLM response into a ChangelogAction."""
    try:
        response_text = response_text.strip()

        # Strip markdown code blocks
        if '```' in response_text:
            lines = response_text.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith('```'):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = '\n'.join(json_lines).strip()

        # Find JSON object
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            response_text = response_text[start:end+1]

        action_dict = json.loads(response_text)
        return ChangelogAction(**action_dict)

    except Exception as e:
        print(f"  Failed to parse action: {e}")
        return ChangelogAction(action_type='noop')


def run_episode(task_id: str, openai_client, verbose: bool = True) -> float:
    """Run a single episode. Returns the final score."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {task_id}")
        print(f"{'='*60}")

    try:
        from client import ChangelogEnv

        with ChangelogEnv(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(task_id=task_id)
            observation = result.observation

            if verbose:
                print(f"  Commits: {len(observation.commits)}")

            step = 0
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not result.done and step < MAX_STEPS:
                step += 1

                obs_json = json.dumps({
                    'task_id': observation.task_id,
                    'commits': [
                        {
                            'hash': c.hash,
                            'message': c.message,
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

                messages.append({
                    "role": "user",
                    "content": f"Step {step}:\n{obs_json}"
                })

                try:
                    response = openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500,
                    )
                    action_text = response.choices[0].message.content
                except Exception as e:
                    print(f"  LLM call failed: {e}")
                    action_text = '{"action_type": "submit"}'

                if verbose:
                    print(f"  Step {step}: {action_text[:80]}...")

                action = parse_action(action_text)
                messages.append({"role": "assistant", "content": action_text})

                result = env.step(action)
                observation = result.observation

            final_score = observation.score_so_far
            if verbose:
                print(f"  Final score: {final_score:.2f}")
            return final_score

    except Exception as e:
        print(f"  ERROR in episode {task_id}: {e}")
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

    # Initialize OpenAI client inside main with error handling
    try:
        from openai import OpenAI
        openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"WARNING: Could not initialize OpenAI client: {e}")
        openai_client = None

    scores = {}
    total_time = time.time()

    for task_id in TASK_IDS:
        task_start = time.time()
        try:
            score = run_episode(task_id, openai_client, verbose=True)
        except Exception as e:
            print(f"ERROR running {task_id}: {e}")
            score = 0.0
        task_time = time.time() - task_start
        scores[task_id] = score
        print(f"  Time: {task_time:.1f}s")

    total_time = time.time() - total_time

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.2f}")

    weighted_avg = (
        scores.get('task_easy', 0) * 0.25 +
        scores.get('task_medium', 0) * 0.35 +
        scores.get('task_hard', 0) * 0.40
    )
    print(f"\n  Weighted Average: {weighted_avg:.2f}")
    print(f"  Total Time: {total_time:.1f}s")

    return scores


if __name__ == '__main__':
    main()
