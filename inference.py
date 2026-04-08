"""Baseline inference script for ChangelogEnv."""

import json
import os
import sys
import time
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ChangelogAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "changelog-env"
MAX_STEPS = 20
TASK_IDS = ["task_easy", "task_medium", "task_hard"]
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an AI agent that generates changelogs from git commit data.
Respond with a single JSON action object only.

ACTIONS:
- {"action_type": "classify_commit", "commit_hash": "abc", "label": "feature"}
  Labels: feature, bugfix, breaking, internal, chore, docs
- {"action_type": "add_bullet", "section": "Features", "content": "Added X"}
- {"action_type": "set_version", "version_bump": "minor"}
  Values: patch, minor, major
- {"action_type": "submit"}

STRATEGY: classify all commits, add bullets for user-facing changes, set version, submit."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_action(text: str) -> ChangelogAction:
    try:
        text = text.strip()
        if "```" in text:
            lines, json_lines, in_block = text.split("\n"), [], False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines).strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            text = text[s:e + 1]
        return ChangelogAction(**json.loads(text))
    except Exception:
        return ChangelogAction(action_type="noop")


def get_action(client: OpenAI, messages: list) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return '{"action_type": "submit"}'


def run_episode(task_id: str, client: OpenAI) -> float:
    from client import ChangelogEnv

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with ChangelogEnv(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(task_id=task_id)
            observation = result.observation
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_json = json.dumps({
                    "task_id": observation.task_id,
                    "commits": [
                        {"hash": c.hash, "message": c.message, "diff_summary": c.diff_summary}
                        for c in observation.commits
                    ],
                    "draft": observation.draft,
                    "classified": observation.classified,
                    "version_bump": observation.version_bump,
                    "last_action_result": observation.last_action_result,
                    "score_so_far": observation.score_so_far,
                }, indent=2)

                messages.append({"role": "user", "content": f"Step {step}:\n{obs_json}"})
                action_text = get_action(client, messages)
                messages.append({"role": "assistant", "content": action_text})

                action = parse_action(action_text)
                result = env.step(action)
                observation = result.observation

                reward = result.reward or 0.0
                done = result.done
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action.action_type, reward=reward, done=done, error=None)

                if done:
                    break

            score = min(max(observation.score_so_far, 0.01), 0.99)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    print("=" * 60, flush=True)
    print("ChangelogEnv Baseline Inference", flush=True)
    print(f"Environment: {ENV_BASE_URL}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print("=" * 60, flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"[DEBUG] OpenAI client init failed: {e}", flush=True)
        client = None

    scores = {}
    total_start = time.time()

    for task_id in TASK_IDS:
        try:
            score = run_episode(task_id, client)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
            score = 0.0
        scores[task_id] = score

    total_time = time.time() - total_start

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.3f}", flush=True)

    weighted = (
        scores.get("task_easy", 0) * 0.25 +
        scores.get("task_medium", 0) * 0.35 +
        scores.get("task_hard", 0) * 0.40
    )
    print(f"  Weighted Average: {weighted:.3f}", flush=True)
    print(f"  Total Time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
