"""
Reward functions for ChangelogEnv GRPO training.

These provide additional shaping signals beyond the environment's built-in rewards.
"""

from typing import List, Dict, Any


def reward_final(completions: List[str], **kwargs) -> List[float]:
    """
    Use the final episode reward from the environment.

    This is the primary reward signal - the environment's computed score
    based on classification accuracy, coverage, version correctness, etc.
    """
    rewards = kwargs.get("final_reward")
    if rewards:
        return [float(r) for r in rewards]
    return [0.0] * len(completions)


def reward_classification_accuracy(completions: List[str], **kwargs) -> List[float]:
    """
    Bonus reward for correct commit classifications.

    Encourages the model to properly identify commit types.
    """
    # This reward is already captured in the environment's step rewards
    # which contribute to final_reward. Optional additional shaping:
    return [0.0] * len(completions)


def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    Reward for producing well-formatted JSON actions.

    Penalizes malformed actions that would fail parsing.
    """
    rewards = []
    for completion in completions:
        # Check if completion contains valid JSON-like structure
        if "{" in completion and "}" in completion and "action_type" in completion:
            rewards.append(0.1)  # Small bonus for valid format
        else:
            rewards.append(0.0)
    return rewards


def reward_task_completion(completions: List[str], **kwargs) -> List[float]:
    """
    Reward for completing the task (reaching submit).

    Encourages the model to finish episodes rather than stalling.
    """
    steps_taken = kwargs.get("steps_taken", [])
    rewards = []
    for steps in steps_taken:
        if steps and steps >= 5:  # Minimum meaningful work
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def reward_version_correctness(completions: List[str], **kwargs) -> List[float]:
    """
    Bonus reward for setting correct version bump.

    Version bump is a high-stakes decision that affects the entire changelog.
    """
    # This is captured in environment rewards, but can add extra shaping
    return [0.0] * len(completions)


def compute_weighted_reward(
    final_reward: List[float],
    format_reward: List[float] = None,
    completion_reward: List[float] = None,
    weights: Dict[str, float] = None,
) -> List[float]:
    """
    Compute a weighted combination of multiple reward signals.

    Args:
        final_reward: Environment's final score
        format_reward: Format compliance bonus
        completion_reward: Task completion bonus
        weights: Dict of reward component weights

    Returns:
        List of combined rewards
    """
    if weights is None:
        weights = {
            "final": 1.0,
            "format": 0.1,
            "completion": 0.1,
        }

    combined = []
    n = len(final_reward)

    for i in range(n):
        total = weights["final"] * final_reward[i]

        if format_reward and i < len(format_reward):
            total += weights.get("format", 0.1) * format_reward[i]

        if completion_reward and i < len(completion_reward):
            total += weights.get("completion", 0.1) * completion_reward[i]

        combined.append(total)

    return combined
