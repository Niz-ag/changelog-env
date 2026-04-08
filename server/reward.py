"""Reward functions for ChangelogEnv - step-level and terminal rewards."""

from typing import Tuple, Dict, Any, List, Optional
try:
    from .graders import grade, GradingResult
except ImportError:
    from graders import grade, GradingResult


def step_reward(action_type: str, action: Dict[str, Any],
                classified: Dict[str, str], draft: Dict[str, List[str]],
                gold: Dict[str, Any]) -> Tuple[float, str]:
    """
    Compute step-level reward for an action.

    Args:
        action_type: Type of action taken
        action: The full action dict
        classified: Current classifications made by agent
        draft: Current draft changelog
        gold: Gold labels for grading

    Returns:
        Tuple of (reward: float, message: str)
    """
    gold_classifications = gold.get('classifications', {})

    if action_type == 'classify_commit':
        commit_hash = action.get('commit_hash')
        label = action.get('label')

        if not commit_hash or not label:
            return -0.05, "classify_commit requires commit_hash and label"

        # Check if classification is correct
        expected_label = gold_classifications.get(commit_hash)
        if expected_label is None:
            return -0.05, f"Unknown commit hash: {commit_hash}"

        if label == expected_label:
            return +0.10, f"Correctly classified {commit_hash} as {label}"
        else:
            return -0.05, f"Incorrect classification: expected {expected_label}, got {label}"

    elif action_type == 'add_bullet':
        section = action.get('section')
        content = action.get('content')

        if not section or not content:
            return -0.05, "add_bullet requires section and content"

        # Check for well-formed bullet
        if len(content.strip()) < 10:
            return -0.01, "Bullet too short"

        if len(content.strip()) > 500:
            return -0.01, "Bullet too long"

        return +0.01, f"Added bullet to {section}"

    elif action_type == 'remove_bullet':
        section = action.get('section')
        bullet_index = action.get('bullet_index')

        if not section or bullet_index is None:
            return -0.05, "remove_bullet requires section and bullet_index"

        # Small penalty for removing bullets (might be removing good content)
        return -0.05, f"Removed bullet {bullet_index} from {section}"

    elif action_type == 'set_version':
        version_bump = action.get('version_bump')
        expected_version = gold.get('version_bump')

        if not version_bump:
            return -0.05, "set_version requires version_bump"

        if version_bump not in ['patch', 'minor', 'major']:
            return -0.05, f"Invalid version_bump: {version_bump}"

        if version_bump == expected_version:
            return +0.10, f"Correct version bump: {version_bump}"
        else:
            return -0.10, f"Wrong version bump: expected {expected_version}, got {version_bump}"

    elif action_type == 'reorder_sections':
        content = action.get('content')
        if not content:
            return -0.01, "reorder_sections requires content"
        return +0.01, "Sections reordered"

    elif action_type == 'noop':
        return -0.01, "No-op action (penalized to discourage stalling)"

    elif action_type == 'submit':
        # Submit triggers terminal reward calculation
        return 0.0, "Episode submitted - terminal reward calculated separately"

    else:
        return -0.10, f"Unknown action type: {action_type}"


def terminal_reward(draft: Dict[str, List[str]], classified: Dict[str, str],
                    version_bump: str, task_id: str, gold: Dict[str, Any],
                    steps_taken: int) -> Tuple[float, GradingResult]:
    """
    Compute terminal reward at episode end (when submit is called).

    Args:
        draft: Final changelog draft
        classified: Final classifications
        version_bump: Agent's version decision
        task_id: Which task was run
        gold: Gold labels
        steps_taken: Number of steps taken

    Returns:
        Tuple of (total_reward: float, grading_result: GradingResult)
    """
    # Run the grader
    grading_result = grade(task_id, draft, classified, version_bump, gold)

    base_score = grading_result.score

    # Penalty for submitting too early (before step 5)
    early_submit_penalty = 0.0
    if steps_taken < 5:
        early_submit_penalty = 0.20
        grading_result.feedback.append("Early submit penalty: -0.20")

    # Final reward
    final_reward = max(0.01, min(0.99, base_score - early_submit_penalty))

    return final_reward, grading_result


def compute_reward_breakdown(task_id: str, draft: Dict[str, List[str]],
                             classified: Dict[str, str], version_bump: str,
                             gold: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute detailed reward breakdown for analysis.

    Returns dict with individual components of the reward.
    """
    grading_result = grade(task_id, draft, classified, version_bump, gold)

    return {
        'total': grading_result.score,
        **grading_result.breakdown,
    }
