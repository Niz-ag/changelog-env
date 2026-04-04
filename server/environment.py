"""ChangelogEnvironment - the core RL environment for changelog generation."""

import sys
import os
import uuid
from datetime import datetime
from typing import Optional

# Setup path for imports
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)
_server_dir = os.path.join(_base_dir, 'server')
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from openenv.core.env_server import Environment

from models import ChangelogAction, ChangelogObservation, ChangelogState, CommitRecord
from tasks import load_task
from reward import step_reward, terminal_reward


class ChangelogEnvironment(Environment):
    """
    Environment for training agents to generate changelogs from git commits.

    The agent must:
    1. Classify commits by type (feature/bugfix/breaking/internal/chore/docs)
    2. Write changelog bullets for user-facing changes
    3. Set the correct semver bump (patch/minor/major)
    4. Submit the final changelog

    Note: state is a @property (not a method) as per openenv.core spec.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state: Optional[ChangelogState] = None
        self._task: Optional[dict] = None
        self._draft: dict = {}
        self._classified: dict = {}
        self._version_bump: Optional[str] = None
        self._score: float = 0.0
        self._last_action_result: str = ''

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              **kwargs) -> ChangelogObservation:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier
            **kwargs: Additional args including task_id

        Returns:
            Initial ChangelogObservation
        """
        task_id = kwargs.get('task_id', 'task_easy')

        # Load the task
        self._task = load_task(task_id, seed=seed)

        # Reset episode state
        self._draft = {}
        self._classified = {}
        self._version_bump = None
        self._score = 0.0
        self._last_action_result = ''

        # Create new state
        self._state = ChangelogState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_attempts=20,
            started_at=datetime.utcnow().isoformat() + 'Z',
        )

        return self._build_observation(done=False, reward=None, result='Episode started.')

    def step(self, action: ChangelogAction, timeout_s: Optional[float] = None,
             **kwargs) -> ChangelogObservation:
        """
        Take one step in the environment.

        Args:
            action: The action to take
            timeout_s: Optional timeout in seconds
            **kwargs: Additional args

        Returns:
            ChangelogObservation with updated state
        """
        # Check if environment was reset
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step()")

        # Increment step count
        self._state.step_count += 1

        # Apply the action and get reward
        reward, msg = self._apply_action(action)

        # Update score
        self._score += reward

        # Check if episode is done
        done = (
            action.action_type == 'submit'
            or self._state.step_count >= self._state.max_attempts
        )

        # If max steps reached without submit, auto-submit
        if self._state.step_count >= self._state.max_attempts and action.action_type != 'submit':
            done = True
            # Apply terminal reward for auto-submit
            terminal_r, grading = terminal_reward(
                self._draft, self._classified, self._version_bump,
                self._state.task_id, self._task['gold'], self._state.step_count
            )
            self._score += terminal_r
            # Use terminal reward in observation, not step reward
            reward = terminal_r
            msg = f"Auto-submitted at max steps. Final score: {self._score:.2f}"

        return self._build_observation(done=done, reward=reward, result=msg)

    @property
    def state(self) -> ChangelogState:
        """Return current environment state. This is a @property, not a method."""
        return self._state

    def _apply_action(self, action: ChangelogAction) -> tuple:
        """
        Apply an action and return (reward, message).

        Args:
            action: The action to apply

        Returns:
            Tuple of (reward: float, message: str)
        """
        action_type = action.action_type

        if action_type == 'classify_commit':
            return self._apply_classify(action)
        elif action_type == 'add_bullet':
            return self._apply_add_bullet(action)
        elif action_type == 'remove_bullet':
            return self._apply_remove_bullet(action)
        elif action_type == 'set_version':
            return self._apply_set_version(action)
        elif action_type == 'reorder_sections':
            return self._apply_reorder_sections(action)
        elif action_type == 'submit':
            return self._apply_submit()
        elif action_type == 'noop':
            return -0.01, "No-op action"
        else:
            return -0.10, f"Unknown action type: {action_type}"

    def _apply_classify(self, action: ChangelogAction) -> tuple:
        """Apply classify_commit action."""
        commit_hash = action.commit_hash
        label = action.label

        if not commit_hash or not label:
            return -0.05, "classify_commit requires commit_hash and label"

        valid_labels = ['feature', 'bugfix', 'breaking', 'internal', 'chore', 'docs']
        if label not in valid_labels:
            return -0.05, f"Invalid label: {label}. Must be one of {valid_labels}"

        # Store classification (can be updated if wrong)
        self._classified[commit_hash] = label
        self._last_action_result = f"Classified {commit_hash} as {label}"

        # Compute step reward
        return step_reward('classify_commit', action.model_dump(),
                          self._classified, self._draft, self._task['gold'])

    def _apply_add_bullet(self, action: ChangelogAction) -> tuple:
        """Apply add_bullet action."""
        section = action.section
        content = action.content

        if not section or not content:
            return -0.05, "add_bullet requires section and content"

        if section not in self._draft:
            self._draft[section] = []

        self._draft[section].append(content)
        self._last_action_result = f"Added bullet to {section}"

        return step_reward('add_bullet', action.model_dump(),
                          self._classified, self._draft, self._task['gold'])

    def _apply_remove_bullet(self, action: ChangelogAction) -> tuple:
        """Apply remove_bullet action."""
        section = action.section
        bullet_index = action.bullet_index

        if not section or bullet_index is None:
            return -0.05, "remove_bullet requires section and bullet_index"

        if section not in self._draft:
            return -0.05, f"Section {section} does not exist"

        if bullet_index < 0 or bullet_index >= len(self._draft[section]):
            return -0.05, f"Invalid bullet index: {bullet_index}"

        removed = self._draft[section].pop(bullet_index)
        self._last_action_result = f"Removed bullet from {section}"

        return step_reward('remove_bullet', action.model_dump(),
                          self._classified, self._draft, self._task['gold'])

    def _apply_set_version(self, action: ChangelogAction) -> tuple:
        """Apply set_version action."""
        version_bump = action.version_bump

        if not version_bump:
            return -0.05, "set_version requires version_bump"

        if version_bump not in ['patch', 'minor', 'major']:
            return -0.05, f"Invalid version_bump: {version_bump}"

        self._version_bump = version_bump
        self._last_action_result = f"Set version bump to {version_bump}"

        return step_reward('set_version', action.model_dump(),
                          self._classified, self._draft, self._task['gold'])

    def _apply_reorder_sections(self, action: ChangelogAction) -> tuple:
        """Apply reorder_sections action."""
        content = action.content

        if not content:
            return -0.01, "reorder_sections requires content"

        self._last_action_result = f"Sections reordered: {content}"
        return step_reward('reorder_sections', action.model_dump(),
                          self._classified, self._draft, self._task['gold'])

    def _apply_submit(self) -> tuple:
        """Apply submit action - triggers terminal reward."""
        terminal_r, grading = terminal_reward(
            self._draft, self._classified, self._version_bump,
            self._state.task_id, self._task['gold'], self._state.step_count
        )

        self._last_action_result = (
            f"Submitted! Score: {self._score + terminal_r:.2f}. "
            f"Feedback: {'; '.join(grading.feedback)}"
        )

        return terminal_r, self._last_action_result

    def _build_observation(self, done: bool, reward: Optional[float],
                           result: str) -> ChangelogObservation:
        """
        Build an observation from current state.

        Args:
            done: Whether episode is done
            reward: Step reward (None for initial observation)
            result: Result message from last action

        Returns:
            ChangelogObservation for the agent
        """
        return ChangelogObservation(
            task_id=self._state.task_id,
            commits=self._task['commits'],
            draft=self._draft.copy(),
            classified=self._classified.copy(),
            version_bump=self._version_bump,
            last_action_result=result,
            score_so_far=self._score,
            done=done,
            reward=reward,
        )
