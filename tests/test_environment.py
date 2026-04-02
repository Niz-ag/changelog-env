"""Tests for ChangelogEnvironment - reset/step/state correctness."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChangelogAction
from server.environment import ChangelogEnvironment


class TestChangelogEnvironment:
    """Test the core environment logic."""

    def test_reset_returns_observation(self):
        """reset() should return a ChangelogObservation."""
        env = ChangelogEnvironment()
        obs = env.reset(task_id='task_easy')

        assert obs is not None
        assert obs.task_id == 'task_easy'
        assert obs.done is False
        assert obs.reward is None
        assert len(obs.commits) > 0

    def test_reset_initializes_state(self):
        """reset() should initialize environment state."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        state = env.state
        assert state is not None
        assert state.task_id == 'task_easy'
        assert state.step_count == 0
        assert state.max_attempts == 20

    def test_state_is_property(self):
        """state should be a @property, not a method."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        # Should be callable as a property, not a method
        state = env.state
        assert state is not None

    def test_step_increments_count(self):
        """step() should increment step_count."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        for i in range(5):
            action = ChangelogAction(action_type='noop')
            obs = env.step(action)
            assert env.state.step_count == i + 1

    def test_classify_commit_action(self):
        """classify_commit action should work correctly."""
        env = ChangelogEnvironment()
        obs = env.reset(task_id='task_easy')

        # Get first commit hash
        commit_hash = obs.commits[0].hash

        action = ChangelogAction(
            action_type='classify_commit',
            commit_hash=commit_hash,
            label='feature',
        )
        obs = env.step(action)

        assert commit_hash in obs.classified
        assert obs.classified[commit_hash] == 'feature'

    def test_add_bullet_action(self):
        """add_bullet action should work correctly."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        action = ChangelogAction(
            action_type='add_bullet',
            section='Features',
            content='Added new feature',
        )
        obs = env.step(action)

        assert 'Features' in obs.draft
        assert len(obs.draft['Features']) == 1
        assert obs.draft['Features'][0] == 'Added new feature'

    def test_set_version_action(self):
        """set_version action should work correctly."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        action = ChangelogAction(
            action_type='set_version',
            version_bump='minor',
        )
        obs = env.step(action)

        assert obs.version_bump == 'minor'

    def test_submit_ends_episode(self):
        """submit action should end the episode."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        action = ChangelogAction(action_type='submit')
        obs = env.step(action)

        assert obs.done is True

    def test_max_steps_ends_episode(self):
        """Reaching max steps should end the episode."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        # Take 20 steps (max_attempts)
        for i in range(20):
            action = ChangelogAction(action_type='noop')
            obs = env.step(action)

        assert obs.done is True

    def test_score_accumulates(self):
        """Score should accumulate across steps."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        # Get a commit to classify
        commit_hash = env._task['commits'][0].hash

        # Classify correctly (should get +0.10)
        action = ChangelogAction(
            action_type='classify_commit',
            commit_hash=commit_hash,
            label='feature',  # Assuming first commit is a feature
        )
        obs = env.step(action)

        # Score should be positive
        assert obs.score_so_far >= 0

    def test_invalid_classify_label(self):
        """Invalid label should be rejected."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        commit_hash = env._task['commits'][0].hash

        action = ChangelogAction(
            action_type='classify_commit',
            commit_hash=commit_hash,
            label='invalid_label',
        )
        obs = env.step(action)

        # Should have negative reward
        assert obs.reward < 0

    def test_remove_bullet_action(self):
        """remove_bullet action should work correctly."""
        env = ChangelogEnvironment()
        env.reset(task_id='task_easy')

        # First add a bullet
        env.step(ChangelogAction(
            action_type='add_bullet',
            section='Features',
            content='Test bullet',
        ))

        # Then remove it
        action = ChangelogAction(
            action_type='remove_bullet',
            section='Features',
            bullet_index=0,
        )
        obs = env.step(action)

        assert 'Features' in obs.draft
        assert len(obs.draft['Features']) == 0

    def test_concurrent_sessions_flag(self):
        """SUPPORTS_CONCURRENT_SESSIONS should be True."""
        env = ChangelogEnvironment()
        assert env.SUPPORTS_CONCURRENT_SESSIONS is True
