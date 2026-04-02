"""Tests for Pydantic models - validation and serialization."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CommitRecord,
    ChangelogObservation,
    ChangelogAction,
    ChangelogState,
)


class TestCommitRecord:
    """Test CommitRecord model."""

    def test_valid_commit(self):
        commit = CommitRecord(
            hash='a1b2c3d',
            message='feat: add new feature',
            author='test@example.com',
            timestamp='2026-03-01T10:00:00Z',
            files_changed=['src/feature.py'],
            diff_summary='+50 lines',
        )
        assert commit.hash == 'a1b2c3d'
        assert commit.message == 'feat: add new feature'

    def test_commit_serialization(self):
        commit = CommitRecord(
            hash='abc1234',
            message='fix: bug fix',
            author='dev@example.com',
            timestamp='2026-03-01T10:00:00Z',
            files_changed=['src/a.py', 'src/b.py'],
            diff_summary='+10 -5',
        )
        data = commit.model_dump()
        assert data['hash'] == 'abc1234'
        assert len(data['files_changed']) == 2


class TestChangelogAction:
    """Test ChangelogAction model."""

    def test_classify_commit_action(self):
        action = ChangelogAction(
            action_type='classify_commit',
            commit_hash='a1b2c3d',
            label='feature',
        )
        assert action.action_type == 'classify_commit'
        assert action.commit_hash == 'a1b2c3d'
        assert action.label == 'feature'

    def test_add_bullet_action(self):
        action = ChangelogAction(
            action_type='add_bullet',
            section='Features',
            content='Added new authentication system',
        )
        assert action.action_type == 'add_bullet'
        assert action.section == 'Features'

    def test_set_version_action(self):
        action = ChangelogAction(
            action_type='set_version',
            version_bump='minor',
        )
        assert action.version_bump == 'minor'

    def test_submit_action(self):
        action = ChangelogAction(action_type='submit')
        assert action.action_type == 'submit'

    def test_invalid_action_type(self):
        # Should still create - validation happens elsewhere
        action = ChangelogAction(action_type='invalid_type')
        assert action.action_type == 'invalid_type'

    def test_action_serialization(self):
        action = ChangelogAction(
            action_type='classify_commit',
            commit_hash='abc123',
            label='bugfix',
        )
        data = action.model_dump(exclude_none=True)
        assert 'action_type' in data
        assert 'commit_hash' in data
        assert 'label' in data
        # None fields should be excluded
        assert 'section' not in data


class TestChangelogObservation:
    """Test ChangelogObservation model."""

    def test_initial_observation(self):
        obs = ChangelogObservation(
            task_id='task_easy',
            commits=[],
            done=False,
            reward=None,
        )
        assert obs.task_id == 'task_easy'
        assert obs.done is False
        assert obs.draft == {}
        assert obs.classified == {}

    def test_observation_with_commits(self):
        commit = CommitRecord(
            hash='a1b2c3d',
            message='feat: test',
            author='test@example.com',
            timestamp='2026-03-01T10:00:00Z',
            files_changed=['src/a.py'],
            diff_summary='+10',
        )
        obs = ChangelogObservation(
            task_id='task_medium',
            commits=[commit],
            done=False,
            reward=None,
        )
        assert len(obs.commits) == 1
        assert obs.commits[0].hash == 'a1b2c3d'

    def test_observation_with_draft(self):
        obs = ChangelogObservation(
            task_id='task_easy',
            commits=[],
            draft={'Features': ['Added feature A', 'Added feature B']},
            done=False,
            reward=None,
        )
        assert 'Features' in obs.draft
        assert len(obs.draft['Features']) == 2


class TestChangelogState:
    """Test ChangelogState model."""

    def test_initial_state(self):
        state = ChangelogState(
            episode_id='test-episode-123',
            step_count=0,
            task_id='task_easy',
            max_attempts=20,
            started_at='2026-03-01T10:00:00Z',
        )
        assert state.episode_id == 'test-episode-123'
        assert state.step_count == 0
        assert state.max_attempts == 20

    def test_state_serialization(self):
        state = ChangelogState(
            episode_id='ep-456',
            step_count=5,
            task_id='task_hard',
            max_attempts=20,
            started_at='2026-03-01T10:00:00Z',
        )
        data = state.model_dump()
        assert data['episode_id'] == 'ep-456'
        assert data['step_count'] == 5
