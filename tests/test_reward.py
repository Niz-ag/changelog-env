"""Tests for reward functions - step and terminal reward correctness."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChangelogAction
from server.reward import (
    step_reward,
    terminal_reward,
    compute_reward_breakdown,
)


class TestStepReward:
    """Test step_reward function."""

    def test_correct_classification_reward(self):
        """Correct classification should give +0.10."""
        gold = {
            'classifications': {'abc123': 'feature'},
        }

        action = {
            'action_type': 'classify_commit',
            'commit_hash': 'abc123',
            'label': 'feature',
        }

        reward, msg = step_reward('classify_commit', action, {}, {}, gold)
        assert reward == +0.10
        assert 'Correct' in msg

    def test_wrong_classification_reward(self):
        """Wrong classification should give -0.05."""
        gold = {
            'classifications': {'abc123': 'feature'},
        }

        action = {
            'action_type': 'classify_commit',
            'commit_hash': 'abc123',
            'label': 'bugfix',  # Wrong!
        }

        reward, msg = step_reward('classify_commit', action, {}, {}, gold)
        assert reward == -0.05

    def test_correct_version_bump_reward(self):
        """Correct version bump should give +0.10."""
        gold = {'version_bump': 'minor'}

        action = {
            'action_type': 'set_version',
            'version_bump': 'minor',
        }

        reward, msg = step_reward('set_version', action, {}, {}, gold)
        assert reward == +0.10

    def test_wrong_version_bump_reward(self):
        """Wrong version bump should give -0.10."""
        gold = {'version_bump': 'minor'}

        action = {
            'action_type': 'set_version',
            'version_bump': 'major',  # Wrong!
        }

        reward, msg = step_reward('set_version', action, {}, {}, gold)
        assert reward == -0.10

    def test_add_bullet_reward(self):
        """Well-formed bullet should give +0.01."""
        gold = {}

        action = {
            'action_type': 'add_bullet',
            'section': 'Features',
            'content': 'Added new authentication system with JWT',
        }

        reward, msg = step_reward('add_bullet', action, {}, {}, gold)
        assert reward == +0.01

    def test_short_bullet_penalty(self):
        """Too short bullet should be penalized."""
        gold = {}

        action = {
            'action_type': 'add_bullet',
            'section': 'Features',
            'content': 'x',  # Too short
        }

        reward, msg = step_reward('add_bullet', action, {}, {}, gold)
        assert reward < 0

    def test_noop_penalty(self):
        """noop action should be penalized -0.01."""
        reward, msg = step_reward('noop', {}, {}, {}, {})
        assert reward == -0.01

    def test_unknown_action_penalty(self):
        """Unknown action type should be heavily penalized."""
        reward, msg = step_reward('unknown_action', {}, {}, {}, {})
        assert reward == -0.10

    def test_missing_classify_fields(self):
        """classify_commit without required fields should fail."""
        action = {
            'action_type': 'classify_commit',
            # Missing commit_hash and label
        }

        reward, msg = step_reward('classify_commit', action, {}, {}, {})
        assert reward == -0.05

    def test_missing_bullet_fields(self):
        """add_bullet without required fields should fail."""
        action = {
            'action_type': 'add_bullet',
            # Missing section and content
        }

        reward, msg = step_reward('add_bullet', action, {}, {}, {})
        assert reward == -0.05


class TestTerminalReward:
    """Test terminal_reward function."""

    def test_terminal_reward_computes_grade(self):
        """terminal_reward should compute and return grade."""
        gold = {
            'classifications': {'abc': 'feature'},
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        draft = {}
        classified = {'abc': 'feature'}

        reward, grading = terminal_reward(
            draft, classified, 'minor', 'task_easy', gold, steps_taken=10
        )

        assert reward >= 0
        assert grading is not None
        assert hasattr(grading, 'score')
        assert hasattr(grading, 'breakdown')
        assert hasattr(grading, 'feedback')

    def test_early_submit_penalty(self):
        """Submitting before step 5 should incur penalty."""
        gold = {
            'classifications': {},
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        reward, grading = terminal_reward(
            {}, {}, 'minor', 'task_easy', gold, steps_taken=3
        )

        # Should have early submit penalty mentioned
        assert any('Early submit' in f for f in grading.feedback)

    def test_normal_submit_no_penalty(self):
        """Submitting after step 5 should not incur penalty."""
        gold = {
            'classifications': {},
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        reward, grading = terminal_reward(
            {}, {}, 'minor', 'task_easy', gold, steps_taken=6
        )

        # Should NOT have early submit penalty
        assert not any('Early submit' in f for f in grading.feedback)


class TestComputeRewardBreakdown:
    """Test compute_reward_breakdown function."""

    def test_breakdown_contains_components(self):
        """Breakdown should contain individual score components."""
        gold = {
            'classifications': {'abc': 'feature'},
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        breakdown = compute_reward_breakdown(
            'task_easy', {}, {'abc': 'feature'}, 'minor', gold
        )

        assert 'total' in breakdown
        assert len(breakdown) >= 2  # total + at least one component
