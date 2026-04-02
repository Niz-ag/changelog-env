"""Tests for grader functions - per-task grading correctness."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.graders import grade, grade_easy, grade_medium, grade_hard, GradingResult


class TestGradeEasy:
    """Test grading for task_easy."""

    def test_perfect_submission(self):
        """Perfect classification and coverage should score 1.0."""
        gold = {
            'classifications': {
                'a1b2c3d': 'feature',
                'e4f5g6h': 'feature',
                'i7j8k9l': 'bugfix',
                'm0n1o2p': 'docs',
                'q3r4s5t': 'chore',
            },
            'version_bump': 'minor',
            'expected_bullets': {
                'Features': [
                    'Added user authentication with JWT tokens',
                    'Added password reset email functionality',
                ],
                'Bug Fixes': [
                    'Fixed session timeout issue on mobile devices',
                ],
            },
        }

        draft = {
            'Features': [
                'Added user authentication with JWT tokens',
                'Added password reset email functionality',
            ],
            'Bug Fixes': [
                'Fixed session timeout issue on mobile devices',
            ],
        }

        classified = gold['classifications'].copy()
        version_bump = 'minor'

        result = grade_easy(draft, classified, version_bump, gold)

        # Should score very high (may not be exactly 1.0 due to format scoring)
        assert result.score > 0.8

    def test_wrong_version_bump(self):
        """Wrong version bump should reduce score."""
        gold = {
            'classifications': {'a1': 'feature'},
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        result = grade_easy({}, {'a1': 'feature'}, 'major', gold)

        # Should lose the 15% version points
        assert result.breakdown.get('version', 0) == 0.0

    def test_wrong_classification(self):
        """Wrong classifications should reduce score."""
        gold = {
            'classifications': {
                'a1': 'feature',
                'a2': 'bugfix',
            },
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        # Classify both as feature (one wrong)
        classified = {'a1': 'feature', 'a2': 'feature'}

        result = grade_easy({}, classified, 'minor', gold)

        # Should lose some classification points
        assert result.breakdown.get('classification', 0) < 0.40

    def test_hallucinated_bullets(self):
        """Hallucinated bullets should be penalized."""
        gold = {
            'classifications': {},
            'version_bump': 'minor',
            'expected_bullets': {
                'Features': ['Real feature'],
            },
        }

        draft = {
            'Features': [
                'Real feature',
                'Made up feature that does not exist',
                'Another hallucination',
            ],
        }

        result = grade_easy(draft, {}, 'minor', gold)

        # Should have hallucination penalty
        assert result.breakdown.get('hallucination_penalty', 0) < 0


class TestGradeMedium:
    """Test grading for task_medium."""

    def test_breaking_change_detection(self):
        """Breaking change must be detected for full score."""
        gold = {
            'classifications': {'a000006': 'breaking'},
            'version_bump': 'major',
            'breaking_commits': ['a000006'],
            'expected_bullets': {},
        }

        # Agent correctly identifies breaking change
        result = grade_medium(
            {}, {'a000006': 'breaking'}, 'major', gold
        )
        assert result.breakdown.get('breaking', 0) == 0.15

        # Agent misses breaking change
        result = grade_medium(
            {}, {'a000006': 'feature'}, 'major', gold
        )
        assert result.breakdown.get('breaking', 0) == 0.0

    def test_major_version_for_breaking(self):
        """Breaking changes require major version bump."""
        gold = {
            'classifications': {},
            'version_bump': 'major',
            'breaking_commits': [],
            'expected_bullets': {},
        }

        result = grade_medium({}, {}, 'major', gold)
        assert result.breakdown.get('version', 0) == 0.15

        result = grade_medium({}, {}, 'minor', gold)
        assert result.breakdown.get('version', 0) == 0.0


class TestGradeHard:
    """Test grading for task_hard."""

    def test_version_sections_detection(self):
        """Agent must produce 3 version sections."""
        gold = {
            'classifications': {},
            'version_boundaries': {
                'v1.0.0': ['c1', 'c15'],
                'v1.1.0': ['c16', 'c35'],
                'v2.0.0': ['c36', 'c52'],
            },
            'expected_sections': {
                'v1.0.0': {'version_bump': 'minor'},
                'v1.1.0': {'version_bump': 'minor'},
                'v2.0.0': {'version_bump': 'major'},
            },
        }

        # Agent produces all 3 versions
        draft = {
            'v1.0.0': ['Feature A'],
            'v1.1.0': ['Feature B'],
            'v2.0.0': ['Breaking change'],
        }

        result = grade_hard(draft, {}, 'major', gold)
        assert result.breakdown.get('sections', 0) == 0.30

        # Agent only produces 1 version
        draft = {'v1.0.0': ['Feature A']}
        result = grade_hard(draft, {}, 'major', gold)
        assert result.breakdown.get('sections', 0) < 0.30


class TestGradeFunction:
    """Test the main grade() dispatcher."""

    def test_grade_dispatches_to_correct_task(self):
        """grade() should dispatch to correct task grader."""
        gold = {
            'classifications': {},
            'version_bump': 'minor',
            'expected_bullets': {},
        }

        result_easy = grade('task_easy', {}, {}, 'minor', gold)
        result_medium = grade('task_medium', {}, {}, 'minor', gold)
        result_hard = grade('task_hard', {}, {}, 'major', gold)

        # All should return GradingResult
        assert isinstance(result_easy, GradingResult)
        assert isinstance(result_medium, GradingResult)
        assert isinstance(result_hard, GradingResult)

    def test_grade_unknown_task(self):
        """grade() should raise error for unknown task."""
        with pytest.raises(ValueError):
            grade('task_unknown', {}, {}, 'minor', {})
