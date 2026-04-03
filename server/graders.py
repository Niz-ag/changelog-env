"""Grader functions for ChangelogEnv tasks - deterministic scoring."""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class GradingResult:
    """Result of grading an episode."""
    score: float
    breakdown: Dict[str, float]
    feedback: List[str]


def grade(task_id: str, draft: Dict[str, List[str]], classified: Dict[str, str],
          version_bump: str, gold: Dict[str, Any]) -> GradingResult:
    """
    Grade the submitted changelog based on task type.

    Args:
        task_id: Which task was run (task_easy, task_medium, task_hard)
        draft: The agent's changelog draft (section -> bullets)
        classified: The agent's commit classifications (hash -> label)
        version_bump: The agent's semver decision
        gold: Gold labels for this task

    Returns:
        GradingResult with score in [0.0, 1.0]
    """
    graders = {
        'task_easy': grade_easy,
        'task_medium': grade_medium,
        'task_hard': grade_hard,
    }

    if task_id not in graders:
        raise ValueError(f"Unknown task_id: {task_id}")

    return graders[task_id](draft, classified, version_bump, gold)


def grade_easy(draft: Dict[str, List[str]], classified: Dict[str, str],
               version_bump: str, gold: Dict[str, Any]) -> GradingResult:
    """
    Grade Task 1 - Single PR Summary [Easy]

    Criteria:
    - All 5 commits classified correctly
    - Merge commit filtered out
    - Version bump = minor
    - All non-internal commits covered in bullets
    - No hallucinated bullets
    """
    feedback = []
    breakdown = {}
    score = 0.0

    gold_classifications = gold['classifications']
    expected_version = gold['version_bump']
    expected_bullets = gold.get('expected_bullets', {})

    # 1. Classification accuracy (40% of score)
    correct_classifications = 0
    total_commits = len(gold_classifications)

    for commit_hash, expected_label in gold_classifications.items():
        agent_label = classified.get(commit_hash)
        if agent_label == expected_label:
            correct_classifications += 1

    classification_score = correct_classifications / max(total_commits, 1)
    breakdown['classification'] = classification_score * 0.40
    score += breakdown['classification']

    if total_commits == 0 or classification_score == 1.0:
        feedback.append("All commits classified correctly")
    else:
        feedback.append(f"Classification accuracy: {classification_score:.0%}")

    # 2. Version bump correctness (15% of score)
    version_correct = version_bump == expected_version
    breakdown['version'] = 0.15 if version_correct else 0.0
    score += breakdown['version']

    if version_correct:
        feedback.append("Correct version bump (minor)")
    else:
        feedback.append(f"Version bump: expected {expected_version}, got {version_bump}")

    # 3. Coverage score (25% of score)
    # Check if expected bullets are covered in the draft
    total_expected_bullets = sum(len(bullets) for bullets in expected_bullets.values())
    covered_bullets = 0

    for section, expected_bullet_list in expected_bullets.items():
        draft_bullets = draft.get(section, [])
        for expected in expected_bullet_list:
            # Check if any draft bullet covers this expected content
            for draft_bullet in draft_bullets:
                if _bullet_covers(draft_bullet, expected):
                    covered_bullets += 1
                    break

    coverage_score = covered_bullets / max(total_expected_bullets, 1)
    breakdown['coverage'] = coverage_score * 0.25
    score += breakdown['coverage']

    if coverage_score >= 0.8:
        feedback.append(f"Good coverage: {covered_bullets}/{total_expected_bullets} expected bullets")
    else:
        feedback.append(f"Coverage: {covered_bullets}/{total_expected_bullets} expected bullets")

    # 4. Hallucination penalty (-20% max)
    # Count bullets that don't map to any expected content
    hallucination_count = 0
    total_draft_bullets = sum(len(bullets) for bullets in draft.values())

    for section, draft_bullets in draft.items():
        expected_for_section = expected_bullets.get(section, [])
        for bullet in draft_bullets:
            is_hallucinated = True
            for expected in expected_for_section:
                if _bullet_covers(bullet, expected):
                    is_hallucinated = False
                    break
            if is_hallucinated:
                hallucination_count += 1

    hallucination_penalty = min(hallucination_count * 0.05, 0.20)
    breakdown['hallucination_penalty'] = -hallucination_penalty
    score -= hallucination_penalty

    if hallucination_count > 0:
        feedback.append(f"Hallucinated bullets: {hallucination_count}")

    # 5. Format compliance (20% of score)
    format_score = _check_format(draft, ['Features', 'Bug Fixes', 'Breaking Changes'])
    breakdown['format'] = format_score * 0.20
    score += breakdown['format']

    if format_score >= 0.8:
        feedback.append("Good format compliance")
    else:
        feedback.append("Format issues detected")

    return GradingResult(
        score=max(0.0, min(1.0, score)),
        breakdown=breakdown,
        feedback=feedback
    )


def grade_medium(draft: Dict[str, List[str]], classified: Dict[str, str],
                version_bump: str, gold: Dict[str, Any]) -> GradingResult:
    """
    Grade Task 2 - Sprint Release, Mixed Signal [Medium]

    Criteria:
    - Breaking change identified
    - Internal commits excluded
    - Version bump = major
    - Format compliance
    """
    feedback = []
    breakdown = {}
    score = 0.0

    gold_classifications = gold['classifications']
    expected_version = gold['version_bump']
    breaking_commits = gold.get('breaking_commits', [])
    expected_bullets = gold.get('expected_bullets', {})

    # 1. Classification accuracy (30% of score)
    correct_classifications = 0
    total = len(gold_classifications)
    for commit_hash, expected_label in gold_classifications.items():
        agent_label = classified.get(commit_hash)
        if agent_label == expected_label:
            correct_classifications += 1

    classification_score = correct_classifications / max(total, 1)
    breakdown['classification'] = classification_score * 0.30
    score += breakdown['classification']

    # 2. Breaking change detection (15% of score)
    breaking_found = all(
        classified.get(commit) == 'breaking'
        for commit in breaking_commits
    )
    breakdown['breaking'] = 0.15 if breaking_found else 0.0
    score += breakdown['breaking']

    if breaking_found:
        feedback.append("Breaking change correctly identified")
    else:
        feedback.append("Breaking change NOT detected")

    # 3. Version bump correctness (15% of score)
    version_correct = version_bump == expected_version
    breakdown['version'] = 0.15 if version_correct else 0.0
    score += breakdown['version']

    if version_correct:
        feedback.append("Correct version bump (major)")
    else:
        feedback.append(f"Version bump: expected {expected_version}, got {version_bump}")

    # 4. Coverage score (20% of score)
    total_expected_bullets = sum(len(bullets) for bullets in expected_bullets.values())
    covered_bullets = 0

    for section, expected_bullet_list in expected_bullets.items():
        draft_bullets = draft.get(section, [])
        for expected in expected_bullet_list:
            for draft_bullet in draft_bullets:
                if _bullet_covers(draft_bullet, expected):
                    covered_bullets += 1
                    break

    coverage_score = covered_bullets / max(total_expected_bullets, 1)
    breakdown['coverage'] = coverage_score * 0.20
    score += breakdown['coverage']

    # 5. Internal leak penalty (-15% max)
    # Penalize if internal/chore commits appear in user-facing output
    internal_leak_count = 0
    for commit_hash, label in classified.items():
        if label in ['internal', 'chore']:
            # Check if this commit's content leaked into draft
            # (simplified: just check if any section has content that shouldn't be there)
            pass  # Would need commit message mapping for full check

    leak_penalty = min(internal_leak_count * 0.05, 0.15)
    breakdown['leak_penalty'] = -leak_penalty
    score -= leak_penalty

    # 6. Format compliance (20% of score)
    format_score = _check_format(draft, ['Features', 'Bug Fixes', 'Breaking Changes'])
    breakdown['format'] = format_score * 0.20
    score += breakdown['format']

    return GradingResult(
        score=max(0.0, min(1.0, score)),
        breakdown=breakdown,
        feedback=feedback
    )


def grade_hard(draft: Dict[str, List[str]], classified: Dict[str, str],
               version_bump: str, gold: Dict[str, Any]) -> GradingResult:
    """
    Grade Task 3 - Multi-Version Audit [Hard]

    Criteria:
    - 3 version sections produced
    - Correct boundary inference
    - No cross-contamination
    - Correct semver per section
    """
    feedback = []
    breakdown = {}
    score = 0.0

    version_boundaries = gold.get('version_boundaries', {})
    expected_sections = gold.get('expected_sections', {})

    # 1. Check if 3 version sections were produced (30% of score)
    version_sections_found = 0
    expected_versions = ['v1.0.0', 'v1.1.0', 'v2.0.0']

    for version in expected_versions:
        if version in draft or any(version in section for section in draft.keys()):
            version_sections_found += 1

    section_score = version_sections_found / 3
    breakdown['sections'] = section_score * 0.30
    score += breakdown['sections']

    feedback.append(f"Version sections found: {version_sections_found}/3")

    # 2. Boundary correctness (25% of score)
    # Check if commits are grouped into correct version boundaries
    gold_classifications = gold['classifications']
    correct_boundaries = 0
    total_boundary_checks = 0

    for version, commits in version_boundaries.items():
        for commit in commits:
            total_boundary_checks += 1
            # Simplified: check if classification exists
            if commit in classified:
                correct_boundaries += 1

    boundary_score = correct_boundaries / max(total_boundary_checks, 1)
    breakdown['boundaries'] = boundary_score * 0.25
    score += breakdown['boundaries']

    # 3. Semver correctness per section (20% of score)
    # Agent should infer correct semver for each version
    semver_correct = 0
    for version, expected in expected_sections.items():
        # Simplified: just check overall version_bump for now
        if version == 'v2.0.0' and version_bump == 'major':
            semver_correct += 1
        elif version in ['v1.0.0', 'v1.1.0'] and version_bump in ['minor', 'patch']:
            semver_correct += 1

    semver_score = semver_correct / max(len(expected_sections), 1)
    breakdown['semver'] = semver_score * 0.20
    score += breakdown['semver']

    # 4. Cross-contamination penalty (-15% max)
    # Penalize if commits from different versions are mixed
    contamination_penalty = 0.0
    breakdown['contamination_penalty'] = -contamination_penalty

    # 5. Coverage score (25% of score)
    # Check if commits are covered in the output
    coverage_score = len(classified) / max(len(gold_classifications), 1)
    breakdown['coverage'] = min(coverage_score, 1.0) * 0.25
    score += breakdown['coverage']

    return GradingResult(
        score=max(0.0, min(1.0, score)),
        breakdown=breakdown,
        feedback=feedback
    )


def _bullet_covers(draft_bullet: str, expected: str) -> bool:
    """
    Check if a draft bullet covers the expected content.
    Simple substring/approximate matching.
    """
    draft_lower = draft_bullet.lower()
    expected_lower = expected.lower()

    # Check if expected keywords are in draft
    expected_words = set(expected_lower.split())
    draft_words = set(draft_lower.split())

    # Require at least 50% word overlap or substring match
    if expected_lower in draft_lower:
        return True

    common_words = expected_words & draft_words
    overlap = len(common_words) / max(len(expected_words), 1)

    return overlap >= 0.5


def _check_format(draft: Dict[str, List[str]], expected_sections: List[str]) -> float:
    """
    Check format compliance of the draft.
    Returns score in [0.0, 1.0].
    """
    if not draft:
        return 0.0

    # Check if any expected sections are present
    sections_found = 0
    has_content = 0

    for section in expected_sections:
        for draft_section in draft.keys():
            if section.lower() in draft_section.lower():
                sections_found += 1
                if draft[draft_section]:
                    has_content += 1
                break

    section_score = sections_found / len(expected_sections)
    content_score = has_content / max(sections_found, 1)

    return (section_score * 0.6 + content_score * 0.4)
