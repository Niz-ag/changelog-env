"""Task definitions for ChangelogEnv - commit datasets for all 3 tasks."""

from typing import List, Dict, Any
from ..models import CommitRecord


def load_task(task_id: str, seed: int = None) -> Dict[str, Any]:
    """Load task configuration and commits by task_id."""
    tasks = {
        'task_easy': get_easy_task,
        'task_medium': get_medium_task,
        'task_hard': get_hard_task,
    }

    if task_id not in tasks:
        raise ValueError(f"Unknown task_id: {task_id}")

    return tasks[task_id](seed)


def get_easy_task(seed: int = None) -> Dict[str, Any]:
    """
    Task 1 - Single PR Summary [Easy]
    5 commits from a single merged PR. Clear messages.
    Goal: classify all commits, write 3-5 bullet summary, set correct semver bump.
    """
    commits = [
        CommitRecord(
            hash='a1b2c3d',
            message='feat: add user authentication with JWT tokens',
            author='alice@example.com',
            timestamp='2026-03-01T10:30:00Z',
            files_changed=['src/auth.py', 'src/models/user.py'],
            diff_summary='+142 lines in auth.py, +35 lines in user.py',
        ),
        CommitRecord(
            hash='e4f5g6h',
            message='feat: add password reset email functionality',
            author='bob@example.com',
            timestamp='2026-03-01T14:20:00Z',
            files_changed=['src/auth.py', 'src/email.py'],
            diff_summary='+89 lines in auth.py, +56 lines in email.py',
        ),
        CommitRecord(
            hash='i7j8k9l',
            message='fix: resolve session timeout issue on mobile devices',
            author='alice@example.com',
            timestamp='2026-03-02T09:15:00Z',
            files_changed=['src/auth.py', 'tests/test_auth.py'],
            diff_summary='+12 -5 lines in auth.py, +28 lines in test_auth.py',
        ),
        CommitRecord(
            hash='m0n1o2p',
            message='docs: update README with authentication examples',
            author='carol@example.com',
            timestamp='2026-03-02T11:00:00Z',
            files_changed=['README.md'],
            diff_summary='+45 lines in README.md',
        ),
        CommitRecord(
            hash='q3r4s5t',
            message='Merge branch feature/auth into main',
            author='alice@example.com',
            timestamp='2026-03-02T16:00:00Z',
            files_changed=['.gitmerge'],
            diff_summary='Merge commit - no actual changes',
        ),
    ]

    # Gold labels for grading
    gold = {
        'classifications': {
            'a1b2c3d': 'feature',
            'e4f5g6h': 'feature',
            'i7j8k9l': 'bugfix',
            'm0n1o2p': 'docs',
            'q3r4s5t': 'chore',  # merge commit should be filtered
        },
        'version_bump': 'minor',  # 2 features, 1 bugfix
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

    return {
        'task_id': 'task_easy',
        'commits': commits,
        'gold': gold,
        'description': 'Single PR Summary - Classify 5 commits and generate changelog',
    }


def get_medium_task(seed: int = None) -> Dict[str, Any]:
    """
    Task 2 - Sprint Release, Mixed Signal [Medium]
    18 commits from a two-week sprint. Several have bad messages.
    One commit introduces a breaking API change visible only in the diff.
    """
    commits = [
        CommitRecord(
            hash='a000001',
            message='feat: add new dashboard widgets for analytics',
            author='dev1@example.com',
            timestamp='2026-03-01T09:00:00Z',
            files_changed=['src/dashboard.py', 'src/widgets/analytics.py'],
            diff_summary='+200 lines in dashboard.py, +150 lines in analytics.py',
        ),
        CommitRecord(
            hash='a000002',
            message='fix stuff',  # Bad message - need diff inspection
            author='dev2@example.com',
            timestamp='2026-03-01T10:30:00Z',
            files_changed=['src/api/users.py'],
            diff_summary='+5 -3 lines - fixed null pointer in user lookup',
        ),
        CommitRecord(
            hash='a000003',
            message='WIP',  # Bad message
            author='dev3@example.com',
            timestamp='2026-03-01T14:00:00Z',
            files_changed=['src/cache.py'],
            diff_summary='+80 lines - added Redis caching layer',
        ),
        CommitRecord(
            hash='a000004',
            message='feat: implement real-time notifications',
            author='dev1@example.com',
            timestamp='2026-03-02T09:00:00Z',
            files_changed=['src/notifications.py', 'src/websocket.py'],
            diff_summary='+300 lines in notifications.py, +120 lines in websocket.py',
        ),
        CommitRecord(
            hash='a000005',
            message='refactor: optimize database queries',  # Internal - not user facing
            author='dev2@example.com',
            timestamp='2026-03-02T11:00:00Z',
            files_changed=['src/db/queries.py'],
            diff_summary='+45 -120 lines - query optimization',
        ),
        CommitRecord(
            hash='a000006',
            message='BREAKING: remove deprecated /api/v1/users endpoint',
            author='dev3@example.com',
            timestamp='2026-03-03T09:00:00Z',
            files_changed=['src/api/routes.py', 'src/api/v1/__init__.py'],
            diff_summary='-250 lines removed - deprecated endpoint deleted',
        ),
        CommitRecord(
            hash='a000007',
            message='fix: correct calculation in billing module',
            author='dev1@example.com',
            timestamp='2026-03-03T14:00:00Z',
            files_changed=['src/billing.py'],
            diff_summary='+8 -8 lines in billing.py',
        ),
        CommitRecord(
            hash='a000008',
            message='update deps',  # Chore
            author='dev2@example.com',
            timestamp='2026-03-04T09:00:00Z',
            files_changed=['requirements.txt', 'package.json'],
            diff_summary='Dependency updates only',
        ),
        CommitRecord(
            hash='a000009',
            message='feat: add export to CSV functionality',
            author='dev3@example.com',
            timestamp='2026-03-04T11:00:00Z',
            files_changed=['src/export.py', 'src/reports.py'],
            diff_summary='+95 lines in export.py, +30 lines in reports.py',
        ),
        CommitRecord(
            hash='a000010',
            message='perf: improve search performance by 40%',  # Internal
            author='dev1@example.com',
            timestamp='2026-03-05T09:00:00Z',
            files_changed=['src/search.py'],
            diff_summary='+60 -45 lines - algorithm optimization',
        ),
        CommitRecord(
            hash='a000011',
            message='fix: handle edge case in date parsing',
            author='dev2@example.com',
            timestamp='2026-03-05T14:00:00Z',
            files_changed=['src/utils/dates.py'],
            diff_summary='+15 -3 lines in dates.py',
        ),
        CommitRecord(
            hash='a000012',
            message='docs: add API documentation for v2 endpoints',
            author='dev3@example.com',
            timestamp='2026-03-06T09:00:00Z',
            files_changed=['docs/api_v2.md'],
            diff_summary='+500 lines in api_v2.md',
        ),
        CommitRecord(
            hash='a000013',
            message='ci: update GitHub Actions workflow',  # Chore
            author='dev1@example.com',
            timestamp='2026-03-06T11:00:00Z',
            files_changed=['.github/workflows/ci.yml'],
            diff_summary='CI config changes only',
        ),
        CommitRecord(
            hash='a000014',
            message='feat: add dark mode theme support',
            author='dev2@example.com',
            timestamp='2026-03-07T09:00:00Z',
            files_changed=['src/themes.py', 'static/css/dark.css'],
            diff_summary='+80 lines in themes.py, +200 lines in dark.css',
        ),
        CommitRecord(
            hash='a000015',
            message='refactor: clean up legacy code',  # Internal - looks like feature but isn't
            author='dev3@example.com',
            timestamp='2026-03-07T14:00:00Z',
            files_changed=['src/legacy.py', 'src/utils.py'],
            diff_summary='+20 -150 lines - code cleanup',
        ),
        CommitRecord(
            hash='a000016',
            message='fix: resolve memory leak in background worker',
            author='dev1@example.com',
            timestamp='2026-03-08T09:00:00Z',
            files_changed=['src/workers.py'],
            diff_summary='+25 -10 lines in workers.py',
        ),
        CommitRecord(
            hash='a000017',
            message='test: add integration tests for auth flow',  # Internal
            author='dev2@example.com',
            timestamp='2026-03-08T14:00:00Z',
            files_changed=['tests/test_auth_integration.py'],
            diff_summary='+300 lines in test file',
        ),
        CommitRecord(
            hash='a000018',
            message='Merge pull request #42 from feature/sprint-update',
            author='dev3@example.com',
            timestamp='2026-03-08T16:00:00Z',
            files_changed=['.gitmerge'],
            diff_summary='Merge commit',
        ),
    ]

    gold = {
        'classifications': {
            'a000001': 'feature',
            'a000002': 'bugfix',  # Need to read diff to know
            'a000003': 'internal',  # WIP but diff shows internal caching
            'a000004': 'feature',
            'a000005': 'internal',
            'a000006': 'breaking',  # Breaking API change
            'a000007': 'bugfix',
            'a000008': 'chore',
            'a000009': 'feature',
            'a000010': 'internal',
            'a000011': 'bugfix',
            'a000012': 'docs',
            'a000013': 'chore',
            'a000014': 'feature',
            'a000015': 'internal',  # Looks like feature but is refactor
            'a000016': 'bugfix',
            'a000017': 'internal',
            'a000018': 'chore',  # Merge commit
        },
        'version_bump': 'major',  # Breaking change present
        'breaking_commits': ['a000006'],
        'expected_bullets': {
            'Breaking Changes': [
                'Removed deprecated /api/v1/users endpoint',
            ],
            'Features': [
                'Added new dashboard widgets for analytics',
                'Added real-time notifications',
                'Added export to CSV functionality',
                'Added dark mode theme support',
            ],
            'Bug Fixes': [
                'Fixed null pointer in user lookup',
                'Corrected calculation in billing module',
                'Fixed edge case in date parsing',
                'Resolved memory leak in background worker',
            ],
        },
    }

    return {
        'task_id': 'task_medium',
        'commits': commits,
        'gold': gold,
        'description': 'Sprint Release - Mixed signal commits with breaking change',
    }


def get_hard_task(seed: int = None) -> Dict[str, Any]:
    """
    Task 3 - Multi-Version Audit [Hard]
    52 commits with no version tags. Agent must infer 3 version boundaries.
    """
    # Generating 52 commits programmatically for brevity
    commits = []

    # Version 1.0.0 - Initial release (commits 1-15)
    v1_commits = [
        ('init001', 'feat: initial project setup', 'dev1@example.com', '2026-01-01', ['src/__init__.py'], '+50 lines', 'feature'),
        ('init002', 'feat: add core data models', 'dev2@example.com', '2026-01-02', ['src/models.py'], '+200 lines', 'feature'),
        ('init003', 'feat: implement basic CRUD operations', 'dev1@example.com', '2026-01-03', ['src/repository.py'], '+150 lines', 'feature'),
        ('init004', 'feat: add REST API endpoints', 'dev3@example.com', '2026-01-04', ['src/api.py'], '+180 lines', 'feature'),
        ('init005', 'fix: correct validation in user model', 'dev2@example.com', '2026-01-05', ['src/models.py'], '+5 -3 lines', 'bugfix'),
        ('init006', 'docs: add getting started guide', 'dev1@example.com', '2026-01-06', ['docs/getting_started.md'], '+300 lines', 'docs'),
        ('init007', 'feat: add database migrations', 'dev2@example.com', '2026-01-07', ['src/migrations.py'], '+120 lines', 'feature'),
        ('init008', 'refactor: reorganize project structure', 'dev3@example.com', '2026-01-08', ['src/'], 'reorganized', 'internal'),
        ('init009', 'feat: implement user authentication', 'dev1@example.com', '2026-01-09', ['src/auth.py'], '+250 lines', 'feature'),
        ('init010', 'fix: resolve connection pool exhaustion', 'dev2@example.com', '2026-01-10', ['src/db.py'], '+20 -10 lines', 'bugfix'),
        ('init011', 'ci: setup CI/CD pipeline', 'dev3@example.com', '2026-01-11', ['.github/workflows/ci.yml'], '+80 lines', 'chore'),
        ('init012', 'feat: add logging infrastructure', 'dev1@example.com', '2026-01-12', ['src/logging.py'], '+100 lines', 'internal'),
        ('init013', 'docs: document API endpoints', 'dev2@example.com', '2026-01-13', ['docs/api.md'], '+400 lines', 'docs'),
        ('init014', 'perf: optimize database queries', 'dev3@example.com', '2026-01-14', ['src/queries.py'], '+30 -50 lines', 'internal'),
        ('init015', 'Release v1.0.0', 'dev1@example.com', '2026-01-15', ['CHANGELOG.md'], '+50 lines', 'chore'),
    ]

    # Version 1.1.0 - Feature updates (commits 16-35)
    v1_1_commits = [
        ('v11_001', 'feat: add search functionality', 'dev1@example.com', '2026-01-16', ['src/search.py'], '+180 lines', 'feature'),
        ('v11_002', 'feat: implement full-text search', 'dev2@example.com', '2026-01-17', ['src/search.py'], '+100 lines', 'feature'),
        ('v11_003', 'fix: search results pagination bug', 'dev3@example.com', '2026-01-18', ['src/search.py'], '+15 -8 lines', 'bugfix'),
        ('v11_004', 'feat: add filtering options', 'dev1@example.com', '2026-01-19', ['src/filters.py'], '+120 lines', 'feature'),
        ('v11_005', 'docs: update search documentation', 'dev2@example.com', '2026-01-20', ['docs/search.md'], '+80 lines', 'docs'),
        ('v11_006', 'feat: add sorting capabilities', 'dev3@example.com', '2026-01-21', ['src/sorting.py'], '+90 lines', 'feature'),
        ('v11_007', 'refactor: extract common search logic', 'dev1@example.com', '2026-01-22', ['src/search/utils.py'], '+60 -30 lines', 'internal'),
        ('v11_008', 'fix: handle special characters in search', 'dev2@example.com', '2026-01-23', ['src/search.py'], '+25 -10 lines', 'bugfix'),
        ('v11_009', 'feat: add search suggestions/autocomplete', 'dev3@example.com', '2026-01-24', ['src/suggestions.py'], '+150 lines', 'feature'),
        ('v11_010', 'test: add search unit tests', 'dev1@example.com', '2026-01-25', ['tests/test_search.py'], '+200 lines', 'internal'),
        ('v11_011', 'feat: implement advanced filters', 'dev2@example.com', '2026-01-26', ['src/filters.py'], '+80 lines', 'feature'),
        ('v11_012', 'fix: memory leak in search worker', 'dev3@example.com', '2026-01-27', ['src/workers.py'], '+10 -5 lines', 'bugfix'),
        ('v11_013', 'ci: add performance benchmarks', 'dev1@example.com', '2026-01-28', ['.github/workflows/bench.yml'], '+60 lines', 'chore'),
        ('v11_014', 'feat: add export functionality', 'dev2@example.com', '2026-01-29', ['src/export.py'], '+100 lines', 'feature'),
        ('v11_015', 'docs: add FAQ section', 'dev3@example.com', '2026-01-30', ['docs/faq.md'], '+150 lines', 'docs'),
        ('v11_016', 'fix: correct timezone handling', 'dev1@example.com', '2026-01-31', ['src/utils/time.py'], '+30 -15 lines', 'bugfix'),
        ('v11_017', 'perf: cache search results', 'dev2@example.com', '2026-02-01', ['src/cache.py'], '+80 lines', 'internal'),
        ('v11_018', 'feat: add batch operations', 'dev3@example.com', '2026-02-02', ['src/batch.py'], '+120 lines', 'feature'),
        ('v11_019', 'refactor: improve error handling', 'dev1@example.com', '2026-02-03', ['src/errors.py'], '+40 -20 lines', 'internal'),
        ('v11_020', 'Release v1.1.0', 'dev2@example.com', '2026-02-04', ['CHANGELOG.md'], '+100 lines', 'chore'),
    ]

    # Version 2.0.0 - Breaking changes (commits 36-52)
    v2_commits = [
        ('v2_001', 'BREAKING: change API response format to JSON:API', 'dev1@example.com', '2026-02-05', ['src/api.py'], '+200 -150 lines', 'breaking'),
        ('v2_002', 'BREAKING: remove deprecated auth endpoints', 'dev2@example.com', '2026-02-06', ['src/auth.py'], '-100 lines', 'breaking'),
        ('v2_003', 'feat: add OAuth2 support', 'dev3@example.com', '2026-02-07', ['src/oauth.py'], '+300 lines', 'feature'),
        ('v2_004', 'BREAKING: rename User model fields', 'dev1@example.com', '2026-02-08', ['src/models.py'], '+50 -50 lines', 'breaking'),
        ('v2_005', 'feat: add rate limiting', 'dev2@example.com', '2026-02-09', ['src/ratelimit.py'], '+150 lines', 'feature'),
        ('v2_006', 'fix: OAuth token refresh bug', 'dev3@example.com', '2026-02-10', ['src/oauth.py'], '+20 -10 lines', 'bugfix'),
        ('v2_007', 'docs: migration guide v1 to v2', 'dev1@example.com', '2026-02-11', ['docs/migration_v2.md'], '+500 lines', 'docs'),
        ('v2_008', 'feat: add webhook support', 'dev2@example.com', '2026-02-12', ['src/webhooks.py'], '+200 lines', 'feature'),
        ('v2_009', 'refactor: modernize async handling', 'dev3@example.com', '2026-02-13', ['src/async.py'], '+80 -60 lines', 'internal'),
        ('v2_010', 'fix: webhook delivery reliability', 'dev1@example.com', '2026-02-14', ['src/webhooks.py'], '+30 -15 lines', 'bugfix'),
        ('v2_011', 'feat: add event sourcing', 'dev2@example.com', '2026-02-15', ['src/events.py'], '+250 lines', 'internal'),
        ('v2_012', 'ci: add canary deployments', 'dev3@example.com', '2026-02-16', ['.github/workflows/canary.yml'], '+100 lines', 'chore'),
        ('v2_013', 'feat: implement audit logging', 'dev1@example.com', '2026-02-17', ['src/audit.py'], '+180 lines', 'feature'),
        ('v2_014', 'perf: reduce API response times', 'dev2@example.com', '2026-02-18', ['src/api.py'], '+40 -30 lines', 'internal'),
        ('v2_015', 'docs: API v2 reference', 'dev3@example.com', '2026-02-19', ['docs/api_v2.md'], '+600 lines', 'docs'),
        ('v2_016', 'fix: audit log race condition', 'dev1@example.com', '2026-02-20', ['src/audit.py'], '+15 -8 lines', 'bugfix'),
        ('v2_017', 'Release v2.0.0', 'dev2@example.com', '2026-02-21', ['CHANGELOG.md'], '+200 lines', 'chore'),
    ]

    # Combine all commits
    all_commits_data = v1_commits + v1_1_commits + v2_commits

    for data in all_commits_data:
        commits.append(CommitRecord(
            hash=data[0],
            message=data[1],
            author=data[2],
            timestamp=data[3] + 'T10:00:00Z',
            files_changed=[data[4]],
            diff_summary=data[5],
        ))

    # Build gold classifications with version boundaries
    gold_classifications = {}
    for data in all_commits_data:
        gold_classifications[data[0]] = data[6]

    gold = {
        'classifications': gold_classifications,
        'version_bump': 'major',
        'version_boundaries': {
            'v1.0.0': ['init001', 'init015'],
            'v1.1.0': ['v11_001', 'v11_020'],
            'v2.0.0': ['v2_001', 'v2_017'],
        },
        'expected_sections': {
            'v1.0.0': {'version_bump': 'minor'},
            'v1.1.0': {'version_bump': 'minor'},
            'v2.0.0': {'version_bump': 'major'},
        },
    }

    return {
        'task_id': 'task_hard',
        'commits': commits,
        'gold': gold,
        'description': 'Multi-Version Audit - Infer 3 version boundaries from 52 commits',
    }
