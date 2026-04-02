"""ChangelogEnv - A Release Notes Generation Environment for RL Agents."""

# Imports handled directly - avoid relative imports for package compatibility
__all__ = [
    'ChangelogEnv',
    'ChangelogAction',
    'ChangelogObservation',
    'ChangelogState',
    'CommitRecord',
]

# Lazy imports to avoid issues during testing
def __getattr__(name):
    if name in ('ChangelogEnv',):
        from client import ChangelogEnv
        return ChangelogEnv
    if name in ('ChangelogAction', 'ChangelogObservation', 'ChangelogState', 'CommitRecord'):
        from models import ChangelogAction, ChangelogObservation, ChangelogState, CommitRecord
        return {'ChangelogAction': ChangelogAction, 'ChangelogObservation': ChangelogObservation,
                'ChangelogState': ChangelogState, 'CommitRecord': CommitRecord}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
