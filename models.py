"""Pydantic models for ChangelogEnv with correct openenv.core base class inheritance."""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel

from openenv.core.env_server import Action, Observation, State


class CommitRecord(BaseModel):
    """Represents a single git commit."""
    hash: str
    message: str
    author: str
    timestamp: str
    files_changed: List[str]
    diff_summary: str


class ChangelogObservation(Observation):
    """
    Observation returned to the agent after each step.

    Note: done and reward are INHERITED from Observation base class.
    Do NOT redefine them here.
    """
    task_id: str
    commits: List[CommitRecord]
    draft: Dict[str, List[str]] = {}
    classified: Dict[str, str] = {}
    version_bump: Optional[str] = None
    last_action_result: str = ''
    score_so_far: float = 0.0


class ChangelogAction(Action):
    """
    Action taken by the agent.

    action_type must be one of:
    - classify_commit
    - add_bullet
    - remove_bullet
    - set_version
    - reorder_sections
    - submit
    - noop
    """
    action_type: str
    commit_hash: Optional[str] = None
    label: Optional[str] = None
    section: Optional[str] = None
    content: Optional[str] = None
    bullet_index: Optional[int] = None
    version_bump: Optional[str] = None


class ChangelogState(State):
    """
    Internal state of the environment.

    Note: episode_id and step_count are INHERITED from State base class.
    Do NOT redefine them here.
    """
    task_id: str = ''
    max_attempts: int = 20
    started_at: str = ''
