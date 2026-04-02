"""ChangelogEnv client - what users import in their training code."""

from typing import Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import ChangelogAction, ChangelogObservation, ChangelogState


class ChangelogEnv(EnvClient[ChangelogAction, ChangelogObservation, ChangelogState]):
    """
    Client for interacting with the ChangelogEnv environment.

    Usage:
        with ChangelogEnv(base_url='https://YOUR_USERNAME-changelog-env.hf.space').sync() as env:
            result = env.reset(task_id='task_easy')
            result = env.step(ChangelogAction(action_type='classify_commit', ...))
            state = env.state()
    """

    def _step_payload(self, action: ChangelogAction) -> dict:
        """Serialize action to JSON payload for step endpoint."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse step result from server response."""
        obs_data = payload.get('observation', {})
        return StepResult(
            observation=ChangelogObservation(**obs_data),
            reward=payload.get('reward'),
            done=payload.get('done', False),
        )

    def _parse_state(self, payload: dict) -> ChangelogState:
        """Parse state from server response."""
        return ChangelogState(**payload)
