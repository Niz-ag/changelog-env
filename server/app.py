"""FastAPI app for ChangelogEnv - auto-generated endpoints via openenv.core."""

from openenv.core.env_server import create_fastapi_app

from environment import ChangelogEnvironment

# create_fastapi_app() generates all endpoints:
# /ws (WebSocket), /reset, /step, /state, /health, /web, /docs
app = create_fastapi_app(ChangelogEnvironment)
