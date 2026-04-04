"""FastAPI app for ChangelogEnv - auto-generated endpoints via openenv.core."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server'))

from openenv.core import create_fastapi_app

from environment import ChangelogEnvironment
from models import ChangelogAction, ChangelogObservation

app = create_fastapi_app(ChangelogEnvironment, ChangelogAction, ChangelogObservation)
