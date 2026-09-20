"""
Shared long-lived services used by the engine's nodes.

One BrainMemory and one EpisodicLogger per process, the same arrangement the
old CrewAI runner had, so SQL learning, semantic memory and task history all
go through a single instance each.
"""

from __future__ import annotations

from app.state.brain import BrainMemory
from app.state.episodic_log import EpisodicLogger

brain = BrainMemory()
episodic_logger = EpisodicLogger()
