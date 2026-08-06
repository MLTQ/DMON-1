"""SOL2: a typed, bounded, continuously running neural organism."""

from .config import Sol2Config
from .model import Sol2
from .state import OrganismState

__all__ = ["OrganismState", "Sol2", "Sol2Config"]
