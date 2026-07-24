"""DMON-1: growing a creature whose form is not specified anywhere.

See PROJECT.md for the session briefing, ARCHITECTURE.md for why things are shaped
the way they are, HANDOFF.md for what to do first.
"""

from .substrate import Substrate, SubstrateConfig, descriptors, make_sources

__all__ = ["Substrate", "SubstrateConfig", "descriptors", "make_sources"]
