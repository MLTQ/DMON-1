"""SOL: a persistent, streamed neural cellular organism prototype.

The public surface intentionally exposes the field, its persistent state, and the
character stream primitives without importing the training CLI.
"""

from .model import FieldState, FieldTrace, SolConfig, SparseAxonField
from .stream import CharacterVocabulary, ContinuousCharStream

__all__ = [
    "CharacterVocabulary",
    "ContinuousCharStream",
    "FieldState",
    "FieldTrace",
    "SolConfig",
    "SparseAxonField",
]
