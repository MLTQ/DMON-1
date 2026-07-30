"""Back-compat re-export — stream.py owns the corpus/stream types."""

from .stream import CharCorpus, CharacterVocabulary, ContinuousCharStream, ensure_shakespeare

__all__ = [
    "CharCorpus",
    "CharacterVocabulary",
    "ContinuousCharStream",
    "ensure_shakespeare",
]
