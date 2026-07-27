# `__init__.py`

## Purpose

Defines the small public API for the SOL prototype without importing its trainer or CLI.

## Components

### Public exports
- **Does**: Exposes the field configuration, persistent state, trace, vocabulary, and
  continuous stream.
- **Interacts with**: `model.py` and `stream.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiments and tests | Stable imports from `sol` | Renaming or removing exports |
