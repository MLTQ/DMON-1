# `__init__.py`

## Purpose

Defines the small public surface of the SOL2 experimental package.

## Components

- `Sol2Config` — complete, checkpointable experiment configuration.
- `Sol2` — typed persistent organism.
- `OrganismState` — state that survives optimizer boundaries.

## Contracts

Importing `sol2` has no training, filesystem, or device side effects.
