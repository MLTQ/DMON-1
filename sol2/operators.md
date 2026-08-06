# `operators.py`

## Purpose

Provides the bounded-operator treatment used to test whether SOL/Fable's long-horizon
gradient cliff comes from unconstrained shared transforms.

## Components

### `EffectiveLinear`

- Owns an ordinary raw weight and optional bias.
- In bounded mode, estimates the leading singular value by power iteration and scales
  only when the estimate exceeds the configured limit.
- In unbounded mode, uses the exact same raw parameters without rescaling.
- Reports the effective spectral norm for health telemetry.

### `inverse_sigmoid`

Converts desired initial gate values into biases.

## Contracts

- Bounded and unbounded treatments have identical trainable parameter counts.
- Bounding never amplifies a weight whose estimated norm is below the limit.
- Power-iteration vectors are buffers, not learned parameters.
- A one-step estimate is not a mathematical hard bound. Exact norm telemetry audits
  estimator lag during the experiment.
