# S7: Distributed predictive coding

## Question

Can stronger backward task signal improve SOL by asking internal cells to predict the
next character during development, while designated output cells remain the only
emitted text organ?

## Candidate mechanism

The experiment projected cell hidden state through the existing character embedding,
adding no trainable parameters and no inference path. Emitted output-organ
cross-entropy remained the reported BPC and the source of streamed reward. Three
variants were tested:

1. every cell predicts through a fully tied embedding;
2. every cell predicts through a decoder-detached embedding, so classifier gradients
   enter cell state and incoming axons but do not directly rewrite the sensory codebook;
3. only ordinary body cells receive the detached local objective, leaving sensory and
   output organs on their native objectives.

All runs used Tiny Shakespeare, 16 cells by 16 channels, four dendrites, two message
steps, 800 updates, four lanes, 16-character windows, seed-matched initialization, no
metabolism, and zero added parameters. Gain `0.03` was the replicated setting.

## Initial falsification

The fully tied all-cell decoder was strongly seed-sensitive. Final BPC deltas versus
matched controls were:

| Seed | Control final | Candidate final | Candidate − control |
|---:|---:|---:|---:|
| 7 | 3.84640 | 3.75234 | −0.09406 |
| 13 | 3.79172 | 3.88243 | +0.09072 |
| 21 | 3.74129 | 3.88346 | +0.14217 |

Mean final delta was `+0.04628` BPC worse. Seed-7 gains `0.1` and `0.3` also worsened
best BPC to `3.96629` and `3.91684`, so the result was not a monotonic benefit hidden at
higher strength.

Detaching the decoder-side embedding removed most of the damage but remained
unresolved: final BPC was `3.77733`, `3.80294`, and `3.77767`, for a mean improvement of
only about `0.007` BPC and a worse mean best checkpoint.

## Body-only result

Restricting local prediction to ordinary body cells improved final BPC on all three
seeds when delayed reward was disabled:

| Seed | Control best/final | Body prediction best/final | Final improvement |
|---:|---:|---:|---:|
| 7 | 3.79295 / 3.84640 | 3.79248 / 3.83252 | +0.01388 |
| 13 | 3.75953 / 3.79172 | 3.71895 / 3.71895 | +0.07277 |
| 21 | 3.68405 / 3.74129 | 3.71216 / 3.72014 | +0.02115 |

Mean final improvement was `0.03593` BPC. Mean best improvement was `0.00432`, and mean
paired evaluation improvement was `0.00854`.

That isolated result did not survive the real SOL learning loop. With delayed cell
reward active and only fast efficacy disabled:

| Seed | Control best/final | Body prediction best/final | Final improvement |
|---:|---:|---:|---:|
| 7 | 3.80448 / 3.90971 | 3.81650 / 3.83261 | +0.07711 |
| 13 | 3.69447 / 3.81898 | 3.71966 / 3.83395 | −0.01496 |
| 21 | 3.71146 / 3.72659 | 3.69853 / 3.78413 | −0.05754 |

Mean final improvement was only `0.00153` BPC, mean best was `0.00809` worse, and two
of three seeds regressed at the endpoint. Mean paired evaluation improvement was
`0.00188` BPC, far below a practical or statistical claim.

## Decision

Distributed next-character supervision is rejected in this form. It can improve a
reward-disabled field, but it does not add capability once SOL's existing continuous
reward path is present. The experimental code was removed rather than retained behind a
dormant flag.

The negative result sharpens the architectural requirement: backward signal should
remain event-addressed and coordinated with the existing reward/eligibility process,
not introduce a second global teaching objective that competes with it. The queued S6
within-organism exploratory-traffic GPU experiment remains the next structural test
after CUDA service is restored.
