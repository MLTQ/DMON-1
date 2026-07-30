# Findings imported into grok/

Distilled from `sol/` experiments S0–S24. These are **constraints**, not suggestions.

## Capability

1. Sparse directed cellular fields **learn** continuous char LM; they trail a matched GRU
   by ~0.2 BPC at equal param/update budget until proven otherwise.
2. Persistent distributed state is causal: reset-each-token and cell-shuffle ablations
   must both hurt. If they do not, the substrate is not working.
3. Learned slow edge scalars alone do not close the GRU gap.
4. Early metabolism and first-pass cell reward were neutral or slightly harmful at S0.

## Credit

5. Exact truncated BPTT handles within-window credit. Cross-window credit needs
   **event memory** (eligibility) that delayed reward can meet.
6. Channel-shaped decoder correction (transpose of forward message transform) is the
   right reverse signal; scalar reverse waves alone are weak.
7. Instantaneous eligibility-routing amplitude is a dead end until calibrated, and even
   then often capability-neutral. Prefer delayed fitness assignment over bigger scores.

## Morphology

8. Grow/prune only from **causal** with/without probes inside the living body.
9. Global sequence-loss benefit (S17) improves adaptive connectomes vs local credit only.
10. Fixed ABBA structural trials phase-lock to the corpus — randomize assignments for
    honest inference (S23/S24). Conservative p-thresholds suppress useful growth (S25).

## Process

11. Probes-only controls are mandatory for any rewiring claim.
12. Full held-out curves and multi-seed paired gaps beat single best-BPC headlines.
13. Default-disable experimental organs; checkpoint compatibility for older runs.

## grok policy

- **Default on**: dendritic attention, eligibility, reverse vector credit, fast efficacy,
  multi-lane chunk training, reset/shuffle eval.
- **Default off**: metabolism, structural rewiring, reverse-credit routing games.
- **Promotion bar for S0**: held-out BPC competitive with matched GRU on identical stream.
