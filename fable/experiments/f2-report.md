# F2 — adaptability under regime cycling

## Savings (cost of re-adapting to a returning regime)

Cost = mean excess BPC over the first 2048 chars of a block relative to that same block's final quarter. Falling cost across visits = the learner is retaining something.

**A-only arms are the ongoing-learning control**: they walk the same position blocks with no regime change, so any decline there is general improvement, not adaptation. Savings are real only to the extent the cycled slope exceeds the A-only slope.

| Arm | Seed | Stream | Visit 0 | Visit last | Slope /visit | Visits |
|---|---|---|---:|---:|---:|---:|
| aonly_creature_s13 | 13 | A-only | +0.1315 | +0.0512 | -0.0803 | 16 |
| aonly_creature_s21 | 21 | A-only | +0.1255 | +0.0556 | -0.0699 | 16 |
| aonly_creature_s7 | 7 | A-only | +0.1373 | +0.0573 | -0.0801 | 16 |
| aonly_gru_s13 | 13 | A-only | +0.0826 | +0.0252 | -0.0573 | 16 |
| aonly_gru_s21 | 21 | A-only | +0.0743 | +0.0255 | -0.0487 | 16 |
| aonly_gru_s7 | 7 | A-only | +0.0851 | +0.0333 | -0.0518 | 16 |
| cycled_creature_s13 | 13 | cycled | +1.1523 | +0.9136 | -0.2387 | 16 |
| cycled_creature_s21 | 21 | cycled | +1.1391 | +0.9230 | -0.2161 | 16 |
| cycled_creature_s7 | 7 | cycled | +1.1925 | +0.9757 | -0.2169 | 16 |
| cycled_gru_s13 | 13 | cycled | +0.3731 | +0.2391 | -0.1340 | 16 |
| cycled_gru_s21 | 21 | cycled | +0.3312 | +0.2016 | -0.1296 | 16 |
| cycled_gru_s7 | 7 | cycled | +0.3290 | +0.2037 | -0.1253 | 16 |

- **creature / A-only** (3 seeds): visit0 +0.1314 → last +0.0547, mean slope **-0.0767/visit**
- **creature / cycled** (3 seeds): visit0 +1.1613 → last +0.9374, mean slope **-0.2239/visit**
- **gru / A-only** (3 seeds): visit0 +0.0807 → last +0.0280, mean slope **-0.0526/visit**
- **gru / cycled** (3 seeds): visit0 +0.3444 → last +0.2148, mean slope **-0.1296/visit**

**Regime-attributable savings** (cycled slope − A-only slope; more negative = more genuine adaptation):

- **creature**: -0.2239 − -0.0767 = **-0.1471/visit**
- **gru**: -0.1296 − -0.0526 = **-0.0770/visit**

## Interference (regime-A steady state: cycled vs A-only)

Cycled arms see half the A-tokens, so a gap here bundles interference with reduced A-exposure; the creature-vs-GRU *difference* in that gap is the comparable quantity.

| Kind | Seed | Cycled A steady | A-only steady | Gap |
|---|---|---:|---:|---:|
| creature | 13 | 2.3902 | 2.1891 | +0.2011 |
| creature | 21 | 2.4433 | 2.2442 | +0.1990 |
| creature | 7 | 2.4154 | 2.2182 | +0.1972 |
| gru | 13 | 2.2267 | 2.0495 | +0.1772 |
| gru | 21 | 2.2145 | 2.0473 | +0.1672 |
| gru | 7 | 2.2215 | 2.0440 | +0.1775 |

- **creature** mean interference gap: **+0.1991 BPC**
- **gru** mean interference gap: **+0.1739 BPC**

## Compression sanity (A-only steady state)

| Kind | Seed | A-only steady BPC |
|---|---|---:|
| creature | 13 | 2.1891 |
| creature | 21 | 2.2442 |
| creature | 7 | 2.2182 |
| gru | 13 | 2.0495 |
| gru | 21 | 2.0473 |
| gru | 7 | 2.0440 |

- **creature** mean: **2.2172 BPC**
- **gru** mean: **2.0469 BPC**
- creature − gru: **+0.1702 BPC**

## Health

Arms with skipped (non-finite gradient) updates: none
