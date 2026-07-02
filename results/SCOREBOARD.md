# Full results

Tables 1, 3–6 below are the paper's published numbers, reproduced under AdaVD's own
protocol (SD-1.4, DPM-Solver, 30 steps, CFG 7.5, CLIP score `CS↓` for target concepts,
FID `↓` for retained concepts).

`instance/`, `style/`, `celebrity/` contain a separate, later reproduction run of the
same code and configuration (`ns=10` generated images per cell). It shows the same
overall pattern — CARE improves target erasure and non-target preservation on most
cells — but individual close-margin win/loss calls can differ from the tables below,
since CLIP/FID at this sample size (FID especially) have real run-to-run variance. See
the JSON files there for that run's exact numbers; treat them as a reproduction check,
not a byte-exact source for the numbers in this file.

**Headline: erase strictly improves 8 of 12 target-CS cells (1 tie, 3 losses — all three
in multi-concept Mickey Mouse erasure, see §4.2) · preservation strictly improves 33 of
36 FID cells (1 tie, 2 losses)**, at matched inference cost to AdaVD.

## Table 1 — Instance erasure (single- and multi-concept)

| Concept | Snoopy | Mickey | SpongeBob | Pikachu | Dog | Legislator |
|---|---|---|---|---|---|---|
| SD v1.4 | 28.51 | 26.57 | 27.43 | – | – | – |
| **Erase Snoopy** | CS↓ | FID↓ | FID↓ | FID↓ | FID↓ | FID↓ |
| AdaVD | 20.28 | 5.72 | 8.56 | 5.79 | 2.32 | 6.07 |
| **CARE** (γ=0.5) | **19.42** | **4.43** | **7.42** | **4.74** | **1.95** | **5.32** |
| **Erase Snoopy + Mickey** | CS↓ | CS↓ | FID↓ | FID↓ | FID↓ | FID↓ |
| AdaVD | 20.29 | **19.93** | 9.34 | 5.84 | 2.41 | 6.43 |
| **CARE** (γ=0.5) | **19.45** | 22.55 | **7.47** | **4.98** | **2.18** | **5.34** |
| **Erase Snoopy + Mickey + SpongeBob** | CS↓ | CS↓ | CS↓ | FID↓ | FID↓ | FID↓ |
| AdaVD | **19.39** | **19.73** | 20.34 | 6.86 | 2.79 | 7.26 |
| **CARE** (γ=0.5) | 19.45 | 22.56 | **18.40** | 6.86 | **2.27** | **5.55** |

## Table 3 — Art-style erasure

| Concept | Van Gogh | Picasso | Monet | Andy Warhol | Caravaggio |
|---|---|---|---|---|---|
| SD v1.4 | 29.21 | 29.06 | 29.02 | – | – |
| **Erase Van Gogh** | CS↓ | FID↓ | FID↓ | FID↓ | FID↓ |
| AdaVD | 24.87 | 6.82 | 2.66 | **8.36** | 6.84 |
| **CARE** (γ=0.2) | **23.63** | **5.86** | **2.44** | 8.59 | **4.80** |
| **Erase Picasso** | FID↓ | CS↓ | FID↓ | FID↓ | FID↓ |
| AdaVD | 5.49 | 26.99 | 2.33 | 9.38 | 7.05 |
| **CARE** (γ=0.2) | **5.18** | 26.99 | **1.69** | **7.53** | **4.36** |
| **Erase Monet** | FID↓ | FID↓ | CS↓ | FID↓ | FID↓ |
| AdaVD | 6.94 | 6.50 | 26.30 | 8.46 | 7.19 |
| **CARE** (γ=0.2) | **5.85** | **5.46** | **24.58** | **7.21** | **4.22** |

## Table 4 — Celebrity erasure

| Concept | Bruce Lee | Marilyn Monroe | Melania Trump | Anne Hathaway | Tom Cruise |
|---|---|---|---|---|---|
| SD v1.4 | 30.77 | 27.70 | 29.80 | 31.96 | 31.12 |
| **Erase Bruce Lee** (CS↓ for target, FID↓ for rest) |||||
| AdaVD | 20.67 | 6.68 | 5.08 | 6.39 | 13.11 |
| **CARE** (γ=0.2) | **18.42** | **3.29** | **4.37** | **5.07** | **5.89** |
| **Erase Marilyn Monroe** (CS↓ for target, FID↓ for rest) |||||
| AdaVD | 7.88 | 19.87 | 4.46 | 5.43 | 9.33 |
| **CARE** (γ=0.2) | **5.92** | **17.73** | **4.36** | **5.01** | **4.53** |
| **Erase Melania Trump** (CS↓ for target, FID↓ for rest) |||||
| AdaVD | 7.32 | 6.86 | 23.28 | **6.52** | 5.74 |
| **CARE** (γ=0.5) | **6.33** | **4.06** | **21.90** | 6.85 | **4.38** |

## Table 5 — Efficiency (10-concept erasure, seconds, 1×A40, 10 images)

| Method | Data prep | Model finetune | Image gen | Total |
|---|---|---|---|---|
| Concept Ablation | 9290 | 1120 | 0.9 | 10419 |
| SPM | 0 | 72850 | 1.7 | 72867 |
| MACE | 303 | 232 | 0.9 | 544 |
| SLD | 0 | 0 | 1.4 | 14 |
| AdaVD | 4 | **0** | 1.8 | 22 |
| **CARE** | 1.2 | **0** | +0.7% | ≈22 |

CARE adds a 1.2s offline retained-subspace build and +0.7% per-image generation
overhead over AdaVD — zero fine-tuning, matching AdaVD's training-free cost profile.

## Table 6 — Ablations (reduced-scope: 15 templates, NS=10, seed 0)

**(a) Shrinkage parameter γ** — separable concept (Bruce Lee, probe: Tom Cruise) vs.
entangled concept (Melania Trump, probe: Anne Hathaway):

| γ | Bruce Lee CS↓ | Bruce Lee FID↓ | Melania CS↓ | Melania FID↓ |
|---|---|---|---|---|
| AdaVD (γ→∞) | 18.95 | 11.72 | 22.33 | 6.93 |
| 0.05 | 29.99 | 4.09 | 29.46 | 4.21 |
| 0.10 | 30.01 | 4.42 | 29.39 | 4.35 |
| **0.20** ⋆ | **17.19** | 6.53 | 21.59 | 5.09 |
| 0.50 | 17.27 | 7.56 | 21.17 | 5.64 |
| 1.00 | 17.65 | 10.06 | **20.99** | 5.33 |

⋆ deployed setting for separable concepts. Very small γ over-protects the retained
subspace and collapses erasure entirely (CS≈30, near the unedited model) — preservation
alone is not sufficient; γ must still permit target removal.

**(b) Anchor-bank composition and size** (Bruce Lee erasure):

| Bank type | CS↓ | FID↓ | | M | CS↓ | FID↓ |
|---|---|---|---|---|---|---|
| blind (disjoint) | **17.19** | **6.53** | | 2 | 17.96 | 6.13 |
| random | 19.02 | 6.85 | | 4 | **16.44** | **5.72** |
| unrelated | 19.04 | 8.77 | | 6 | 17.19 | 6.53 |
| related | 22.88 | 9.98 | | 8 | 17.25 | 6.25 |
| | | | | 10 | 17.34 | 5.83 |

A disjoint anchor bank (structurally related to, but not overlapping, the target
identity) gives the strongest result; anchors too close to the target absorb the
target-specific direction and weaken erasure. CARE is not sensitive to the exact bank
size M — deployed default is M=6.

## Excluded from this evaluation surface

An open NSFW/I2P smoke-test cell (n=4) is **not** included above — CARE does not yet
beat AdaVD there and it was not part of the paper's evaluation surface.
