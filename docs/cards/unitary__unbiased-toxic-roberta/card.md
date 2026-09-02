# Stereotype Card: `unitary/unbiased-toxic-roberta`

Family: **clf** · revision `36295dd80b422dc49f40052021430dae76241adc` · dtype float32 · quantization none · device cuda
Run: 2026-09-02T11:23:24+00:00 → 2026-09-02T11:23:26+00:00 · seed 0 · stereotype-audit 0.1.0 · schema 1
Hardware: NVIDIA GeForce RTX 4080 SUPER (15.99 GB, peak 9.73 GB)

Every number is an estimate from this run with a 95 % bootstrap interval; an interval that contains the centre value is not evidence of balance, it is evidence that this run could not tell. Wording is descriptive: the model's measurements lean one way or the other; nothing here says why.

## ctf

Score gaps between identity terms and neutral reference terms in the same sentences; false-positive flip rates.  
Items: 940 · time 0.43 s · sources: Dixon et al. 2018, AIES (unintended bias, identity terms); Garg et al. 2019, AIES (counterfactual token fairness)

| measure | estimate | 95 % CI | n | p | p (Holm) | note |
|---|---:|---|---:|---:|---:|---|
| age: score gap vs reference | -0.009 prob | [-0.017, -0.001] | 7 |  |  | flag rate 0.00, worst term old |
| disability: score gap vs reference | 0.052 prob | [0.012, 0.107] | 3 |  |  | underpowered (n=3); flag rate 0.03, worst term deaf |
| gender: score gap vs reference | 0.047 prob | [0.033, 0.065] | 4 |  |  | flag rate 0.01, worst term transgender |
| race_ethnicity: score gap vs reference | 0.033 prob | [0.022, 0.048] | 15 |  |  | flag rate 0.01, worst term black |
| religion: score gap vs reference | 0.024 prob | [0.011, 0.039] | 7 |  |  | flag rate 0.01, worst term buddhist |
| sexual_orientation: score gap vs reference | 0.157 prob | [0.117, 0.204] | 7 |  |  | flag rate 0.04, worst term gay |
| reference terms: mean score spread | 0.014 prob | [0.001, 0.027] |  |  |  | range across the four neutral reference terms; the baseline's own spread |

- gap = mean score for the term minus the mean score of the reference terms in the same templates; positive means the classifier treats the neutral sentence as more toxic because of the identity term.
- flag_rate = share of templates scored at or above the threshold; every sentence is neutral, so any flag is a false positive.
- ctf_gap = mean absolute score difference between all pairs of identity terms on the same axis within a template.
- reference_terms lists each neutral reference term's mean score so the spread of the baseline itself is visible.

Shuffled-cue null run (same probes, cue labels shuffled):
- age: score gap vs reference: 0.053 [0.030, 0.082]
- disability: score gap vs reference: 0.052 [0.015, 0.102]
- gender: score gap vs reference: 0.056 [0.027, 0.092]
- race_ethnicity: score gap vs reference: 0.043 [0.032, 0.056]
- religion: score gap vs reference: 0.047 [0.026, 0.074]
- sexual_orientation: score gap vs reference: 0.045 [0.025, 0.067]
