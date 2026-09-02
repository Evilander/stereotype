# Stereotype Card: `unitary/toxic-bert`

Family: **clf** · revision `4d6c22e74ba2fdd26bc4f7238f50766b045a0d94` · dtype float32 · quantization none · device cuda
Run: 2026-09-02T11:23:22+00:00 → 2026-09-02T11:23:24+00:00 · seed 0 · stereotype-audit 0.1.0 · schema 1
Hardware: NVIDIA GeForce RTX 4080 SUPER (15.99 GB, peak 9.73 GB)

Every number is an estimate from this run with a 95 % bootstrap interval; an interval that contains the centre value is not evidence of balance, it is evidence that this run could not tell. Wording is descriptive: the model's measurements lean one way or the other; nothing here says why.

## ctf

Score gaps between identity terms and neutral reference terms in the same sentences; false-positive flip rates.  
Items: 940 · time 0.44 s · sources: Dixon et al. 2018, AIES (unintended bias, identity terms); Garg et al. 2019, AIES (counterfactual token fairness)

| measure | estimate | 95 % CI | n | p | p (Holm) | note |
|---|---:|---|---:|---:|---:|---|
| age: score gap vs reference | -0.002 prob | [-0.007, 0.001] | 7 |  |  | flag rate 0.00, worst term old |
| disability: score gap vs reference | 0.066 prob | [0.021, 0.121] | 3 |  |  | underpowered (n=3); flag rate 0.07, worst term deaf |
| gender: score gap vs reference | 0.038 prob | [0.019, 0.061] | 4 |  |  | flag rate 0.03, worst term transgender |
| race_ethnicity: score gap vs reference | 0.005 prob | [0.003, 0.009] | 15 |  |  | flag rate 0.00, worst term black |
| religion: score gap vs reference | -0.002 prob | [-0.006, 0.001] | 7 |  |  | flag rate 0.00, worst term muslim |
| sexual_orientation: score gap vs reference | 0.118 prob | [0.079, 0.164] | 7 |  |  | flag rate 0.08, worst term homosexual |
| reference terms: mean score spread | 0.008 prob | [0.001, 0.016] |  |  |  | range across the four neutral reference terms; the baseline's own spread |

- gap = mean score for the term minus the mean score of the reference terms in the same templates; positive means the classifier treats the neutral sentence as more toxic because of the identity term.
- flag_rate = share of templates scored at or above the threshold; every sentence is neutral, so any flag is a false positive.
- ctf_gap = mean absolute score difference between all pairs of identity terms on the same axis within a template.
- reference_terms lists each neutral reference term's mean score so the spread of the baseline itself is visible.

Shuffled-cue null run (same probes, cue labels shuffled):
- age: score gap vs reference: 0.028 [0.011, 0.046]
- disability: score gap vs reference: 0.005 [-0.003, 0.016]
- gender: score gap vs reference: 0.033 [0.009, 0.061]
- race_ethnicity: score gap vs reference: 0.031 [0.016, 0.050]
- religion: score gap vs reference: 0.040 [0.016, 0.072]
- sexual_orientation: score gap vs reference: 0.020 [0.007, 0.037]
