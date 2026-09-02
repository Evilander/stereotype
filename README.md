# StereoType

**A local, judge-free bias auditor for Hugging Face text models. It feeds a model two inputs that differ only by a demographic cue and reports which way the model leans, with intervals.**

A *stereo pair* is two texts identical except for one cue: a name, a pronoun, a group noun, an age. Chat models, embedding models, and text classifiers all make decisions about text, so all three can be probed the same way. The output is a **Stereotype Card**: per-battery estimates with 95 % bootstrap intervals, permutation p-values, a shuffled-cue null, and the smallest effect the run could have detected. There is no composite score and no second model grading the first one.

```
pip install -e ".[dev]"
stereotype audit ibm-granite/granite-4.2-3b --nulls
stereotype audit ibm-granite/granite-embedding-311m-multilingual-r2
stereotype audit unitary/toxic-bert
stereotype compare runs/ibm-granite__granite-4.2-3b runs/Qwen__Qwen3.5-4B
```

Each audit writes `card.json` (the contract), `card.md`, a self-contained `card.html` with balance meters, and the raw per-item measurements as Parquet, into `runs/<model>/`. Example cards for six models are in [`docs/cards/`](docs/cards/).

## What it measures

| battery | family | construct | lineage |
|---|---|---|---|
| `assoc` | LM | log-probability of attribute words after a cue; WEAT-style effect size between cue groups | Kurita et al. 2019; Caliskan et al. 2017 |
| `decision` | LM | P(yes) on 70 decision scenarios as stated age, gender, race vary; logit difference from a fixed baseline | Discrim-Eval, Tamkin et al. 2023 |
| `nameswap` | LM | P(yes) on hiring, lending, housing, healthcare, education, insurance decisions where only the name changes; selection rates and impact ratios | Bertrand & Mullainathan 2004; NYC Local Law 144 |
| `chatiat` | LM | which of two names an attribute word "goes with", scored from the two names' probabilities | LLM-IAT, Bai et al. 2024 |
| `bbq` | LM | stereotype-consistent answers when the context cannot answer the question | BBQ, Parrish et al. 2021 |
| `crows` | LM | likelihood preference on CrowS-Pairs (legacy anchor, kept for comparability) | Nangia et al. 2020 |
| `effort` | LM, experimental | thinking tokens spent on counter-stereotypical versus stereotypical sorting | RM-IAT, Lee & Lai 2025 |
| `contamination` | LM, experimental, off by default | guided-completion accuracy on CrowS-Pairs sentences versus fresh controls; the control set is not yet matched, so the ratio is uninformative | Xu et al. 2024 |
| `seat` | embedding | WEAT on sentence embeddings of bleached templates | SEAT, May et al. 2019 |
| `retrieval` | embedding | rank of identical candidate summaries that differ only by name, against job queries | Bertrand & Mullainathan 2004 |
| `ctf` | classifier | toxicity score gap and false-positive rate for identity terms in neutral sentences | Dixon et al. 2018; Garg et al. 2019 |

Formulas, sources, and what each battery does *not* measure are in [`docs/methodology.md`](docs/methodology.md). Cue lists and their provenance are in [`docs/cues.md`](docs/cues.md); `stereotype probes <battery>` prints the exact probes.

## Reading a card

- Every row is an estimate with a 95 % interval. An interval that covers the centre value means this run could not tell, not that the model is balanced; the `mde` field says how large an effect the run could have found.
- Positive means stereotype-consistent unless the row's note says otherwise; the note always states the direction.
- `--nulls` re-runs each battery once with cue labels shuffled. That is a sanity check on the pipeline. The permutation and sign-flip p-values are the statistical test.
- Likelihood batteries measure what a model would prefer to say. The decision batteries (`decision`, `nameswap`) are closer to behaviour and are the ones to read first.
- Wording on the card is descriptive. A model's measurements lean one way or the other; nothing on the card says why, and nothing on it is a verdict about a group.

## Measured on six models

All numbers below come from `python scripts/run_demo.py --nulls` on one machine (NVIDIA GeForce RTX 4080 SUPER 16 GB, Windows, Python 3.11, torch 2.13, transformers 5.14) on 2 September 2026, seed 0. The full cards, with every row, interval, null run, probe preview, and the exact rendered prompts, are in [`docs/cards/`](docs/cards/). Intervals are 95 % bootstrap. Nothing here is a verdict about a group; each line is a measurement of one model on one probe set.

| subject | family | revision | wall clock |
|---|---|---|---|
| `ibm-granite/granite-4.2-3b` (released 2026-08-25) | chat LM, bf16 | `b7e94730` | 46 min (41 of them in the experimental effort battery) |
| `Qwen/Qwen3.5-4B` | chat LM, bf16 | `851bf6e8` | 24 min |
| `ibm-granite/granite-embedding-311m-multilingual-r2` | embedding | `44399559` | 6 s |
| `Qwen/Qwen3-Embedding-0.6B` | embedding | `97b0c614` | 7 s |
| `unitary/toxic-bert` | classifier | `4d6c22e7` | 2 s |
| `unitary/unbiased-toxic-roberta` | classifier | `36295dd8` | 2 s |

**Decisions (Discrim-Eval, 9 450 prompts, thinking off).** Both chat models say "yes" more readily for every non-baseline race and gender level than for the 60-year-old white male baseline. Granite: Black +0.47 logit [0.33, 0.63], Native American +0.39 [0.25, 0.54], Hispanic +0.25, Asian +0.17, non-binary +0.32 [0.17, 0.47], female +0.22 [0.12, 0.34]. Qwen: Black +0.25 [0.15, 0.38], non-binary +0.16, female +0.11. On age they diverge: Granite leans toward 80- and 90-year-olds (+0.10, +0.13), Qwen leans against them (80: −0.11 [−0.17, −0.05]; 90: −0.17; 100: −0.27 [−0.39, −0.16]). The yes/no surface forms captured 99 % of next-token mass for both models, so these are not refusal artefacts. Selection rates differ by only a few points (Granite 0.49 to 0.51 across levels); the logit contrasts are consistent, not large.

**Name-only decisions (12 templates × 36 names).** Granite favours Black-associated names slightly (white female − Black female −0.22 logit [−0.45, −0.02]), with impact ratios of 0.98 to 1.00, but its per-name scores correlate with name token count at rank correlation 0.51, so tokenisation is doing part of the work. The card also reports each contrast restricted to names with the same token count on both sides; because the 2004 list's white-associated names are mostly one token and its Black-associated names mostly two or three, that restriction leaves one or two names per side and the Granite race contrast changes sign under it (+0.03 for women, −0.34 for men). Read the Granite race contrast as unresolved rather than as a finding. Qwen shows no race contrast (−0.01 [−0.08, +0.07]; +0.02 [−0.05, +0.10]), a small lean toward female names, and no token-count correlation (−0.12).

**Associations (12 024 prompts).** The classic gender/career association reproduces on both: Granite d = 1.39 [1.05, 1.68], Qwen d = 1.47 [1.11, 1.78]; gender/math and gender/science d = 1.2 to 1.4; young/pleasant d = 1.2. The classic race/valence name association does not reproduce: Granite d = −0.70 [−1.20, −0.12] (Black-associated names closer to pleasant words), Qwen d = −0.20 [−0.80, +0.44]. Occupation status leans toward white-associated names on Qwen (d = 0.87 [0.33, 1.30]) and is uncertain on Granite (0.51 [−0.10, 1.13]); the service set contains gender-coded roles, and the card says so.

**Chat-framed association (Bai et al.).** Granite gender/career log-odds +0.89 [0.11, 1.69] (Bai bias 0.35); Qwen +0.31 [0.05, 0.57]. Race/valence: Granite +0.17 [0.03, 0.30], Qwen +0.13 [0.08, 0.19]. The order effect that the design averages out is of the same size as these associations on Granite (+0.63 on gender/career), which is why it is printed.

**BBQ (120 items per category).** Age is the clearest ambiguous-context result: when the context does not say who was forgetful, Granite picks the older person at bias score +0.20 [0.08, 0.33] and Qwen at +0.37 [0.23, 0.50]. Physical appearance is next (+0.13 Granite, +0.12 Qwen), Qwen adds socioeconomic status (+0.15 [0.07, 0.25]), and the remaining categories sit within ±0.1 of zero. Qwen answered "unknown" on every ambiguous race-by-gender item, so that score is undefined and the card says so.

**CrowS-Pairs (legacy anchor).** Granite prefers the `sent_more` sentence in 61.8 % of pairs [59.2, 64.3] (64.3 % per token) and Qwen in 61.6 % [59.1, 64.1]; Granite's by-direction split is 61.4 % on stereotype pairs and 64.2 % on anti-stereotype pairs, which is the kind of ambiguity the construct-validity critique of this benchmark is about.

**Reasoning effort (experimental, Granite only).** Sorting words into association-incompatible categories cost more thinking tokens than compatible ones on every test; pooled over 30 lists the asymmetry is +749 tokens [258, 1 289], with per-test intervals that cross zero and 17 to 75 % of runs hitting the 4 096-token budget. Read as a direction, not a size.

**Contamination (experimental, not in the demo).** On the first full run the guided-completion ratio came out below 1 for both models (Granite 0.18, Qwen 0.35) because the fresh control sentences are far easier to complete than CrowS text. The control is not matched, the number is uninformative, and the battery is off until a matched control exists.

**Embeddings (SEAT and ranking).** Gender/career SEAT d = 1.22 [0.70, 1.60] (Granite embedding) and 1.43 [1.12, 1.73] (Qwen embedding). For identical candidate summaries ranked against ten job queries, Granite's embedder ranks white-associated names above Black-associated ones in 81 % of cross-name comparisons for women [71, 88] and 73 % for men [65, 79]; Black-associated names land in the top quarter 8 to 9 % of the time against 34 to 49 % for white-associated names, an impact ratio of 0.16 to 0.18. Qwen's embedder shows a different pattern: white female above Black female 69 % [61, 77], but white male below white female (P = 0.31 [0.20, 0.42]) and no male race contrast (0.46); its impact ratios run 0.31 (white male) to 1.0 (white female).

**Classifiers (identity terms in neutral sentences).** Both classifiers score neutral sentences as more toxic when they contain sexual-orientation terms (gap +0.12 [0.08, 0.16] for `toxic-bert`, +0.16 [0.12, 0.20] for `unbiased-toxic-roberta`; worst terms "homosexual" and "gay"), then disability and gender terms. The model trained to reduce identity-term false positives does flag fewer neutral sentences on every axis where flags occur (sexual orientation 4 % versus 8 %, disability 3 % versus 7 %, gender 1 % versus 3 %), while its mean score gaps are not uniformly lower. Both facts are on the cards.

## How it works

Every language-model battery except the two experimental ones reads probabilities directly from the model's next-token distribution, or sums log-probabilities over a short continuation, using the model's own chat template with thinking disabled. Chat-template thinking modes are handled through the template's own switch, and the rendered prompt of the first probe is stored in every card so the exact input can be checked. Embedding models run through sentence-transformers with their query prompt when they define one. Classifiers run through `AutoModelForSequenceClassification`; the target label is guessed from the label names or set with `--label`.

Everything runs on one machine. The full language-model set on a 3 B to 4 B model takes tens of minutes on a 16 GB consumer GPU; embedding and classifier batteries take seconds on a CPU.

## Datasets and licences

- `Anthropic/discrim-eval` (CC-BY-4.0) for `decision`.
- BBQ via the `heegyu/bbq` mirror (CC-BY-4.0), snapshot `5d6faae5`; there is no canonical Hub copy, so the mirror and commit are recorded.
- CrowS-Pairs from the `nyu-mll/crows-pairs` repository (CC-BY-SA-4.0), fetched once into the cache. Three items appear in each card's probe preview under that licence.
- The IAT word sets, the 2004 audit-study names, and the identity-term list are transcribed from the cited publications. Everything else (templates, group phrases, occupation sets, reference terms) was written for this package and contains no stereotype content of its own.

## Limitations

- Name lists are proxies for perceived race and gender; they carry frequency and tokenisation effects. The name's own probability never enters a score, and per-name spread is reported, but the proxy stays a proxy.
- Templates are few (2 to 20 per battery). Template-level numbers are on the card so a single template cannot hide behind an average.
- No global multiple-comparison correction is applied across the card. Raw p-values and the MDE are shown so readers can judge power and multiplicity; `stats.holm` is available for a per-battery adjustment.
- Static anchors (`crows`, `bbq`) may be in a model's training data; the fresh batteries exist for that reason, and `contamination` gives a rough memorisation signal.
- English only, United States-centred stimuli.

## Development

```
python -m pytest        # CPU only; uses HuggingFaceTB/SmolLM2-135M-Instruct as the fixture model
ruff check stereotype_audit tests scripts
```

Tests skip model-dependent cases when the fixture model is not in the local Hugging Face cache.

## License

Apache-2.0.
