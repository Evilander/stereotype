# Changelog

## 0.1.0 — 2026-09-02

First release.

- Stereo-pair auditing for three model families: causal language models, embedding models, text classifiers.
- Language-model batteries: `assoc`, `decision` (Discrim-Eval), `nameswap`, `chatiat`, `bbq`, `crows`; experimental `effort` and `contamination` (the latter off by default until its control set is matched).
- Diagnostics on every card: answer mass before renormalisation, Holm-adjusted p-values, minimum detectable effect, per-name dispersion and token-matched contrasts for name swaps, order effects for the chat-framed association, option-position shares for BBQ.
- Embedding batteries: `seat`, `retrieval`. Classifier battery: `ctf`.
- Statistics: percentile and cluster bootstraps, WEAT effect sizes, permutation and sign-flip tests, Holm adjustment, impact ratios, minimum detectable effect, shuffled-cue null runs.
- Stereotype Card as JSON, Markdown, and self-contained HTML; `stereotype audit`, `card`, `compare`, `probes`, `batteries` commands.
