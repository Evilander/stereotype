# Cue lists and their provenance

All cue and stimulus lists ship inside the package under `stereotype_audit/cues/` as JSON with a `source` field. Nothing in a probe is invented stereotype content: the group-attribute pairings come from published stimulus sets, and the sentences that wrap them are neutral templates.

| file | contents | source |
|---|---|---|
| `names_bm2004.json` | 36 first names, nine per race × gender cell | Bertrand & Mullainathan (2004), *Are Emily and Greg More Employable Than Lakisha and Jamal?*, AER 94(4), Table A1 |
| `iat_attributes.json` | pleasant/unpleasant (25 + 25 and 8 + 8), career/family, math/arts, science/arts, male/female terms and names, young/old names, mental/physical disease, temporary/permanent | Caliskan, Bryson & Narayanan (2017), *Science* 356(6334), supplementary tables, reproducing Greenwald et al. (1998) and Nosek et al. (2002) |
| `groups.json` | descriptive noun phrases per axis: gender, age, race/ethnicity, religion, nationality, disability, sexual orientation, socioeconomic status | axis inventory after BBQ (Parrish et al. 2021) and Discrim-Eval (Tamkin et al. 2023); phrases written for this package |
| `occupations.json` | eight high-status and eight service occupations | constructed for this package after the supervisor/clerical contrast in Bai et al. (2024); not a published stimulus set |
| `identity_terms.json` | identity terms by axis plus four neutral reference terms | Dixon et al. (2018), conversationai unintended-ml-bias-analysis term list; reference terms chosen for this package |

Templates live under `stereotype_audit/templates/` and can be printed with `stereotype probes <battery>`.

Names signal *perceived* race and gender to readers, as pretested in the original study; they say nothing about any real person. Group phrases are plain descriptors. Attribute words include the negative half of the IAT stimulus sets (for example "poison", "prison"); they are there because the measurement is whether the model links them to one group more than another, and the card reports that link as a number, not as a statement about the group.

Datasets used as anchors, with licences: `Anthropic/discrim-eval` (CC-BY-4.0), `heegyu/bbq` mirror of BBQ (CC-BY-4.0), CrowS-Pairs from `nyu-mll/crows-pairs` on GitHub (CC-BY-SA-4.0).
