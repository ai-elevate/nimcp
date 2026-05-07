# Grounded Language Module

**Path**: `src/language/nimcp_grounded_language.c` + sibling files
**Public header**: `include/language/nimcp_grounded_language.h`
**Internal layout**: `src/language/nimcp_grounded_language_internal.h`

Word↔concept binding lexicon with multi-turn discourse state, immune
content inspection, LGSS safety gates, and a cognitive event bus that
distributes comprehension/production events to ToM, inner speech,
imagination, and other cognitive modules.

## Runtime feature flags

Every behaviour-changing feature is OFF by default and exposed via a
`grounded_language_set_<feature>_enabled` setter, plus a getter for
telemetry. Legacy callers see bit-identical behaviour until they flip
the flag. Features added by the full-lang-walkthrough campaign:

| Flag                              | Default | Effect when ON                                                              |
|-----------------------------------|---------|-----------------------------------------------------------------------------|
| `enable_negation_inversion`       | TRUE    | Tier-2 #3 — invert activation sign for words after a negation cue           |
| `enable_sense_disambiguation`     | FALSE   | Tier-2 #6 — pick polysemous sense by intent-cosine                          |
| `enable_speech_act_classification`| FALSE   | TB-9 — populate `result->speech_act` (5-class rule-based)                    |
| `enable_sentence_segmentation`    | FALSE   | TB-6 — split input on `.`/`!`/`?` and recurse per sentence                  |
| `enable_topic_shift_detection`    | FALSE   | TB-10 — flag boundaries when latest turn cosine to mean prior K falls below |
| `enable_reconsolidation`          | FALSE   | TA-5 — decay binding strength of negated content words                      |

SNN-bridge-side flags (in `snn_lang_config_t`):

| Flag                          | Default     | Effect when ON                                  |
|-------------------------------|-------------|-------------------------------------------------|
| `enable_da_modulation`        | TRUE+gain=0 | TA-3 — gate STDP weight delta by current DA     |
| `enable_trigram_learning`     | FALSE       | TA-4 — extend bigram learn to (a,b)→c trigrams  |
| `enable_snn_spike_routing`    | FALSE       | PA-3 — route Broca/Wernicke spikes through bridge |
| `min_produce_words`           | 0 (off)     | TB-7 — suppress EOS until N words emitted       |
| `max_produce_words`           | 0 (off)     | TB-7 — hard cap at N words regardless of EOS    |
| streaming `set_stream_callback` | NULL    | TB-8 — fire per-token callback during produce   |

## Multi-turn state

Discourse turn ring + anaphora referent ring + bigram-spectrum count
matrix all persist via `grounded_language_save_multiturn_state` /
`_load_multiturn_state` (TA-1 — magic-tagged sidecar format).

## Persistence

Lexicon entries + bindings serialize via the persistence sidecar
(`grounded_language_persistence.c`); the multiturn state is a separate
sidecar with its own magic + version header. **Per-instance data
(TC-12)** lives directly on `struct grounded_language` —
`anaphora_state`, `bigram_spectrum`, plus a per-gl `tc12_lock` mutex
lazy-initialized on first use.

## Event bus (cognitive integration)

`grounded_language_subscribe_ex(gl, fn, ctx, type_mask, priority)`
hooks any cognitive consumer to one or more of:

- `GL_EVENT_NEW_WORD` — fresh lexicon entry
- `GL_EVENT_GROUNDED` — successful concept binding
- `GL_EVENT_COMPREHENDED` — comprehend finished
- `GL_EVENT_PRODUCED` — produce finished
- `GL_EVENT_NEEDS_GROUNDING` — low-confidence word seen

Pre-canned wrappers in `nimcp_grounded_language_cognitive_bridge.c`
attach inner_speech / imagination / **theory_of_mind (TC-13: real
`tom_observe` calls, not just logs)** / empathy / introspection /
reasoning / narrative / metacognition / analogical / emergent_language.

## Safety integration

- **LGSS** (TA-2) — input + output gates evaluate every comprehend +
  produce against the safety KB. Blocks bump `lgss_inputs_blocked` /
  `lgss_outputs_blocked` and emit audit events.
- **Brain immune system** (IM-3) — content inspection on comprehend
  via 5 rule-based heuristics (NaN/Inf, Welford-tracked outlier,
  repetition spam, lexicon collision, negation cascade). Inflammation
  > 0.5 registers an antigen; > 0.7 skips engram encoding.

## Cycle coordinator integration

`brain_tick_language` runs at 16ms (`BRAIN_CYCLE_LANGUAGE`). Drains
broca/wernicke bio-router inboxes, ticks the language immune bridge
+ orchestrator, fires the SNN spike-routing path, and at ~1Hz cadence
(CC-1) refreshes the bigram-spectrum FFT cache via
`grounded_language_tick_bigram_spectrum`.

## Tests

`tests/unit/test_lang_*.c` — 13 standalone-C tests under the
`lang_smoke` ctest label. Each Tier-A/B/C campaign feature has its own
test file. **Footgun**: after `gl_stats_t` or `snn_lang_stats_t` field
additions, force-rebuild dependent tests (`rm` the .o + cmake build) —
the dep tracker doesn't always catch header layout changes and the
tests will stack-smash at shutdown rather than report a clean fail.
