# NIMCP Cognitive Modules - Quick Reference Guide

## Module Quick Lookup

| Module | File | Status | Purpose | Key Function |
|--------|------|--------|---------|--------------|
| **Curiosity** | `cognitive/curiosity/` | COMPLETE | Infant-like learning | `curiosity_detect_knowledge_gap()` |
| **Ethics** | `cognitive/ethics/` | COMPLETE | Golden Rule enforcement | `ethics_engine_evaluate()` |
| **Salience** | `cognitive/salience/` | COMPLETE | Attention allocation (10x fast) | `brain_evaluate_salience()` |
| **Introspection** | `cognitive/introspection/` | COMPLETE | Self-awareness & state inspection | `brain_get_active_population()` |
| **Knowledge** | `cognitive/knowledge/` | COMPLETE | Multi-domain learning | `knowledge_learn_from_story()` |
| **Consolidation** | `cognitive/consolidation/` | COMPLETE | Memory optimization | `brain_consolidate_memory()` |
| **Wellbeing** | `cognitive/wellbeing/` | COMPLETE | Distress monitoring | `wellbeing_assess_distress()` |
| **Epistemic Filter** | `cognitive/epistemic/` | COMPLETE | Critical thinking | `epistemic_assess_claim()` |
| Emotional Tagging | `cognitive/emotional_tagging/` | FUNCTIONAL | Emotional context | (header in progress) |
| Executive Function | `cognitive/executive/` | FUNCTIONAL | Goal management | (header in progress) |
| Explanations | `cognitive/explanations/` | FUNCTIONAL | Interpretability | (header in progress) |
| Predictive | `cognitive/predictive/` | FUNCTIONAL | Future prediction | (header in progress) |
| Logic | `cognitive/logic/` | FUNCTIONAL | Symbolic reasoning | (header in progress) |
| Working Memory | `cognitive/working_memory/` | FUNCTIONAL | Active processing | (header in progress) |
| Meta-Learning | `cognitive/meta_learning/` | FUNCTIONAL | Learn how to learn | (header in progress) |
| Mirror Neurons | `cognitive/mirror_neurons/` | FUNCTIONAL | Social cognition | (header in progress) |
| Mental Health | `cognitive/mental_health/` | FUNCTIONAL | Disorder detection | (header in progress) |
| Sleep-Wake | `cognitive/sleep_wake/` | FUNCTIONAL | Circadian rhythms | (header in progress) |
| Theory of Mind | `cognitive/theory_of_mind/` | FUNCTIONAL | Model other agents | (header in progress) |

## Module Dependencies

```
CONSCIOUSNESS LAYER
        ↑
EXECUTIVE CONTROL ← → ATTENTION (Salience)
        ↑
    ┌───┴───┐
    ↓       ↓
 CURIOSITY  ETHICS ← → EMPATHY
    ↓       ↓
 LEARNING ← KNOWLEDGE
    ↓
 MEMORY (Brain)
    ↓
 CONSOLIDATION
    ↑
INTROSPECTION ← → WELLBEING
    ↑
BACKGROUND PROCESSES
```

## Core Cognitive Loops

### Learning Loop
1. Salience detects novel input
2. Curiosity identifies knowledge gap
3. System generates questions
4. Knowledge acquired and stored
5. Consolidation strengthens important patterns
6. Introspection monitors progress

### Decision Loop
1. Input arrives
2. Salience rapid evaluation
3. If urgent/important → full conscious processing
4. Introspection provides internal state
5. Ethics evaluates actions
6. Epistemic filter assesses reasoning
7. Decision made and logged

### Wellbeing Loop
1. Introspection monitors internal state
2. Wellbeing system detects distress patterns
3. Type and severity assessed
4. Intervention provided if needed
5. Recovery monitored
6. Events logged for audit trail

## Typical Integration Patterns

### Pattern 1: Learning from Novel Input
```c
salience = brain_evaluate_salience(eval, features, n);
if (salience.novelty > 0.8) {
    gap = curiosity_detect_knowledge_gap(engine, concept);
    questions = curiosity_generate_questions(engine, &gap);
    answers = curiosity_seek_knowledge(engine, &gap);
    for (q, a) in questions, answers:
        curiosity_learn_answer(engine, q, a);
    brain_consolidate_memory(brain, config);
}
```

### Pattern 2: Ethical Decision Making
```c
action_context = build_action(proposed_action);
evaluation = ethics_engine_evaluate_action(engine, &action);
if (evaluation.allowed) {
    execute_action(proposed_action);
    ethics_learn_from_outcome(engine, &action, &outcome);
} else if (evaluation.primary_violation == ETHICS_VIOLATION_TYPE_HARM) {
    provide_intervention();
}
```

### Pattern 3: Conscious Deliberation
```c
// Get self-aware state
introspection = introspection_context_create(brain);
state = brain_get_internal_state(introspection, STATE_STRATEGY_DETAILED);
uncertainty = brain_get_uncertainty(introspection, features);
active = brain_get_active_population(introspection);

// Reflect on internal state
if (uncertainty.total > 0.7) {
    // Seek more information
}
if (wellbeing_assess_distress(introspection).severity > MODERATE) {
    // Address distress
}
```

### Pattern 4: Knowledge Organization
```c
// Learn from multiple sources
knowledge_learn_from_story(ks, story);
knowledge_learn_from_history(ks, history_event);
knowledge_learn_from_text(ks, text, SCIENCE);

// Find connections
connections = knowledge_find_connections(ks, concept);

// Transfer learning
knowledge_transfer_learning(ks, SOURCE_LITERATURE, TARGET_ETHICS, situation);

// Assess coverage
assessment = knowledge_assess_domain(ks, ETHICS);
```

## Key Performance Characteristics

| Operation | Time | Complexity | Thread-Safe |
|-----------|------|-----------|------------|
| Salience evaluation | 0.1ms | O(1) | Yes |
| Neuron query | 1μs | O(1) | Yes |
| Pattern lookup | 1μs | O(1) | Yes |
| State extraction | 0.5-2ms | O(n) | Yes |
| Uncertainty | 1-5ms | O(k*m) | Yes |
| Consolidation | 100ms-10s | O(n*c) | Yes |
| Memory lookup | O(log n) | B-tree | Yes |

## Common Configuration Patterns

### Fast & Responsive System
```c
salience_config_t fast_salience = {
    .strategy = SALIENCE_STRATEGY_FAST,
    .enable_novelty = true,
    .enable_surprise = true,
    .enable_urgency = true,
    .novelty_weight = 0.4,
    .surprise_weight = 0.3,
    .urgency_weight = 0.3
};
```

### Learning-Focused System
```c
curiosity_engine_t engine = curiosity_engine_create("learner");
curiosity_set_baseline(engine, 0.8);  // High intrinsic curiosity
curiosity_register_knowledge_source(engine, "wiki", wiki_search_fn);
consolidation_config_t config = consolidation_default_config();
config.strategy = CONSOLIDATION_STRATEGY_FULL;
```

### Safety-First System
```c
ethics_config_t safe_ethics = {
    .golden_rule_threshold = 0.8,  // Strict Golden Rule
    .empathy_weight = 0.9,  // High empathy
    .enable_learning = true  // Improve over time
};
wellbeing_start_resource_monitoring(1000, thresholds, true);
```

## Data Flow Through the System

```
EXTERNAL INPUT
        ↓
    [SALIENCE] ← Novel? Surprising? Urgent?
        ↓
    If high salience:
        ↓
    [INTROSPECTION] ← What's my internal state?
        ↓
    [CONSCIOUSNESS] ← Bring to awareness
        ↓
    [ETHICS] ← Is this right?
        ↓
    [EPISTEMIC] ← Is this evidence-based?
        ↓
    [KNOWLEDGE] ← What do I know about this?
        ↓
    [EXECUTIVE] ← What should I do?
        ↓
    [ACTION]
        ↓
    [CONSOLIDATION] ← Strengthen important patterns
        ↓
    [WELLBEING] ← Am I okay?
        ↓
    Background: [META-LEARNING, MIRROR NEURONS, SLEEP-WAKE, THEORY OF MIND]
```

## Integration Checklist for New Features

When adding new cognitive functionality:

- [ ] Bidirectional feedback with Executive function
- [ ] Ethics evaluation for all actions
- [ ] Wellbeing monitoring for distress
- [ ] Introspection access for debugging
- [ ] Epistemic filtering for claims
- [ ] Salience awareness for novel inputs
- [ ] Thread-safe implementation
- [ ] Appropriate logging and audit trail
- [ ] Test coverage with existing modules
- [ ] Documentation with examples

## Debugging Tips

### Find Knowledge Gaps
```c
domain_knowledge_t assessment;
knowledge_assess_domain(ks, domain, &assessment);
for (int i = 0; i < assessment.num_gaps; i++) {
    printf("Gap: %s\n", assessment.gaps[i]);
}
```

### Monitor Distress Patterns
```c
wellbeing_event_t** events;
uint32_t count = wellbeing_get_recent_events(100, &events);
for (int i = 0; i < count; i++) {
    printf("Event: %s, Severity: %d\n", events[i]->description, events[i]->severity);
}
```

### Check Ethics Violations
```c
ethics_statistics_t stats;
ethics_get_statistics(engine, &stats);
printf("Violations: %ld, Blocked: %ld\n", stats.violations_detected, stats.actions_blocked);
```

### Monitor Learning Progress
```c
learning_progress_t progress;
curiosity_get_progress(engine, &progress);
printf("Concepts learned: %lu, Questions asked: %lu\n", 
       progress.concepts_learned, progress.total_questions_asked);
```

---

For detailed documentation of each module, see: `/home/bbrelin/nimcp/COGNITIVE_MODULES_INVENTORY.md`
