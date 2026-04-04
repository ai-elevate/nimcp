/**
 * @file nimcp_probe_builtins.c
 * @brief Built-in sampler functions that wrap existing metrics structs
 *
 * Each sampler reads from one of the brain's inline metric structs
 * (network_metrics, cognitive_stats, training_dashboard, stats) and
 * populates probe_metric_t key-value pairs.
 */

#include "core/probes/nimcp_brain_probes.h"
#include "core/brain/nimcp_brain_internal.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper: set metric values (avoids repetitive boilerplate)
 * ============================================================================ */

static inline void set_float(probe_metric_t* m, const char* key, float val, uint64_t ts) {
    strncpy(m->key, key, PROBE_KEY_LEN - 1);
    m->key[PROBE_KEY_LEN - 1] = '\0';
    m->type = PROBE_METRIC_FLOAT;
    m->value.f = val;
    m->timestamp_us = ts;
}

static inline void set_int(probe_metric_t* m, const char* key, int64_t val, uint64_t ts) {
    strncpy(m->key, key, PROBE_KEY_LEN - 1);
    m->key[PROBE_KEY_LEN - 1] = '\0';
    m->type = PROBE_METRIC_INT;
    m->value.i = val;
    m->timestamp_us = ts;
}

static inline void set_string(probe_metric_t* m, const char* key, const char* val, uint64_t ts) {
    strncpy(m->key, key, PROBE_KEY_LEN - 1);
    m->key[PROBE_KEY_LEN - 1] = '\0';
    m->type = PROBE_METRIC_STRING;
    strncpy(m->value.s, val ? val : "", PROBE_STRING_LEN - 1);
    m->value.s[PROBE_STRING_LEN - 1] = '\0';
    m->timestamp_us = ts;
}

/* ============================================================================
 * Network Metrics Sampler
 * ============================================================================ */

static void sample_network_metrics(
    void* module_ptr, uint16_t module_id, brain_t brain,
    probe_metric_t* metrics, uint32_t* count, void* user_data)
{
    (void)module_ptr; (void)module_id; (void)user_data;
    if (!brain || !metrics || !count) return;
    uint32_t max = *count;
    uint32_t n = 0;
    extern uint64_t nimcp_time_get_us(void);
    uint64_t ts = nimcp_time_get_us();

    #define EMIT_F(key, val) do { if (n < max) set_float(&metrics[n++], key, val, ts); } while(0)
    #define EMIT_I(key, val) do { if (n < max) set_int(&metrics[n++], key, val, ts); } while(0)

    EMIT_F("ann.ema_loss",   brain->network_metrics.ema_ann_loss);
    EMIT_F("ann.last_loss",  brain->network_metrics.last_ann_loss);
    EMIT_I("ann.steps",      (int64_t)brain->network_metrics.ann_steps);
    EMIT_F("cnn.ema_loss",   brain->network_metrics.ema_cnn_loss);
    EMIT_I("cnn.steps",      (int64_t)brain->network_metrics.cnn_steps);
    EMIT_F("snn.ema_loss",   brain->network_metrics.ema_snn_loss);
    EMIT_F("snn.last_loss",  brain->network_metrics.last_snn_loss);
    EMIT_I("snn.steps",      (int64_t)brain->network_metrics.snn_steps);
    EMIT_F("lnn.ema_loss",   brain->network_metrics.ema_lnn_loss);
    EMIT_I("lnn.steps",      (int64_t)brain->network_metrics.lnn_steps);
    EMIT_F("hnn.energy",     brain->network_metrics.hnn_energy);
    EMIT_F("hnn.deviation",  brain->network_metrics.hnn_energy_deviation);
    EMIT_I("hnn.active",     brain->network_metrics.hnn_active ? 1 : 0);
    EMIT_F("fno.ema_loss",   brain->network_metrics.fno_audio_ema_loss);
    EMIT_I("fno.steps",      (int64_t)brain->network_metrics.fno_audio_steps);
    EMIT_I("fno.params",     (int64_t)brain->network_metrics.fno_audio_params);

    #undef EMIT_F
    #undef EMIT_I

    *count = n;
}

probe_handle_t probe_attach_network_metrics(probe_registry_t* reg, uint32_t interval_ms) {
    uint16_t modules[] = {0x0100};  /* BIO_MODULE_BRAIN */
    return probe_create_active(reg, "network_metrics", modules, 1,
                                sample_network_metrics, NULL, interval_ms);
}

/* ============================================================================
 * Cognitive Stats Sampler
 * ============================================================================ */

static void sample_cognitive_stats(
    void* module_ptr, uint16_t module_id, brain_t brain,
    probe_metric_t* metrics, uint32_t* count, void* user_data)
{
    (void)module_ptr; (void)module_id; (void)user_data;
    if (!brain || !metrics || !count) return;
    uint32_t max = *count;
    uint32_t n = 0;
    extern uint64_t nimcp_time_get_us(void);
    uint64_t ts = nimcp_time_get_us();

    #define EMIT_F(key, val) do { if (n < max) set_float(&metrics[n++], key, val, ts); } while(0)
    #define EMIT_I(key, val) do { if (n < max) set_int(&metrics[n++], key, val, ts); } while(0)

    EMIT_I("grounded_lang.steps", (int64_t)brain->cognitive_stats.grounded_lang_steps);
    EMIT_I("knowledge.steps",     (int64_t)brain->cognitive_stats.knowledge_steps);
    EMIT_I("vae.steps",           (int64_t)brain->cognitive_stats.vae_steps);
    EMIT_F("vae.last_loss",       brain->cognitive_stats.vae_last_loss);
    EMIT_I("fep_parietal.steps",  (int64_t)brain->cognitive_stats.fep_parietal_steps);
    EMIT_I("physics_nn.steps",    (int64_t)brain->cognitive_stats.physics_nn_steps);
    EMIT_I("pred_hierarchy.steps",(int64_t)brain->cognitive_stats.pred_hierarchy_steps);
    EMIT_F("pred_hierarchy.loss", brain->cognitive_stats.pred_hierarchy_last_loss);
    EMIT_I("jepa.steps",          (int64_t)brain->cognitive_stats.jepa_steps);
    EMIT_F("jepa.last_loss",      brain->cognitive_stats.jepa_last_loss);
    EMIT_I("creative.steps",      (int64_t)brain->cognitive_stats.creative_steps);
    EMIT_I("self_heal.steps",     (int64_t)brain->cognitive_stats.self_heal_steps);
    EMIT_I("intuition.steps",     (int64_t)brain->cognitive_stats.intuition_steps);
    EMIT_I("fep_orch.steps",      (int64_t)brain->cognitive_stats.fep_orchestrator_steps);

    #undef EMIT_F
    #undef EMIT_I

    *count = n;
}

probe_handle_t probe_attach_cognitive_stats(probe_registry_t* reg, uint32_t interval_ms) {
    uint16_t modules[] = {0x0100};
    return probe_create_active(reg, "cognitive_stats", modules, 1,
                                sample_cognitive_stats, NULL, interval_ms);
}

/* ============================================================================
 * Training Dashboard Sampler
 * ============================================================================ */

static void sample_training_dashboard(
    void* module_ptr, uint16_t module_id, brain_t brain,
    probe_metric_t* metrics, uint32_t* count, void* user_data)
{
    (void)module_ptr; (void)module_id; (void)user_data;
    if (!brain || !metrics || !count) return;
    uint32_t max = *count;
    uint32_t n = 0;
    extern uint64_t nimcp_time_get_us(void);
    uint64_t ts = nimcp_time_get_us();

    #define EMIT_F(key, val) do { if (n < max) set_float(&metrics[n++], key, val, ts); } while(0)
    #define EMIT_I(key, val) do { if (n < max) set_int(&metrics[n++], key, val, ts); } while(0)
    #define EMIT_S(key, val) do { if (n < max) set_string(&metrics[n++], key, val, ts); } while(0)

    EMIT_I("stage",            (int64_t)brain->training_dashboard.current_stage);
    EMIT_I("step",             (int64_t)brain->training_dashboard.current_step);
    EMIT_S("domain",           brain->training_dashboard.current_domain);
    EMIT_F("fact_ratio",       brain->training_dashboard.fact_ratio);
    EMIT_I("warm_start_done",  brain->training_dashboard.warm_start_complete ? 1 : 0);
    EMIT_I("warm_start_step",  (int64_t)brain->training_dashboard.warm_start_step);
    EMIT_F("lr_physics",       brain->training_dashboard.lr_physics);
    EMIT_F("lr_chemistry",     brain->training_dashboard.lr_chemistry);
    EMIT_F("lr_biology",       brain->training_dashboard.lr_biology);
    EMIT_I("wm_steps",         (int64_t)brain->training_dashboard.wm_steps);
    EMIT_I("wm_phys",          (int64_t)brain->training_dashboard.wm_physics_transitions);
    EMIT_I("wm_chem",          (int64_t)brain->training_dashboard.wm_chemistry_transitions);
    EMIT_I("wm_bio",           (int64_t)brain->training_dashboard.wm_biology_transitions);
    EMIT_I("collapse_events",  (int64_t)brain->training_dashboard.collapse_events);
    EMIT_I("surprises",        (int64_t)brain->training_dashboard.surprises_stored);
    EMIT_I("replays",          (int64_t)brain->training_dashboard.replays_done);
    EMIT_I("vocab_size",       (int64_t)brain->training_dashboard.vocab_size);
    EMIT_F("lang_confidence",  brain->training_dashboard.language_confidence);
    EMIT_I("active_engines",   (int64_t)brain->training_dashboard.active_engines);

    #undef EMIT_F
    #undef EMIT_I
    #undef EMIT_S

    *count = n;
}

probe_handle_t probe_attach_training_dashboard(probe_registry_t* reg, uint32_t interval_ms) {
    uint16_t modules[] = {0x0100};
    return probe_create_active(reg, "training_dashboard", modules, 1,
                                sample_training_dashboard, NULL, interval_ms);
}

/* ============================================================================
 * Inference Stats Sampler
 * ============================================================================ */

static void sample_inference(
    void* module_ptr, uint16_t module_id, brain_t brain,
    probe_metric_t* metrics, uint32_t* count, void* user_data)
{
    (void)module_ptr; (void)module_id; (void)user_data;
    if (!brain || !metrics || !count) return;
    uint32_t max = *count;
    uint32_t n = 0;
    extern uint64_t nimcp_time_get_us(void);
    uint64_t ts = nimcp_time_get_us();

    #define EMIT_F(key, val) do { if (n < max) set_float(&metrics[n++], key, val, ts); } while(0)
    #define EMIT_I(key, val) do { if (n < max) set_int(&metrics[n++], key, val, ts); } while(0)

    EMIT_I("total_inferences",   (int64_t)brain->stats.total_inferences);
    EMIT_I("total_learn_steps",  (int64_t)brain->stats.total_learning_steps);
    EMIT_F("avg_inference_us",   brain->stats.avg_inference_time_us);
    EMIT_F("current_lr",         brain->stats.current_learning_rate);
    EMIT_F("avg_sparsity",       brain->stats.avg_sparsity);
    EMIT_F("accuracy",           brain->stats.accuracy);
    EMIT_F("running_accuracy",   brain->stats.running_accuracy);
    EMIT_I("num_neurons",        (int64_t)brain->stats.num_neurons);
    EMIT_I("num_synapses",       (int64_t)brain->stats.num_synapses);
    EMIT_I("memory_bytes",       (int64_t)brain->stats.memory_bytes);
    EMIT_F("novelty_score",      brain->last_novelty_score);
    EMIT_F("curiosity_drive",    brain->last_curiosity_drive);
    EMIT_I("recurrent_iters",    (int64_t)brain->recurrent_iteration_count);

    #undef EMIT_F
    #undef EMIT_I

    *count = n;
}

probe_handle_t probe_attach_inference(probe_registry_t* reg, uint32_t interval_ms) {
    uint16_t modules[] = {0x0100};
    return probe_create_active(reg, "inference", modules, 1,
                                sample_inference, NULL, interval_ms);
}
