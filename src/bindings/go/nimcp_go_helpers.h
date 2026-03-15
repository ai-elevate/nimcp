// nimcp_go_helpers.h — C helper functions for Go CGo bindings
// These wrap internal-brain-access patterns that CGo cannot call directly.
#ifndef NIMCP_GO_HELPERS_H
#define NIMCP_GO_HELPERS_H

#include "nimcp.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Brain State Accessors ----
float go_brain_medulla_get_arousal(nimcp_brain_t brain);
float go_brain_sleep_get_pressure(nimcp_brain_t brain);
float go_brain_bg_get_dopamine(nimcp_brain_t brain);
const char* go_brain_substrate_get_health(nimcp_brain_t brain);

// ---- Configuration ----
bool go_brain_set_fast_training(nimcp_brain_t brain, bool enabled);
bool go_brain_set_task_type(nimcp_brain_t brain, const char* task_str);
bool go_brain_enable_biological_plasticity(nimcp_brain_t brain, bool enabled);
int  go_brain_enable_multi_network(nimcp_brain_t brain);

// ---- LNN ----
bool go_brain_lnn_create(nimcp_brain_t brain,
                         uint32_t n_sensory, uint32_t n_inter,
                         uint32_t n_command, uint32_t n_output);

// LNN stats output struct (flat for CGo)
typedef struct {
    uint64_t forward_steps;
    uint64_t backward_steps;
    uint64_t ode_evaluations;
    float    avg_tau;
    float    state_norm;
    float    gradient_norm;
    uint32_t nan_count;
    uint32_t inf_count;
    bool     valid;
} go_lnn_stats_t;
bool go_brain_lnn_get_stats(nimcp_brain_t brain, go_lnn_stats_t* out);

// SNN stats output struct
typedef struct {
    uint64_t total_steps;
    uint64_t total_spikes;
    float    mean_firing_rate;
    float    max_firing_rate;
    float    sparsity;
    float    synchrony;
    float    spikes_per_sample;
    uint32_t silent_neurons;
    uint32_t hyperactive_neurons;
    int      health;
    uint64_t memory_usage_bytes;
    bool     valid;
} go_snn_stats_t;
bool go_brain_snn_get_stats(nimcp_brain_t brain, go_snn_stats_t* out);

// CNN stats output struct
typedef struct {
    uint32_t num_layers;
    uint64_t num_parameters;
    uint32_t num_labels;
    bool     active;
    bool     valid;
} go_cnn_stats_t;
bool go_brain_cnn_get_stats(nimcp_brain_t brain, go_cnn_stats_t* out);

// ---- Sensory ----
// Submit sensory data for a modality
// modality: "visual", "audio", "speech", "somatosensory"
// For visual: width/height/channels used; for somato: n_segments used
nimcp_status_t go_brain_submit_sensory(nimcp_brain_t brain,
                                        const char* modality,
                                        const float* data, uint32_t data_len,
                                        uint32_t width, uint32_t height,
                                        uint32_t channels, uint32_t n_segments);

// Visual cortex process: returns feature dim written, 0 on error
uint32_t go_brain_visual_cortex_process(nimcp_brain_t brain,
                                         const float* pixels, uint32_t n_pixels,
                                         uint32_t width, uint32_t height,
                                         uint32_t channels,
                                         float* features_out, uint32_t max_features);

// ---- Cortex CNN metrics ----
typedef struct {
    float    last_loss;
    float    ema_loss;
    uint64_t forward_steps;
    uint64_t backward_steps;
    float    embedding_norm;
    float    confidence;
    uint32_t embedding_dim;
    uint32_t num_params;
    bool     valid;
} go_cortex_cnn_metrics_t;

// ci: 0=visual, 1=audio, 2=speech, 3=somato
bool go_brain_get_cortex_cnn_metrics(nimcp_brain_t brain, int ci,
                                      go_cortex_cnn_metrics_t* out);

// ---- Focus Attention ----
bool go_brain_focus_attention(nimcp_brain_t brain, const char* modality);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_GO_HELPERS_H
