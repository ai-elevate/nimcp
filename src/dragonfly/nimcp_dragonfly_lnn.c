/**
 * @file nimcp_dragonfly_lnn.c
 * @brief Sidecar LNN reservoir for dragonfly temporal smoothing (Phase 4i).
 *
 * See include/dragonfly/nimcp_dragonfly_lnn.h for API + rationale.
 */
#include "dragonfly/nimcp_dragonfly_lnn.h"

#include "lnn/nimcp_lnn_network.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/error/nimcp_error_codes.h"

#include <math.h>
#include <string.h>

struct dragonfly_lnn_s {
    dragonfly_system_t* df;       /* non-owning */
    lnn_network_t*      net;
    nimcp_tensor_t*     input;
    nimcp_tensor_t*     output;
    float               last_output[DRAGONFLY_LNN_DIM];
    uint64_t            step_count;
    float               default_dt_ms;
};

dragonfly_lnn_t* dragonfly_lnn_create(dragonfly_system_t* df) {
    if (!df) return NULL;

    dragonfly_lnn_t* dl = (dragonfly_lnn_t*)nimcp_calloc(
        1, sizeof(dragonfly_lnn_t));
    if (!dl) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
                              "dragonfly_lnn_create: handle alloc failed");
        return NULL;
    }
    dl->df            = df;
    dl->default_dt_ms = 1.0f;

    /* Small NCP reservoir: 16 → 8 → 8 → 16. Enough to exhibit LTC dynamics
     * without over-parameterizing for what is a 16-dim smoothing task. */
    dl->net = lnn_network_create_ncp(DRAGONFLY_LNN_DIM, 8, 8,
                                      DRAGONFLY_LNN_DIM);
    if (!dl->net) {
        NIMCP_LOGGING_WARN("dragonfly_lnn_create: NCP alloc failed; "
                           "dragonfly will run without LNN reservoir");
        nimcp_free(dl);
        return NULL;
    }
    /* Fixed seed for reproducibility; the reservoir is untrained so seed
     * choice is cosmetic — just avoid the default time-based seed so two
     * brains built from the same config produce comparable smoothing. */
    if (lnn_network_init_weights(dl->net, 0xD14F04u) != 0) {
        lnn_network_destroy(dl->net);
        nimcp_free(dl);
        return NULL;
    }

    uint32_t dims[1] = { DRAGONFLY_LNN_DIM };
    dl->input  = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    dl->output = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!dl->input || !dl->output) {
        if (dl->input)  nimcp_tensor_destroy(dl->input);
        if (dl->output) nimcp_tensor_destroy(dl->output);
        lnn_network_destroy(dl->net);
        nimcp_free(dl);
        return NULL;
    }

    NIMCP_LOGGING_INFO("dragonfly_lnn_create: 16→8→8→16 reservoir live");
    return dl;
}

void dragonfly_lnn_destroy(dragonfly_lnn_t* dl) {
    if (!dl) return;
    if (dl->input)  nimcp_tensor_destroy(dl->input);
    if (dl->output) nimcp_tensor_destroy(dl->output);
    if (dl->net)    lnn_network_destroy(dl->net);
    nimcp_free(dl);
}

/* Pack current dragonfly state into the LNN's 16-dim input tensor. Returns
 * true on success. Unpopulated slots are zeroed — e.g., if no primary
 * target exists, position/velocity channels stay at 0 and the LNN state
 * naturally relaxes back toward baseline. */
static bool _pack_input(dragonfly_lnn_t* dl, float* in) {
    memset(in, 0, sizeof(float) * DRAGONFLY_LNN_DIM);

    /* Primary target position/velocity/predicted. */
    dragonfly_target_info_t t = {0};
    if (dragonfly_get_primary_target(dl->df, &t) == 0) {
        /* Positions/velocities can be unbounded — tanh-squash so the
         * LNN sees stable [-1,1] inputs regardless of world scale. */
        in[0]  = tanhf(t.position[0]);
        in[1]  = tanhf(t.position[1]);
        in[2]  = tanhf(t.position[2]);
        in[3]  = tanhf(t.velocity[0]);
        in[4]  = tanhf(t.velocity[1]);
        in[5]  = tanhf(t.velocity[2]);
        in[6]  = tanhf(t.predicted_position[0]);
        in[7]  = tanhf(t.predicted_position[1]);
        in[8]  = tanhf(t.predicted_position[2]);
        in[9]  = t.confidence  < 0.0f ? 0.0f :
                 t.confidence  > 1.0f ? 1.0f : t.confidence;
        in[10] = t.threat_level < 0.0f ? 0.0f :
                 t.threat_level > 1.0f ? 1.0f : t.threat_level;
    }

    /* TSDN vector. */
    tsdn_vector_t tv = {0};
    if (dragonfly_get_tsdn_vector(dl->df, &tv) == 0 && tv.valid) {
        in[11] = sinf(tv.direction);
        in[12] = cosf(tv.direction);
        in[13] = tv.magnitude < 0.0f ? 0.0f :
                 tv.magnitude > 1.0f ? 1.0f : tv.magnitude;
        in[14] = tanhf(tv.angular_velocity);
    }

    /* Current mode, normalized to [0,1]. DRAGONFLY_MODE_INTERCEPTING = 4. */
    int mi = (int)dragonfly_get_mode(dl->df);
    if (mi < 0) mi = 0;
    if (mi > 4) mi = 4;
    in[15] = (float)mi / 4.0f;

    return true;
}

int dragonfly_lnn_step(dragonfly_lnn_t* dl, float dt_ms) {
    if (!dl || !dl->net || !dl->input || !dl->output) return -1;

    float* in_data = (float*)nimcp_tensor_data(dl->input);
    if (!in_data) return -1;
    if (!_pack_input(dl, in_data)) return -1;

    float use_dt = dt_ms > 0.0f ? dt_ms : dl->default_dt_ms;
    if (lnn_network_forward_step(dl->net, dl->input,
                                  dl->output, use_dt) != 0) {
        return -1;
    }

    const float* out_data =
        (const float*)nimcp_tensor_data_const(dl->output);
    if (!out_data) return -1;

    /* Snapshot the output into last_output[] so callers can read at any
     * cadence without holding the tensor pointer. tanh-squash for bounded
     * downstream use. */
    for (uint32_t i = 0; i < DRAGONFLY_LNN_DIM; i++) {
        dl->last_output[i] = tanhf(out_data[i]);
    }
    dl->step_count++;
    return 0;
}

uint32_t dragonfly_lnn_get_output(const dragonfly_lnn_t* dl,
                                   float* out,
                                   uint32_t max_dim) {
    if (!dl || !out || max_dim == 0) return 0;
    uint32_t n = max_dim < DRAGONFLY_LNN_DIM ? max_dim : DRAGONFLY_LNN_DIM;
    memcpy(out, dl->last_output, sizeof(float) * n);
    return n;
}

uint64_t dragonfly_lnn_get_step_count(const dragonfly_lnn_t* dl) {
    return dl ? dl->step_count : 0;
}
