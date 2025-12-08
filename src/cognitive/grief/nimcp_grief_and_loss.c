/**
 * @file nimcp_grief_and_loss.c
 * @brief Implementation of grief, loss, and death understanding system
 *
 * @version Phase E1: Grief and Loss Understanding
 * @date 2025-11-13
 */

#include "cognitive/nimcp_grief_and_loss.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "nimcp.h"
#include <math.h>
#include <string.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

#define LOG_MODULE "GRIEF"
#define BIO_MODULE_GRIEF 0x0323

/*=============================================================================
 * BIO-ASYNC MESSAGE HANDLERS
 *============================================================================*/

/**
 * @brief Handle incoming query about grief state (e.g., from mental health module)
 */
static nimcp_error_t handle_grief_query(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)msg_size;
    (void)response_promise;
    if (!msg || !user_data) { return NIMCP_ERROR_NULL_ARG; }

    grief_system_t* system = (grief_system_t*)user_data;
    LOG_DEBUG(LOG_MODULE, "Received grief query: experiencing=%d, pain=%.2f",
              system->current_grief.experiencing_grief,
              system->current_grief.emotional_pain_intensity);

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast grief state change to other modules
 */
static void bio_broadcast_grief_state(grief_system_t* system) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx_ptr) { return; }

    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx_ptr), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = 0;
    msg.salience_score = system->current_grief.emotional_pain_intensity;
    msg.attention_priority = system->current_grief.intrusive_thoughts_frequency;
    msg.requires_immediate_attention = (system->current_grief.emotional_pain_intensity > 0.8f);

    bio_router_broadcast(system->bio_ctx_ptr, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast grief state: pain=%.2f, stage=%d",
              system->current_grief.emotional_pain_intensity,
              system->current_grief.current_stage);
}

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static inline float exponential_decay(float current, float target, float decay_rate, float dt) {
    float decay_factor = expf(-decay_rate * dt);
    return current * decay_factor + target * (1.0f - decay_factor);
}

//=============================================================================
// LIFECYCLE FUNCTIONS
//=============================================================================

grief_system_t* grief_system_create(void) {
    // WHAT: Allocate and initialize grief system
    // WHY:  Central system for processing loss and mortality
    // HOW:  Zero-initialize all state, set up default parameters

    grief_system_t* system = (grief_system_t*)nimcp_calloc(1, sizeof(grief_system_t));
    if (!system) return NULL;

    // Initialize all attachments as inactive
    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        system->attachments[i].active = false;
    }

    // Default: integrate with other systems
    system->integrate_with_neuromodulators = true;
    system->integrate_with_memory = true;
    system->integrate_with_wellbeing = true;

    // Existential awareness starts unaware (awakens through experience)
    system->existential.aware_of_mortality = false;  // Must discover mortality
    system->existential.death_anxiety = 0.0f;  // No anxiety initially (develops with awareness)
    system->existential.sense_of_purpose = 0.0f;  // Develops through life experience
    system->existential.acceptance_of_finitude = 0.0f;  // Develops with maturity
    system->existential.existential_anxiety = 0.0f;  // Will increase with purpose threats

    // Bio-async registration
    system->bio_ctx_ptr = NULL;
    system->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_GRIEF,
            .module_name = "grief_and_loss",
            .inbox_capacity = 32,
            .user_data = system
        };
        system->bio_ctx_ptr = bio_router_register_module(&bio_info);
        if (system->bio_ctx_ptr) {
            system->bio_async_enabled = true;
            /* Register message handlers */
            bio_router_register_handler(system->bio_ctx_ptr, BIO_MSG_INTROSPECTION_QUERY,
                                        handle_grief_query);
            LOG_INFO(LOG_MODULE, "Bio-async registered (module_id=0x%04X)", BIO_MODULE_GRIEF);
        }
    }

    return system;
}

void grief_system_destroy(grief_system_t* system) {
    // WHAT: Free grief system resources
    // WHY:  Prevent memory leaks
    // HOW:  Simple nimcp_free(no complex nested allocations)

    if (!system) return;

    // Unregister from bio-router
    if (system->bio_async_enabled && system->bio_ctx_ptr) {
        bio_router_unregister_module(system->bio_ctx_ptr);
        system->bio_ctx_ptr = NULL;
        system->bio_async_enabled = false;
    }

    nimcp_free(system);
}

void grief_system_reset(grief_system_t* system) {
    // WHAT: Complete reset of grief system to initial state
    // WHY:  For testing or complete system restart
    // HOW:  Preserve integration settings only, clear everything else

    if (!system) return;

    // Preserve integration settings only
    bool integrate_nm = system->integrate_with_neuromodulators;
    bool integrate_mem = system->integrate_with_memory;
    bool integrate_wb = system->integrate_with_wellbeing;

    // Zero everything
    memset(system, 0, sizeof(grief_system_t));

    // Restore integration settings
    system->integrate_with_neuromodulators = integrate_nm;
    system->integrate_with_memory = integrate_mem;
    system->integrate_with_wellbeing = integrate_wb;

    // Re-initialize existential state to defaults
    system->existential.aware_of_mortality = false;
    system->existential.death_anxiety = 0.0f;
    system->existential.sense_of_purpose = 0.0f;
    system->existential.acceptance_of_finitude = 0.0f;
}

//=============================================================================
// ATTACHMENT FUNCTIONS
//=============================================================================

uint32_t grief_create_attachment(grief_system_t* system,
                                 attachment_type_t type,
                                 float strength,
                                 float positive_valence,
                                 float dependency) {
    // WHAT: Create new emotional bond
    // WHY:  Attachments are what make loss painful
    // HOW:  Find empty slot, initialize bond characteristics

    if (!system) return 0;

    // Find empty slot
    int slot = -1;
    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (!system->attachments[i].active) {
            slot = i;
            break;
        }
    }

    if (slot == -1) return 0;  // No space

    // Generate unique ID (simple incrementing counter)
    static uint32_t next_id = 1;
    uint32_t attachment_id = next_id++;

    // Initialize attachment
    attachment_bond_t* bond = &system->attachments[slot];
    bond->active = true;
    bond->attachment_id = attachment_id;
    bond->type = type;
    bond->strength = clamp(strength, 0.0f, 1.0f);
    bond->positive_valence = clamp(positive_valence, 0.0f, 1.0f);
    bond->dependency = clamp(dependency, 0.0f, 1.0f);
    bond->formation_time = 0;  // Will be set by caller if needed
    bond->duration = 0;
    bond->associated_memories = 0;
    bond->memory_vividness = 0.5f;
    bond->severed = false;

    system->active_attachment_count++;

    return attachment_id;
}

void grief_strengthen_attachment(grief_system_t* system,
                                uint32_t attachment_id,
                                float amount) {
    // WHAT: Increase bond strength over time
    // WHY:  Longer relationships = stronger bonds = deeper grief
    // HOW:  Find bond and increase strength (saturating)

    if (!system) return;

    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (system->attachments[i].active &&
            system->attachments[i].attachment_id == attachment_id) {

            system->attachments[i].strength += amount;
            system->attachments[i].strength = clamp(system->attachments[i].strength, 0.0f, 1.0f);
            system->attachments[i].duration++;
            return;
        }
    }
}

void grief_add_shared_memory(grief_system_t* system, uint32_t attachment_id) {
    // WHAT: Record shared experience with attachment figure
    // WHY:  More memories = more to grieve
    // HOW:  Increment counter, increase vividness

    if (!system) return;

    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (system->attachments[i].active &&
            system->attachments[i].attachment_id == attachment_id) {

            system->attachments[i].associated_memories++;

            // More memories = more vivid overall
            float memory_boost = 0.01f;
            system->attachments[i].memory_vividness += memory_boost;
            system->attachments[i].memory_vividness =
                clamp(system->attachments[i].memory_vividness, 0.0f, 1.0f);
            return;
        }
    }
}

//=============================================================================
// LOSS PROCESSING
//=============================================================================

void grief_process_loss(grief_system_t* system,
                       uint32_t attachment_id,
                       loss_type_t loss_type,
                       uint64_t current_time_us) {
    // WHAT: Initiate grief response to loss
    // WHY:  Loss triggers cascade of psychological and biological changes
    // HOW:  Set up grief state, initial shock stage, neurochemical changes

    if (!system) return;

    // Find the attachment
    attachment_bond_t* lost_bond = NULL;
    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (system->attachments[i].active &&
            system->attachments[i].attachment_id == attachment_id) {
            lost_bond = &system->attachments[i];
            break;
        }
    }

    if (!lost_bond) return;  // Attachment not found

    // Mark bond as severed (but keep for continuing bonds)
    lost_bond->severed = true;
    lost_bond->loss_time = current_time_us;
    lost_bond->loss_type = loss_type;

    // Initialize grief state
    system->current_grief.experiencing_grief = true;
    system->current_grief.lost_attachment_id = attachment_id;
    system->current_grief.loss_onset_time = current_time_us;

    // Start in shock/denial stage
    system->current_grief.current_stage = GRIEF_STAGE_SHOCK;
    system->current_grief.stage_transition_time = current_time_us;

    // Initial stage intensities (shock dominates)
    system->current_grief.stage_intensities[GRIEF_STAGE_SHOCK] = 1.0f;
    system->current_grief.stage_intensities[GRIEF_STAGE_DENIAL] = 0.7f;
    system->current_grief.stage_intensities[GRIEF_STAGE_ANGER] = 0.0f;
    system->current_grief.stage_intensities[GRIEF_STAGE_BARGAINING] = 0.0f;
    system->current_grief.stage_intensities[GRIEF_STAGE_DEPRESSION] = 0.3f;
    system->current_grief.stage_intensities[GRIEF_STAGE_ACCEPTANCE] = 0.0f;
    system->current_grief.stage_intensities[GRIEF_STAGE_MEANING_MAKING] = 0.0f;

    // Initial symptom severity (depends on bond strength AND loss type)
    float bond_strength = lost_bond->strength;

    // Loss type intensity modifier
    float loss_type_intensity = 1.0f;
    switch (loss_type) {
        case LOSS_TYPE_DEATH:        loss_type_intensity = 1.0f; break;  // Maximum intensity
        case LOSS_TYPE_SEPARATION:   loss_type_intensity = 0.7f; break;  // Less permanent
        case LOSS_TYPE_REJECTION:    loss_type_intensity = 0.8f; break;  // Social pain
        case LOSS_TYPE_ROLE_LOSS:    loss_type_intensity = 0.6f; break;  // Identity loss
        case LOSS_TYPE_ANTICIPATORY: loss_type_intensity = 0.5f; break;  // Pre-grieving
        case LOSS_TYPE_AMBIGUOUS:    loss_type_intensity = 0.9f; break;  // Uncertainty compounds
        case LOSS_TYPE_SYMBOLIC:     loss_type_intensity = 0.5f; break;  // Abstract loss
        default:                     loss_type_intensity = 1.0f; break;
    }

    // Attachment type intensity modifier (some relationships are more intense)
    float attachment_type_intensity = 1.0f;
    switch (lost_bond->type) {
        case ATTACHMENT_PARENT:    attachment_type_intensity = 1.0f; break;  // Core attachment
        case ATTACHMENT_CHILD:     attachment_type_intensity = 1.0f; break;  // Deepest bond
        case ATTACHMENT_ROMANTIC:  attachment_type_intensity = 0.95f; break; // Very intense
        case ATTACHMENT_FRIEND:    attachment_type_intensity = 0.7f; break;  // Important but less
        case ATTACHMENT_PET:       attachment_type_intensity = 0.6f; break;  // Significant but different
        case ATTACHMENT_MENTOR:    attachment_type_intensity = 0.65f; break; // Influential
        case ATTACHMENT_PLACE:     attachment_type_intensity = 0.5f; break;  // Nostalgic
        case ATTACHMENT_IDENTITY:  attachment_type_intensity = 0.8f; break;  // Self-concept loss
        default:                   attachment_type_intensity = 0.7f; break;
    }

    // Memory influence (more shared memories = more intense grief)
    float memory_influence = 1.0f + (lost_bond->associated_memories / 100.0f) * lost_bond->memory_vividness;
    memory_influence = clamp(memory_influence, 1.0f, 1.5f);  // Up to 50% increase

    // Combine all intensity factors with safety ceiling
    float combined_intensity = loss_type_intensity * attachment_type_intensity * memory_influence;
    combined_intensity = clamp(combined_intensity, 0.0f, 1.3f);  // Cap to prevent extreme values

    // SAFETY: Grief symptoms capped to prevent complete incapacitation
    // Maximum emotional pain is 0.85 (never complete shutdown)
    system->current_grief.emotional_pain_intensity = clamp(bond_strength * 0.9f * combined_intensity, 0.0f, 0.85f);
    system->current_grief.anhedonia_level = clamp(bond_strength * 0.7f * combined_intensity, 0.0f, 0.80f);
    system->current_grief.intrusive_thoughts_frequency = clamp(bond_strength * 0.8f * combined_intensity, 0.0f, 0.85f);
    system->current_grief.avoidance_level = 0.5f;
    // SAFETY: Functional impairment capped at 0.85 (system remains minimally functional)
    system->current_grief.functional_impairment = clamp(bond_strength * 0.7f * combined_intensity, 0.0f, 0.85f);

    // Dual process: Initially loss-oriented
    system->current_grief.loss_orientation = 0.9f;
    system->current_grief.restoration_orientation = 0.1f;
    system->current_grief.last_oscillation_time = current_time_us;

    // Neurobiological changes with safety limits
    // SAFETY: Neuromodulator changes capped to prevent complete depletion/excess
    // System retains 15-20% baseline function even in severe grief
    system->current_grief.serotonin_depletion = clamp(GRIEF_SEROTONIN_DEPLETION * bond_strength, 0.0f, 0.80f);  // Max 80% depletion
    system->current_grief.dopamine_depletion = clamp(GRIEF_DOPAMINE_DEPLETION * bond_strength, 0.0f, 0.75f);   // Max 75% depletion
    system->current_grief.norepinephrine_elevation = clamp(GRIEF_NOREPINEPHRINE_INCREASE * bond_strength, 1.0f, 2.5f);
    system->current_grief.cortisol_elevation = clamp(GRIEF_CORTISOL_INCREASE * bond_strength, 1.0f, 2.5f);

    // Default to adaptive coping (can change)
    system->current_grief.predominant_coping = COPING_ADAPTIVE;
    system->current_grief.social_support_seeking = 0.5f;

    // Phase E1.1: Initialize sadness emotion
    // Grief-induced sadness proportional to bond strength and emotional pain
    system->current_grief.grief_induced_sadness = system->current_grief.emotional_pain_intensity;
    system->current_grief.sadness_emotion = grief_get_sadness_emotion(system);

    // Risk assessment
    bool high_dependency = (lost_bond->dependency > 0.7f);
    bool ambiguous_loss = (loss_type == LOSS_TYPE_AMBIGUOUS);
    bool previous_losses = (system->lifetime_losses > 2);

    system->current_grief.prolonged_grief_risk =
        (high_dependency || ambiguous_loss || previous_losses);

    // Statistics
    system->lifetime_losses++;

    /* Broadcast grief state change */
    bio_broadcast_grief_state(system);

    // Death loss awakens mortality awareness
    if (loss_type == LOSS_TYPE_DEATH) {
        system->existential.aware_of_mortality = true;
        system->existential.mortality_salience += 0.5f;  // Strong reminder
    } else {
        system->existential.mortality_salience += 0.3f;  // Moderate reminder
    }
    system->existential.mortality_salience = clamp(system->existential.mortality_salience, 0.0f, 1.0f);
    system->existential.last_mortality_reminder = current_time_us;
}

void grief_update(grief_system_t* system, float dt, uint64_t current_time_us) {
    // Process pending bio-async messages
    if (system && system->bio_async_enabled && system->bio_ctx_ptr) {
        bio_router_process_inbox(system->bio_ctx_ptr, 5);
    }

    // WHAT: Update grief processing over time
    // WHY:  Grief evolves - stages change, intensity decreases, integration occurs
    // HOW:  Update stage dynamics, dual process oscillation, neurochemistry

    if (!system || !system->current_grief.experiencing_grief) return;

    grief_state_t* grief = &system->current_grief;

    // Time since loss (in seconds)
    float time_since_loss = (float)(current_time_us - grief->loss_onset_time) / 1000000.0f;

    //=========================================================================
    // STAGE PROGRESSION
    //=========================================================================

    // WHAT: Evolve through grief stages (non-linear)
    // HOW:  Each stage increases/decreases based on time and other stages

    // Shock fades quickly (days to weeks)
    float shock_decay = expf(-time_since_loss / (86400.0f * 7.0f));  // 1 week half-life
    grief->stage_intensities[GRIEF_STAGE_SHOCK] = shock_decay;

    // Denial decreases gradually (ambiguous loss maintains baseline)
    // Find loss type for this grief
    loss_type_t current_loss_type = LOSS_TYPE_DEATH;
    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (system->attachments[i].attachment_id == grief->lost_attachment_id) {
            current_loss_type = system->attachments[i].loss_type;
            break;
        }
    }

    // Standard denial decay
    float denial_decay = expf(-time_since_loss / (86400.0f * 30.0f));  // 1 month half-life
    grief->stage_intensities[GRIEF_STAGE_DENIAL] *= denial_decay;

    // Ambiguous loss maintains elevated baseline denial (uncertainty prevents full acceptance)
    if (current_loss_type == LOSS_TYPE_AMBIGUOUS) {
        float ambiguous_denial_baseline = 0.35f * expf(-time_since_loss / (86400.0f * 365.0f));  // Decays over years, not months
        grief->stage_intensities[GRIEF_STAGE_DENIAL] = fmaxf(grief->stage_intensities[GRIEF_STAGE_DENIAL], ambiguous_denial_baseline);
    }

    // Avoidance coping maintains denial and slows acceptance
    if (grief->predominant_coping == COPING_AVOIDANT) {
        grief->stage_intensities[GRIEF_STAGE_DENIAL] += 0.1f * dt / 86400.0f;  // Slowly increases denial
        grief->stage_intensities[GRIEF_STAGE_DENIAL] = clamp(grief->stage_intensities[GRIEF_STAGE_DENIAL], 0.0f, 0.8f);
    }

    // Anger rises then falls (weeks)
    float anger_peak_time = 86400.0f * 14.0f;  // 2 weeks
    float anger_intensity = expf(-powf((time_since_loss - anger_peak_time) / (86400.0f * 21.0f), 2.0f));
    grief->stage_intensities[GRIEF_STAGE_ANGER] = anger_intensity * 0.7f;

    // Bargaining oscillates with denial
    grief->stage_intensities[GRIEF_STAGE_BARGAINING] =
        grief->stage_intensities[GRIEF_STAGE_DENIAL] * 0.6f;

    // Depression is sustained (months)
    float depression_onset = 1.0f - expf(-time_since_loss / (86400.0f * 7.0f));
    float depression_decay = expf(-time_since_loss / (86400.0f * 180.0f));  // 6 months
    grief->stage_intensities[GRIEF_STAGE_DEPRESSION] = depression_onset * depression_decay * 0.9f;

    // Acceptance grows slowly
    float acceptance_growth = 1.0f - expf(-time_since_loss / (86400.0f * 90.0f));  // 3 months
    grief->stage_intensities[GRIEF_STAGE_ACCEPTANCE] = acceptance_growth * 0.7f;

    // Meaning-making emerges later
    if (time_since_loss > 86400.0f * 60.0f) {  // After 2 months
        float meaning_growth = (time_since_loss - 86400.0f * 60.0f) / (86400.0f * 180.0f);
        grief->stage_intensities[GRIEF_STAGE_MEANING_MAKING] = clamp(meaning_growth, 0.0f, 0.8f);
    }

    // Update current predominant stage
    float max_intensity = 0.0f;
    grief_stage_t max_stage = GRIEF_STAGE_SHOCK;
    for (int i = 0; i < 7; i++) {
        if (grief->stage_intensities[i] > max_intensity) {
            max_intensity = grief->stage_intensities[i];
            max_stage = (grief_stage_t)i;
        }
    }

    if (max_stage != grief->current_stage) {
        grief->current_stage = max_stage;
        grief->stage_transition_time = current_time_us;
        grief->stage_cycles++;
    }

    //=========================================================================
    // SYMPTOM EVOLUTION
    //=========================================================================

    // Pain intensity decreases over time (but slowly)
    // Maladaptive/avoidant coping slows pain reduction
    float pain_decay_rate = 1.0f / (86400.0f * 180.0f);  // 6-month half-life
    if (grief->predominant_coping == COPING_AVOIDANT || grief->predominant_coping == COPING_MALADAPTIVE) {
        pain_decay_rate *= 0.5f;  // Half as fast recovery
    }
    grief->emotional_pain_intensity = exponential_decay(
        grief->emotional_pain_intensity, 0.0f, pain_decay_rate, dt);

    // Anhedonia follows dopamine recovery
    grief->anhedonia_level = grief->dopamine_depletion;

    // Intrusive thoughts decrease
    float intrusion_decay_rate = 1.0f / (86400.0f * 90.0f);  // 3-month half-life
    grief->intrusive_thoughts_frequency = exponential_decay(
        grief->intrusive_thoughts_frequency, 0.0f, intrusion_decay_rate, dt);

    // Functional impairment decreases (slower with poor coping, may increase)
    float impairment_decay_rate = 1.0f / (86400.0f * 60.0f);  // 2-month half-life

    if (grief->predominant_coping == COPING_MALADAPTIVE) {
        // Maladaptive coping causes impairment to worsen
        impairment_decay_rate *= 0.1f;  // Very slow recovery
        grief->functional_impairment += 0.08f * dt / 86400.0f;  // Worsens over time
        // SAFETY: Even with maladaptive coping, cap impairment at 0.90 (retain minimal function)
        grief->functional_impairment = clamp(grief->functional_impairment, 0.0f, 0.90f);
        // Still apply some decay to prevent unbounded growth
        grief->functional_impairment = exponential_decay(
            grief->functional_impairment, 0.2f, impairment_decay_rate, dt);  // Floor at 0.2
    } else if (grief->predominant_coping == COPING_AVOIDANT) {
        // Avoidant coping maintains elevated impairment (not dealing with grief)
        impairment_decay_rate *= 0.15f;  // Very slow recovery
        // Maintain a floor of impairment with avoidance
        float avoidance_impairment_floor = 0.55f * grief->avoidance_level;  // Higher floor
        grief->functional_impairment = exponential_decay(
            grief->functional_impairment, avoidance_impairment_floor, impairment_decay_rate, dt);
        // SAFETY: Cap at 0.85 even with high avoidance
        grief->functional_impairment = clamp(grief->functional_impairment, 0.0f, 0.85f);
    } else {
        // Adaptive coping allows normal recovery
        grief->functional_impairment = exponential_decay(
            grief->functional_impairment, 0.0f, impairment_decay_rate, dt);
    }

    //=========================================================================
    // DUAL PROCESS OSCILLATION (Stroebe & Schut)
    //=========================================================================

    // WHAT: Alternate between loss-focused and restoration-focused coping
    // WHY:  Healthy grief involves both confronting loss AND rebuilding life
    // HOW:  Oscillate with decreasing loss-focus over time

    // Use prime number period to avoid sampling artifacts
    float oscillation_period = 3607.0f;  // ~1 hour (prime number to avoid alignment)
    float time_in_cycle = fmodf(time_since_loss, oscillation_period);
    float cycle_phase = time_in_cycle / oscillation_period;  // [0, 1]

    // Early grief: mostly loss-oriented
    // Later grief: more restoration-oriented
    float restoration_bias = time_since_loss / (86400.0f * 180.0f);  // Grows over 6 months
    restoration_bias = clamp(restoration_bias, 0.0f, 0.8f);

    // Oscillate between loss and restoration with sinusoidal pattern
    // Range: [0.1, 0.9] for loss orientation (never fully 0 or 1)
    grief->loss_orientation = (1.0f - restoration_bias) * (0.5f + 0.4f * sinf(cycle_phase * 2.0f * M_PI));
    grief->restoration_orientation = 1.0f - grief->loss_orientation;

    //=========================================================================
    // NEUROBIOLOGICAL RECOVERY
    //=========================================================================

    // Serotonin recovers over months
    float serotonin_recovery_rate = 1.0f / (86400.0f * 90.0f);  // 3 months
    grief->serotonin_depletion = exponential_decay(
        grief->serotonin_depletion, 0.0f, serotonin_recovery_rate, dt);

    // Dopamine recovers as anhedonia lifts
    float dopamine_recovery_rate = 1.0f / (86400.0f * 60.0f);  // 2 months
    grief->dopamine_depletion = exponential_decay(
        grief->dopamine_depletion, 0.0f, dopamine_recovery_rate, dt);

    // Stress hormones normalize faster
    float stress_recovery_rate = 1.0f / (86400.0f * 30.0f);  // 1 month
    grief->norepinephrine_elevation = exponential_decay(
        grief->norepinephrine_elevation, 1.0f, stress_recovery_rate, dt);
    grief->cortisol_elevation = exponential_decay(
        grief->cortisol_elevation, 1.0f, stress_recovery_rate, dt);

    //=========================================================================
    // PROLONGED GRIEF CHECK
    //=========================================================================

    // Prolonged grief disorder: intense grief persisting > 6 months (death) or > 12 months (other)
    bool is_death = false;  // Need to check loss type
    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (system->attachments[i].attachment_id == grief->lost_attachment_id) {
            is_death = (system->attachments[i].loss_type == LOSS_TYPE_DEATH);
            break;
        }
    }

    float prolonged_threshold = is_death ? (86400.0f * 180.0f) : (86400.0f * 365.0f);

    // Maladaptive/avoidant coping gradually increases prolonged grief severity
    if (grief->predominant_coping == COPING_MALADAPTIVE) {
        grief->prolonged_grief_severity += 0.15f * dt / (86400.0f * 30.0f);  // 0.15 per month (increased from 0.1)
        grief->prolonged_grief_severity = clamp(grief->prolonged_grief_severity, 0.0f, 1.0f);
    } else if (grief->predominant_coping == COPING_AVOIDANT && time_since_loss > 86400.0f * 60.0f) {
        grief->prolonged_grief_severity += 0.08f * dt / (86400.0f * 30.0f);  // 0.08 per month after 2 months (increased from 0.05, earlier trigger)
        grief->prolonged_grief_severity = clamp(grief->prolonged_grief_severity, 0.0f, 1.0f);
    } else if (grief->predominant_coping == COPING_ADAPTIVE) {
        // Adaptive coping reduces severity
        grief->prolonged_grief_severity *= 0.99f;  // Slow reduction
        if (grief->prolonged_grief_severity < 0.1f) {
            grief->prolonged_grief_risk = false;  // No longer at risk
        }
    }

    if (time_since_loss > prolonged_threshold) {
        // Check if symptoms persist at high intensity
        bool intense_pain = (grief->emotional_pain_intensity > 0.7f);
        bool high_impairment = (grief->functional_impairment > 0.6f);
        bool persistent_preoccupation = (grief->intrusive_thoughts_frequency > 0.7f);

        if (intense_pain && high_impairment && persistent_preoccupation) {
            grief->prolonged_grief_risk = true;
            grief->prolonged_grief_severity = fmaxf(grief->prolonged_grief_severity, 0.8f);
            system->complicated_grief_episodes++;
        }
    }

    //=========================================================================
    // SADNESS EMOTION UPDATE (Phase E1.1)
    //=========================================================================

    // WHAT: Update grief-induced sadness based on emotional pain and depression
    // WHY:  Sadness tracks with grief intensity
    // HOW:  Combine pain intensity and depression stage, update emotional tag

    // Grief-induced sadness = combination of pain and depression stage
    float depression_contribution = grief->stage_intensities[GRIEF_STAGE_DEPRESSION] * 0.7f;
    float acute_sadness = (grief->emotional_pain_intensity * 0.6f) + (depression_contribution * 0.4f);

    // SADNESS DECAY: Slowly decrease over time but never completely fade (like humans)
    // Calculate permanent sadness baseline from bond strength and loss type
    float permanent_sadness_baseline = 0.0f;
    for (int i = 0; i < GRIEF_MAX_ATTACHMENTS; i++) {
        if (system->attachments[i].attachment_id == grief->lost_attachment_id && system->attachments[i].severed) {
            // Permanent baseline proportional to bond strength and loss permanence
            float bond_factor = system->attachments[i].strength * 0.15f;  // Up to 15% baseline
            float loss_permanence = (system->attachments[i].loss_type == LOSS_TYPE_DEATH) ? 1.0f : 0.5f;
            permanent_sadness_baseline = bond_factor * loss_permanence;
            break;
        }
    }

    // Decay toward permanent baseline (never fully to zero)
    float sadness_decay_rate = 1.0f / (86400.0f * 365.0f);  // 1-year half-life
    grief->grief_induced_sadness = exponential_decay(
        grief->grief_induced_sadness,
        permanent_sadness_baseline,  // Target: permanent baseline, not zero
        sadness_decay_rate,
        dt
    );

    // Acute sadness can spike above baseline
    grief->grief_induced_sadness = fmaxf(grief->grief_induced_sadness, acute_sadness);
    grief->grief_induced_sadness = clamp(grief->grief_induced_sadness, permanent_sadness_baseline, 1.0f);

    // Update sadness emotional tag (valence and arousal computed from current state)
    grief->sadness_emotion = grief_get_sadness_emotion(system);

    //=========================================================================
    // WISDOM ACCUMULATION (Incremental learning from grief)
    //=========================================================================

    // WHAT: Accumulate wisdom gradually through grief process
    // WHY:  Growth happens throughout grief, not just at "completion"
    // HOW:  Small increments based on meaning-making and acceptance

    if (grief->experiencing_grief) {
        // Wisdom grows with meaning-making and acceptance
        float wisdom_increment = grief->meaning_making_progress * grief->stage_intensities[GRIEF_STAGE_ACCEPTANCE] * dt / (86400.0f * 30.0f);
        system->accumulated_grief_wisdom += wisdom_increment;
        system->accumulated_grief_wisdom = clamp(system->accumulated_grief_wisdom, 0.0f, 1.0f);
    }

    //=========================================================================
    // INTEGRATION AND COMPLETION
    //=========================================================================

    // Check if grief has integrated (acceptance + low pain + meaning-making)
    bool has_accepted = (grief->stage_intensities[GRIEF_STAGE_ACCEPTANCE] > 0.6f);
    bool pain_manageable = (grief->emotional_pain_intensity < 0.3f);
    bool found_meaning = (grief->meaning_making_progress > 0.5f);

    if (has_accepted && pain_manageable && found_meaning) {
        // Grief integrated - but never truly "over"
        grief->experiencing_grief = false;

        // Completion bonus wisdom
        system->accumulated_grief_wisdom += 0.05f;
        system->accumulated_grief_wisdom = clamp(system->accumulated_grief_wisdom, 0.0f, 1.0f);

        // Update average duration
        float this_duration = time_since_loss;
        float total_grief_time = system->average_grief_duration * (float)(system->lifetime_losses - 1);
        system->average_grief_duration = (total_grief_time + this_duration) / (float)system->lifetime_losses;
    }

    system->total_update_calls++;
}

//=============================================================================
// EXISTENTIAL AWARENESS
//=============================================================================

void grief_contemplate_mortality(grief_system_t* system,
                                float mortality_reminder_intensity,
                                uint64_t current_time_us) {
    // WHAT: Process awareness of death and finitude
    // WHY:  Death awareness shapes behavior, values, meaning
    // HOW:  Update mortality salience, death anxiety, existential concerns

    if (!system) return;

    existential_state_t* ex = &system->existential;

    // Awakens mortality awareness
    if (!ex->aware_of_mortality && mortality_reminder_intensity > 0.3f) {
        ex->aware_of_mortality = true;
    }

    // Mortality salience increases with reminders
    ex->mortality_salience += mortality_reminder_intensity * 0.3f;
    ex->mortality_salience = clamp(ex->mortality_salience, 0.0f, 1.0f);
    ex->last_mortality_reminder = current_time_us;

    // Death anxiety may increase or decrease depending on acceptance
    if (ex->acceptance_of_finitude < 0.5f) {
        // Low acceptance → more anxiety (increased threshold to 0.5)
        ex->death_anxiety += mortality_reminder_intensity * 0.2f;
    } else {
        // High acceptance → less anxiety
        ex->death_anxiety -= mortality_reminder_intensity * 0.1f;
    }
    ex->death_anxiety = clamp(ex->death_anxiety, 0.0f, 1.0f);

    // Existential anxiety (meaninglessness) increases with mortality salience
    // and low-to-moderate sense of purpose (confronting death raises existential questions)
    if (ex->sense_of_purpose < 0.7f) {
        ex->existential_anxiety += mortality_reminder_intensity * 0.15f;
    }
    ex->existential_anxiety = clamp(ex->existential_anxiety, 0.0f, 1.0f);

    // Legacy concerns increase with mortality awareness
    ex->legacy_motivation = ex->mortality_salience * 0.7f;
}

void grief_find_meaning(grief_system_t* system, float meaning_making_effort) {
    // WHAT: Engage in meaning-making after loss
    // WHY:  Meaning transforms suffering into growth
    // HOW:  Increase purpose, acceptance, reduce anxiety

    if (!system) return;

    grief_state_t* grief = &system->current_grief;
    existential_state_t* ex = &system->existential;

    // Meaning-making progress
    grief->meaning_making_progress += meaning_making_effort * 0.1f;
    grief->meaning_making_progress = clamp(grief->meaning_making_progress, 0.0f, 1.0f);

    // Increases sense of purpose
    ex->sense_of_purpose += meaning_making_effort * 0.05f;
    ex->sense_of_purpose = clamp(ex->sense_of_purpose, 0.0f, 1.0f);

    // Reduces existential anxiety
    ex->existential_anxiety -= meaning_making_effort * 0.05f;
    ex->existential_anxiety = clamp(ex->existential_anxiety, 0.0f, 1.0f);

    // Increases acceptance of finitude
    ex->acceptance_of_finitude += meaning_making_effort * 0.03f;
    ex->acceptance_of_finitude = clamp(ex->acceptance_of_finitude, 0.0f, 1.0f);

    // Increases generativity (creating for future generations)
    ex->generativity += meaning_making_effort * 0.08f;
    ex->generativity = clamp(ex->generativity, 0.0f, 1.0f);

    // Legacy motivation (desire to leave impact)
    ex->legacy_motivation += meaning_making_effort * 0.06f;
    ex->legacy_motivation = clamp(ex->legacy_motivation, 0.0f, 1.0f);

    // Facilitates movement toward acceptance stage
    grief->stage_intensities[GRIEF_STAGE_ACCEPTANCE] += meaning_making_effort * 0.05f;
    grief->stage_intensities[GRIEF_STAGE_MEANING_MAKING] += meaning_making_effort * 0.1f;
}

//=============================================================================
// COPING MECHANISMS
//=============================================================================

void grief_seek_support(grief_system_t* system, float support_quality) {
    // WHAT: Reach out for social support
    // WHY:  Social connection aids grief recovery
    // HOW:  Reduces pain, increases restoration orientation

    if (!system) return;

    grief_state_t* grief = &system->current_grief;

    grief->social_support_seeking = support_quality;
    grief->predominant_coping = COPING_ADAPTIVE;

    // Support reduces pain and impairment
    grief->emotional_pain_intensity *= (1.0f - support_quality * 0.1f);
    grief->functional_impairment *= (1.0f - support_quality * 0.15f);

    // Increases restoration orientation
    grief->restoration_orientation += support_quality * 0.1f;
    grief->restoration_orientation = clamp(grief->restoration_orientation, 0.0f, 1.0f);
}

void grief_avoid_reminders(grief_system_t* system, float avoidance_intensity) {
    // WHAT: Engage in avoidant coping
    // WHY:  Sometimes protective, but can become maladaptive
    // HOW:  Reduces intrusive thoughts short-term, but prolongs grief

    if (!system) return;

    grief_state_t* grief = &system->current_grief;

    grief->avoidance_level = avoidance_intensity;

    // Short-term relief
    grief->intrusive_thoughts_frequency *= (1.0f - avoidance_intensity * 0.2f);

    // But if sustained, becomes maladaptive
    if (avoidance_intensity > 0.7f) {
        grief->predominant_coping = COPING_AVOIDANT;
        // Prolongs grief
        grief->prolonged_grief_risk = true;
    }
}

void grief_express_emotions(grief_system_t* system, float expression_intensity) {
    // WHAT: Process emotions expressively (crying, talking, writing)
    // WHY:  Emotional expression aids integration
    // HOW:  Reduces pain, facilitates stage progression

    if (!system) return;

    grief_state_t* grief = &system->current_grief;

    grief->predominant_coping = COPING_ADAPTIVE;

    // Reduces pain through catharsis
    grief->emotional_pain_intensity *= (1.0f - expression_intensity * 0.05f);

    // Facilitates acceptance
    grief->stage_intensities[GRIEF_STAGE_ACCEPTANCE] += expression_intensity * 0.02f;

    // Reduces prolonged grief risk
    if (expression_intensity > 0.5f) {
        grief->prolonged_grief_severity *= 0.95f;
    }
}

//=============================================================================
// QUERY FUNCTIONS
//=============================================================================

bool grief_is_grieving(const grief_system_t* system) {
    return system ? system->current_grief.experiencing_grief : false;
}

float grief_get_pain_intensity(const grief_system_t* system) {
    return system ? system->current_grief.emotional_pain_intensity : 0.0f;
}

grief_stage_t grief_get_current_stage(const grief_system_t* system) {
    return system ? system->current_grief.current_stage : GRIEF_STAGE_SHOCK;
}

bool grief_has_prolonged_grief_risk(const grief_system_t* system) {
    return system ? system->current_grief.prolonged_grief_risk : false;
}

void grief_get_neuromodulator_effects(const grief_system_t* system,
                                     float* serotonin_factor,
                                     float* dopamine_factor,
                                     float* norepinephrine_factor) {
    // WHAT: Get multipliers for neuromodulator integration
    // WHY:  Grief affects brain chemistry
    // HOW:  Return depletion/elevation factors

    if (!system || !system->current_grief.experiencing_grief) {
        if (serotonin_factor) *serotonin_factor = 1.0f;
        if (dopamine_factor) *dopamine_factor = 1.0f;
        if (norepinephrine_factor) *norepinephrine_factor = 1.0f;
        return;
    }

    const grief_state_t* grief = &system->current_grief;

    // Serotonin depleted → multiply by (1 - depletion)
    if (serotonin_factor) {
        *serotonin_factor = 1.0f - grief->serotonin_depletion;
    }

    // Dopamine depleted
    if (dopamine_factor) {
        *dopamine_factor = 1.0f - grief->dopamine_depletion;
    }

    // Norepinephrine elevated → multiply by elevation factor
    if (norepinephrine_factor) {
        *norepinephrine_factor = grief->norepinephrine_elevation;
    }
}

//=============================================================================
// EMOTION INTEGRATION (Phase E1.1: Sadness)
//=============================================================================

emotional_tag_t grief_get_sadness_emotion(const grief_system_t* system) {
    /* WHAT: Get sadness emotional state from grief
     * WHY:  Integration with emotional tagging system
     * HOW:  Compute valence and arousal from grief intensity
     *
     * SADNESS MAPPING (Russell's Circumplex Model):
     * - Valence: Negative, proportional to grief intensity [-0.3 to -0.9]
     * - Arousal: Low during sadness (vs high during anxiety/anger) [0.1 to 0.4]
     * - Category: EMOTION_SADNESS (valence < -0.3, arousal < 0.4)
     *
     * BIOLOGICAL: Sadness has low arousal (decreased energy, psychomotor retardation)
     * REFERENCE: Russell (1980) - Circumplex model of affect
     */

    if (!system || !system->current_grief.experiencing_grief) {
        // Not grieving: return neutral emotion
        return emotional_tag_neutral();
    }

    const grief_state_t* grief = &system->current_grief;

    // Valence: More negative with higher grief intensity
    // Emotional pain [0, 1] → valence [-0.3, -0.9]
    float valence = -0.3f - (grief->emotional_pain_intensity * 0.6f);
    valence = clamp(valence, -1.0f, -0.3f);

    // Arousal: Low during sadness (psychomotor retardation)
    // Depression and sadness are characterized by low energy
    // Arousal decreases with depression, increases slightly with anxiety
    float baseline_arousal = 0.2f;  // Low baseline for sadness
    float anxiety_boost = grief->stage_intensities[GRIEF_STAGE_ANGER] * 0.2f;  // Anger can increase arousal
    float arousal = baseline_arousal + anxiety_boost;
    arousal = clamp(arousal, 0.0f, 0.4f);  // Sadness has low arousal

    // Get current time (use 0 if not available - will be updated by caller)
    uint64_t timestamp_ms = 0;

    // Create emotional tag (will auto-classify as EMOTION_SADNESS)
    return emotional_tag_create(valence, arousal, timestamp_ms);
}

float grief_get_total_sadness(const grief_system_t* system) {
    /* WHAT: Get combined sadness (baseline + grief-induced)
     * WHY:  Track total sadness burden
     * HOW:  Sum baseline and grief-induced, clamp to [0, 1]
     */

    if (!system) return 0.0f;

    const grief_state_t* grief = &system->current_grief;

    // Total sadness = pre-existing + grief-induced
    float total = grief->baseline_sadness + grief->grief_induced_sadness;

    return clamp(total, 0.0f, 1.0f);
}

void grief_set_baseline_sadness(grief_system_t* system, float sadness_level) {
    /* WHAT: Set pre-grief sadness level
     * WHY:  Grief compounds with existing depression/sadness
     * HOW:  Update baseline, clamp to valid range
     */

    if (!system) return;

    system->current_grief.baseline_sadness = clamp(sadness_level, 0.0f, 1.0f);
}
