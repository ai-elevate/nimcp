/**
 * @file nimcp_audio_logic_bridge.c
 * @brief Audio-Logic Bridge Implementation
 *
 * Connects auditory perception to symbolic reasoning.
 * Enables speech understanding, sound-based inference, auditory grounding.
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/logic/nimcp_audio_logic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for audio_logic_bridge module */
static nimcp_health_agent_t* g_audio_logic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for audio_logic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void audio_logic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_audio_logic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from audio_logic_bridge module */
static inline void audio_logic_bridge_heartbeat(const char* operation, float progress) {
    if (g_audio_logic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_audio_logic_bridge_health_agent, operation, progress);
    }
}


//=============================================================================
// Internal Constants
//=============================================================================

#define MAX_ACTIVE_SPEAKERS 8
#define MAX_PENDING_COMMANDS 32
#define MAX_RECENT_SOUNDS 64
#define RECENT_WINDOW_US (5 * 1000000ULL)  /* 5 seconds */

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Speaker tracking entry
 */
typedef struct {
    char speaker_id[32];
    float confidence;
    uint64_t last_heard_us;
    bool active;
} speaker_entry_t;

/**
 * @brief Recent sound event
 */
typedef struct {
    sound_category_t category;
    char sound_name[64];
    float confidence;
    uint64_t timestamp_us;
} recent_sound_t;

/**
 * @brief Bridge internal structure
 */
struct audio_logic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* audio;                            /**< Audio cortex handle */
    void* logic;                            /**< Logic module handle */
    audio_logic_config_t config;            /**< Configuration */
    audio_logic_stats_t stats;              /**< Statistics */

    /* Speaker tracking */
    speaker_entry_t speakers[MAX_ACTIVE_SPEAKERS];
    uint32_t speaker_count;

    /* Recent sounds buffer (circular) */
    recent_sound_t recent_sounds[MAX_RECENT_SOUNDS];
    uint32_t recent_head;
    uint32_t recent_count;

    /* Pending commands */
    logic_audio_command_t pending_commands[MAX_PENDING_COMMANDS];
    uint32_t pending_count;

    /* Running averages */
    float speech_confidence_sum;
    float sound_confidence_sum;
    uint64_t speech_count;
    uint64_t sound_count;
};

//=============================================================================
// Category Name Lookup
//=============================================================================

static const char* category_names[SOUND_CAT_COUNT] = {
    "speech",
    "music",
    "environmental",
    "alert",
    "animal",
    "mechanical",
    "unknown"
};

const char* audio_logic_category_name(sound_category_t category) {
    if (category >= SOUND_CAT_COUNT) return "invalid";
    return category_names[category];
}

//=============================================================================
// Configuration API
//=============================================================================

audio_logic_config_t audio_logic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_default_", 0.0f);


    return (audio_logic_config_t){
        .enable_speech_grounding = true,
        .enable_sound_grounding = true,
        .enable_speaker_tracking = true,
        .enable_top_down_attention = true,
        .enable_verification = true,
        .enable_spatial_grounding = true,
        .min_confidence_threshold = 0.4f,
        .min_salience_threshold = 0.3f,
        .speech_priority_boost = 1.5f,
        .max_active_speakers = MAX_ACTIVE_SPEAKERS
    };
}

//=============================================================================
// Lifecycle API
//=============================================================================

audio_logic_bridge_t* audio_logic_bridge_create(
    void* audio,
    void* logic,
    const audio_logic_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__create", 0.0f);


    audio_logic_bridge_t* bridge = nimcp_calloc(1, sizeof(audio_logic_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "audio_logic_bridge_create: failed to allocate bridge");
        return NULL;
    }

    bridge->audio = audio;
    bridge->logic = logic;
    bridge->config = config ? *config : audio_logic_default_config();

    memset(&bridge->stats, 0, sizeof(bridge->stats));
    memset(bridge->speakers, 0, sizeof(bridge->speakers));
    memset(bridge->recent_sounds, 0, sizeof(bridge->recent_sounds));
    bridge->speaker_count = 0;
    bridge->recent_head = 0;
    bridge->recent_count = 0;
    bridge->pending_count = 0;

    bridge->speech_confidence_sum = 0.0f;
    bridge->sound_confidence_sum = 0.0f;
    bridge->speech_count = 0;
    bridge->sound_count = 0;

    return bridge;
}

void audio_logic_bridge_destroy(audio_logic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__destroy", 0.0f);


    if (bridge) {
        nimcp_free(bridge);
    }
}

int audio_logic_bridge_reset(audio_logic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_bridge_reset: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__reset", 0.0f);


    memset(bridge->speakers, 0, sizeof(bridge->speakers));
    memset(bridge->recent_sounds, 0, sizeof(bridge->recent_sounds));
    bridge->speaker_count = 0;
    bridge->recent_head = 0;
    bridge->recent_count = 0;
    bridge->pending_count = 0;

    return 0;
}

//=============================================================================
// Helper Functions
//=============================================================================

static int find_speaker(audio_logic_bridge_t* bridge, const char* speaker_id) {
    for (uint32_t i = 0; i < MAX_ACTIVE_SPEAKERS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && MAX_ACTIVE_SPEAKERS > 256) {
            audio_logic_bridge_heartbeat("audio_logic__loop",
                             (float)(i + 1) / (float)MAX_ACTIVE_SPEAKERS);
        }

        if (bridge->speakers[i].active &&
            strncmp(bridge->speakers[i].speaker_id, speaker_id, 32) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int find_free_speaker_slot(audio_logic_bridge_t* bridge) {
    for (uint32_t i = 0; i < MAX_ACTIVE_SPEAKERS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && MAX_ACTIVE_SPEAKERS > 256) {
            audio_logic_bridge_heartbeat("audio_logic__loop",
                             (float)(i + 1) / (float)MAX_ACTIVE_SPEAKERS);
        }

        if (!bridge->speakers[i].active) {
            return (int)i;
        }
    }
    return -1;
}

static void add_recent_sound(audio_logic_bridge_t* bridge, sound_category_t category,
                             const char* name, float confidence, uint64_t timestamp) {
    recent_sound_t* entry = &bridge->recent_sounds[bridge->recent_head];
    entry->category = category;
    strncpy(entry->sound_name, name, sizeof(entry->sound_name) - 1);
    entry->sound_name[sizeof(entry->sound_name) - 1] = '\0';
    entry->confidence = confidence;
    entry->timestamp_us = timestamp;

    bridge->recent_head = (bridge->recent_head + 1) % MAX_RECENT_SOUNDS;
    if (bridge->recent_count < MAX_RECENT_SOUNDS) {
        bridge->recent_count++;
    }
}

//=============================================================================
// Audio → Logic API
//=============================================================================

int audio_logic_ground_observation(
    audio_logic_bridge_t* bridge,
    const audio_logic_observation_t* obs
) {
    if (!bridge || !obs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_ground_observation: required parameter is NULL");
        return -1;
    }

    /* Filter by confidence and salience */
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_ground_o", 0.0f);


    if (obs->confidence < bridge->config.min_confidence_threshold) {
        return 0;
    }
    if (obs->salience < bridge->config.min_salience_threshold) {
        return 0;
    }

    uint64_t now = obs->timestamp_us ? obs->timestamp_us : nimcp_time_get_us();

    switch (obs->signal_type) {
        case AUDIO_LOGIC_SPEECH_TOKEN:
        case AUDIO_LOGIC_WORD_RECOGNIZED:
            if (!bridge->config.enable_speech_grounding) return 0;

            /* Track speaker if enabled */
            if (bridge->config.enable_speaker_tracking && obs->speaker_id[0] != '\0') {
                int slot = find_speaker(bridge, obs->speaker_id);
                if (slot < 0) {
                    slot = find_free_speaker_slot(bridge);
                    if (slot >= 0) {
                        strncpy(bridge->speakers[slot].speaker_id, obs->speaker_id,
                               sizeof(bridge->speakers[slot].speaker_id) - 1);
                        bridge->speakers[slot].active = true;
                        bridge->speaker_count++;
                        bridge->stats.speakers_tracked++;
                    }
                }
                if (slot >= 0) {
                    bridge->speakers[slot].confidence = obs->confidence;
                    bridge->speakers[slot].last_heard_us = now;
                }
            }

            bridge->stats.words_recognized++;
            bridge->speech_confidence_sum += obs->confidence;
            bridge->speech_count++;
            bridge->stats.avg_speech_confidence =
                bridge->speech_confidence_sum / bridge->speech_count;

            /*
             * TODO: Inject predicate into logic module
             * said(speaker, word) or heard(word)
             */
            break;

        case AUDIO_LOGIC_PHRASE_COMPLETE:
            if (!bridge->config.enable_speech_grounding) return 0;
            bridge->stats.phrases_completed++;
            break;

        case AUDIO_LOGIC_SPEAKER_IDENTIFIED:
            if (!bridge->config.enable_speaker_tracking) return 0;
            /* Update speaker confidence */
            {
                int slot = find_speaker(bridge, obs->speaker_id);
                if (slot >= 0) {
                    bridge->speakers[slot].confidence = obs->confidence;
                    bridge->speakers[slot].last_heard_us = now;
                }
            }
            break;

        case AUDIO_LOGIC_SOUND_DETECTED:
        case AUDIO_LOGIC_MUSIC_PATTERN:
            if (!bridge->config.enable_sound_grounding) return 0;

            add_recent_sound(bridge, obs->category, obs->concept_name,
                            obs->confidence, now);

            bridge->stats.sounds_grounded++;
            bridge->sound_confidence_sum += obs->confidence;
            bridge->sound_count++;
            bridge->stats.avg_sound_confidence =
                bridge->sound_confidence_sum / bridge->sound_count;

            /*
             * TODO: Inject predicate into logic module
             * heard(sound_name), playing(music)
             */
            break;

        default:
            return -1;
    }

    return 0;
}

int audio_logic_report_speech(
    audio_logic_bridge_t* bridge,
    const char* speaker,
    const char* words,
    float confidence
) {
    if (!bridge || !words) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_report_speech: required parameter is NULL");
        return -1;
    }
    if (!bridge->config.enable_speech_grounding) return 0;

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_report_s", 0.0f);


    audio_logic_observation_t obs = {0};
    obs.signal_type = AUDIO_LOGIC_WORD_RECOGNIZED;
    obs.category = SOUND_CAT_SPEECH;
    strncpy(obs.concept_name, words, sizeof(obs.concept_name) - 1);
    obs.confidence = confidence;
    obs.salience = 1.0f; /* Speech is always salient */
    if (speaker) {
        strncpy(obs.speaker_id, speaker, sizeof(obs.speaker_id) - 1);
    }
    obs.timestamp_us = nimcp_time_get_us();

    return audio_logic_ground_observation(bridge, &obs);
}

int audio_logic_report_sound(
    audio_logic_bridge_t* bridge,
    const char* sound_name,
    sound_category_t category,
    float confidence
) {
    if (!bridge || !sound_name) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_report_sound: required parameter is NULL");
        return -1;
    }
    if (!bridge->config.enable_sound_grounding) return 0;

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_report_s", 0.0f);


    audio_logic_observation_t obs = {0};
    obs.signal_type = AUDIO_LOGIC_SOUND_DETECTED;
    obs.category = category;
    strncpy(obs.concept_name, sound_name, sizeof(obs.concept_name) - 1);
    obs.confidence = confidence;
    obs.salience = 0.7f;
    obs.timestamp_us = nimcp_time_get_us();

    return audio_logic_ground_observation(bridge, &obs);
}

int audio_logic_process_batch(
    audio_logic_bridge_t* bridge,
    const audio_logic_observation_t* observations,
    uint32_t count
) {
    if (!bridge || !observations) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_process_batch: required parameter is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_process_", 0.0f);


    int processed = 0;
    for (uint32_t i = 0; i < count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && count > 256) {
            audio_logic_bridge_heartbeat("audio_logic__loop",
                             (float)(i + 1) / (float)count);
        }

        if (audio_logic_ground_observation(bridge, &observations[i]) == 0) {
            processed++;
        }
    }

    return processed;
}

//=============================================================================
// Logic → Audio API
//=============================================================================

int audio_logic_request_attention(
    audio_logic_bridge_t* bridge,
    sound_category_t category,
    float priority
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_request_attention: bridge is NULL");
        return -1;
    }
    if (category >= SOUND_CAT_COUNT) return -1;
    if (!bridge->config.enable_top_down_attention) return 0;

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_request_", 0.0f);


    if (bridge->pending_count >= MAX_PENDING_COMMANDS) {
        return -1;
    }

    logic_audio_command_t* cmd = &bridge->pending_commands[bridge->pending_count++];
    cmd->signal_type = LOGIC_AUDIO_ATTEND_SOUND;
    cmd->target_category = category;
    cmd->priority = priority < 0.0f ? 0.0f : (priority > 1.0f ? 1.0f : priority);
    cmd->expected_content[0] = '\0';
    cmd->target_speaker[0] = '\0';

    bridge->stats.attention_commands++;

    return 0;
}

int audio_logic_focus_speaker(
    audio_logic_bridge_t* bridge,
    const char* speaker_id,
    float priority
) {
    if (!bridge || !speaker_id) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_focus_speaker: required parameter is NULL");
        return -1;
    }
    if (!bridge->config.enable_top_down_attention) return 0;

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_focus_sp", 0.0f);


    if (bridge->pending_count >= MAX_PENDING_COMMANDS) {
        return -1;
    }

    logic_audio_command_t* cmd = &bridge->pending_commands[bridge->pending_count++];
    cmd->signal_type = LOGIC_AUDIO_ATTEND_SOUND;
    cmd->target_category = SOUND_CAT_SPEECH;
    strncpy(cmd->target_speaker, speaker_id, sizeof(cmd->target_speaker) - 1);
    cmd->target_speaker[sizeof(cmd->target_speaker) - 1] = '\0';
    cmd->priority = priority < 0.0f ? 0.0f : (priority > 1.0f ? 1.0f : priority);

    bridge->stats.attention_commands++;

    return 0;
}

int audio_logic_expect_word(
    audio_logic_bridge_t* bridge,
    const char* expected_word
) {
    if (!bridge || !expected_word) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_expect_word: required parameter is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_expect_w", 0.0f);


    if (bridge->pending_count >= MAX_PENDING_COMMANDS) {
        return -1;
    }

    logic_audio_command_t* cmd = &bridge->pending_commands[bridge->pending_count++];
    cmd->signal_type = LOGIC_AUDIO_EXPECT_WORD;
    cmd->target_category = SOUND_CAT_SPEECH;
    strncpy(cmd->expected_content, expected_word, sizeof(cmd->expected_content) - 1);
    cmd->expected_content[sizeof(cmd->expected_content) - 1] = '\0';
    cmd->priority = 0.8f;

    return 0;
}

int audio_logic_verify_predicate(
    audio_logic_bridge_t* bridge,
    const audio_logic_predicate_t* predicate,
    bool* verified,
    float* confidence
) {
    if (!bridge || !predicate || !verified || !confidence) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_verify_predicate: required parameter is NULL");
        return -1;
    }
    if (!bridge->config.enable_verification) {
        *verified = false;
        *confidence = 0.0f;
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_verify_p", 0.0f);


    bridge->stats.verifications_requested++;

    /* Search recent sounds for match */
    uint64_t now = nimcp_time_get_us();
    *verified = false;
    *confidence = 0.0f;

    for (uint32_t i = 0; i < bridge->recent_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->recent_count > 256) {
            audio_logic_bridge_heartbeat("audio_logic__loop",
                             (float)(i + 1) / (float)bridge->recent_count);
        }

        uint32_t idx = (bridge->recent_head + MAX_RECENT_SOUNDS - 1 - i) % MAX_RECENT_SOUNDS;
        recent_sound_t* sound = &bridge->recent_sounds[idx];

        /* Check if within time window */
        if (now - sound->timestamp_us > RECENT_WINDOW_US) {
            break;
        }

        /* Check for match */
        if (strncmp(sound->sound_name, predicate->object,
                   sizeof(sound->sound_name)) == 0) {
            *verified = true;
            *confidence = sound->confidence;
            bridge->stats.verifications_confirmed++;
            return 0;
        }
    }

    /* Also check speakers if looking for speech */
    if (predicate->subject[0] != '\0') {
        int slot = find_speaker(bridge, predicate->subject);
        if (slot >= 0 && bridge->speakers[slot].active) {
            *verified = true;
            *confidence = bridge->speakers[slot].confidence * 0.8f;
            bridge->stats.verifications_confirmed++;
        }
    }

    return 0;
}

int audio_logic_send_command(
    audio_logic_bridge_t* bridge,
    const logic_audio_command_t* command
) {
    if (!bridge || !command) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_send_command: required parameter is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_send_com", 0.0f);


    if (bridge->pending_count >= MAX_PENDING_COMMANDS) {
        return -1;
    }

    bridge->pending_commands[bridge->pending_count++] = *command;

    switch (command->signal_type) {
        case LOGIC_AUDIO_ATTEND_SOUND:
        case LOGIC_AUDIO_EXPECT_WORD:
        case LOGIC_AUDIO_FOCUS_FREQUENCY:
            bridge->stats.attention_commands++;
            break;
        case LOGIC_AUDIO_VERIFY_SPEAKER:
            bridge->stats.verifications_requested++;
            break;
        default:
            break;
    }

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

int audio_logic_is_speaker_active(
    const audio_logic_bridge_t* bridge,
    const char* speaker_id,
    bool* active
) {
    if (!bridge || !speaker_id || !active) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_is_speaker_active: required parameter is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_is_speak", 0.0f);


    for (uint32_t i = 0; i < MAX_ACTIVE_SPEAKERS; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && MAX_ACTIVE_SPEAKERS > 256) {
            audio_logic_bridge_heartbeat("audio_logic__loop",
                             (float)(i + 1) / (float)MAX_ACTIVE_SPEAKERS);
        }

        if (bridge->speakers[i].active &&
            strncmp(bridge->speakers[i].speaker_id, speaker_id, 32) == 0) {
            *active = true;
            return 0;
        }
    }

    *active = false;
    return 0;
}

int audio_logic_get_active_speaker_count(const audio_logic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_get_active_speaker_count: bridge is NULL");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_get_acti", 0.0f);


    return (int)bridge->speaker_count;
}

int audio_logic_category_heard_recently(
    const audio_logic_bridge_t* bridge,
    sound_category_t category,
    uint32_t recent_ms,
    bool* heard
) {
    if (!bridge || !heard) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_category_heard_recently: required parameter is NULL");
        return -1;
    }
    if (category >= SOUND_CAT_COUNT) return -1;

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__audio_logic_category", 0.0f);


    uint64_t now = nimcp_time_get_us();
    uint64_t window_us = (uint64_t)recent_ms * 1000ULL;

    *heard = false;

    for (uint32_t i = 0; i < bridge->recent_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->recent_count > 256) {
            audio_logic_bridge_heartbeat("audio_logic__loop",
                             (float)(i + 1) / (float)bridge->recent_count);
        }

        uint32_t idx = (bridge->recent_head + MAX_RECENT_SOUNDS - 1 - i) % MAX_RECENT_SOUNDS;
        recent_sound_t* sound = &bridge->recent_sounds[idx];

        if (now - sound->timestamp_us > window_us) {
            break;
        }

        if (sound->category == category) {
            *heard = true;
            return 0;
        }
    }

    return 0;
}

//=============================================================================
// Statistics API
//=============================================================================

int audio_logic_bridge_get_stats(
    const audio_logic_bridge_t* bridge,
    audio_logic_stats_t* stats
) {
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "audio_logic_bridge_get_stats: required parameter is NULL");
        return -1;
    }
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__get_stats", 0.0f);


    return 0;
}

void audio_logic_bridge_reset_stats(audio_logic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__reset_stats", 0.0f);


    if (bridge) {
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        bridge->speech_confidence_sum = 0.0f;
        bridge->sound_confidence_sum = 0.0f;
        bridge->speech_count = 0;
        bridge->sound_count = 0;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int audio_logic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    audio_logic_bridge_heartbeat("audio_logic__query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Audio_Logic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                audio_logic_bridge_heartbeat("audio_logic__loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Audio_Logic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Audio_Logic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
