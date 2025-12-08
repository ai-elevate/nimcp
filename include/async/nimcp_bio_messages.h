/**
 * @file nimcp_bio_messages.h
 * @brief Shared message types for bio-async inter-module communication
 *
 * WHAT: Defines all message types exchanged between NIMCP modules via bio-async
 * WHY:  Standardized message format enables decoupled module communication
 * HOW:  Each module includes this header to send/receive typed messages
 *
 * MESSAGE CATEGORIES:
 * 1. BRAIN MESSAGES - Brain state, neuron activation, network queries
 * 2. PLASTICITY MESSAGES - Weight updates, STDP events, learning signals
 * 3. COGNITIVE MESSAGES - Introspection, ethics, salience, decisions
 * 4. GLIAL MESSAGES - Astrocyte signaling, metabolic coordination
 * 5. MIDDLEWARE MESSAGES - Pipeline control, routing, encoding
 * 6. TRAINING MESSAGES - Loss, gradients, optimization steps
 *
 * CHANNEL ASSIGNMENT GUIDELINES:
 * - DOPAMINE: Reward signals, goal completion, weight updates
 * - SEROTONIN: Mood/state changes, slow coordination, ethics decisions
 * - NOREPINEPHRINE: Alerts, priority escalation, anomaly detection
 * - ACETYLCHOLINE: Attention shifts, fast queries, memory access
 *
 * @author NIMCP Development Team
 * @date 2025-11-28
 * @version 1.0.0
 */

#ifndef NIMCP_BIO_MESSAGES_H
#define NIMCP_BIO_MESSAGES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "async/nimcp_bio_async.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * MESSAGE TYPE ENUMERATION
 *============================================================================*/

/**
 * @brief Master enumeration of all bio-async message types
 */
typedef enum {
    /* Brain messages (0x0100 - 0x01FF) */
    BIO_MSG_BRAIN_STATE_QUERY = 0x0100,
    BIO_MSG_BRAIN_STATE_RESPONSE,
    BIO_MSG_NEURON_ACTIVATION_REQUEST,
    BIO_MSG_NEURON_ACTIVATION_RESPONSE,
    BIO_MSG_NETWORK_TOPOLOGY_QUERY,
    BIO_MSG_NETWORK_TOPOLOGY_RESPONSE,
    BIO_MSG_REGION_CONFIG_QUERY,
    BIO_MSG_REGION_CONFIG_RESPONSE,
    BIO_MSG_BRAIN_STEP_REQUEST,
    BIO_MSG_BRAIN_STEP_COMPLETE,

    /* Logic messages (0x0180 - 0x01FF) */
    BIO_MSG_LOGIC_GATE_EVALUATE = 0x0180,
    BIO_MSG_LOGIC_GATE_RESULT,
    BIO_MSG_LOGIC_CIRCUIT_STEP,
    BIO_MSG_LOGIC_CIRCUIT_COMPLETE,
    BIO_MSG_LOGIC_VARIABLE_BIND,
    BIO_MSG_LOGIC_VARIABLE_QUERY,
    BIO_MSG_LOGIC_SPIKE_EVENT,

    /* Plasticity messages (0x0200 - 0x02FF) */
    BIO_MSG_PLASTICITY_UPDATE = 0x0200,     /**< Generic plasticity update */
    BIO_MSG_WEIGHT_UPDATE_REQUEST,
    BIO_MSG_WEIGHT_UPDATE_RESPONSE,
    BIO_MSG_STDP_EVENT,
    BIO_MSG_STDP_BATCH_EVENT,
    BIO_MSG_LEARNING_RATE_UPDATE,
    BIO_MSG_NEUROMODULATOR_RELEASE,
    BIO_MSG_ELIGIBILITY_TRACE_UPDATE,
    BIO_MSG_HOMEOSTATIC_ADJUSTMENT,
    BIO_MSG_BCM_THRESHOLD_UPDATE,
    BIO_MSG_DENDRITIC_SPIKE,
    BIO_MSG_STP_EVENT,                      /**< Short-term plasticity event */
    BIO_MSG_ADAPTIVE_PLASTICITY_EVENT,      /**< Adaptive plasticity event */
    BIO_MSG_PREDICTIVE_CODING_UPDATE,       /**< Predictive coding update */

    /* Cognitive messages (0x0300 - 0x03FF) */
    BIO_MSG_INTROSPECTION_QUERY = 0x0300,
    BIO_MSG_INTROSPECTION_RESPONSE,
    BIO_MSG_ETHICS_EVALUATION_REQUEST,
    BIO_MSG_ETHICS_EVALUATION_RESPONSE,
    BIO_MSG_SALIENCE_QUERY,
    BIO_MSG_SALIENCE_RESPONSE,
    BIO_MSG_ATTENTION_SHIFT,
    BIO_MSG_WORKING_MEMORY_STORE,
    BIO_MSG_WORKING_MEMORY_RETRIEVE,
    BIO_MSG_KNOWLEDGE_QUERY,
    BIO_MSG_KNOWLEDGE_RESPONSE,
    BIO_MSG_CURIOSITY_SIGNAL,
    BIO_MSG_DECISION_REQUEST,
    BIO_MSG_DECISION_RESPONSE,
    BIO_MSG_CONSOLIDATION_TRIGGER,
    BIO_MSG_MIRROR_NEURON_ACTIVATION,

    /* Glial messages (0x0400 - 0x04FF) */
    BIO_MSG_ASTROCYTE_CALCIUM_WAVE = 0x0400,
    BIO_MSG_ASTROCYTE_GLUTAMATE_UPTAKE,
    BIO_MSG_MICROGLIA_ALERT,
    BIO_MSG_MICROGLIA_PRUNE_REQUEST,
    BIO_MSG_OLIGODENDROCYTE_MYELINATE,
    BIO_MSG_METABOLIC_DEMAND,
    BIO_MSG_METABOLIC_SUPPLY,
    BIO_MSG_GLIAL_SYNC_REQUEST,

    /* Middleware messages (0x0500 - 0x05FF) */
    BIO_MSG_PIPELINE_STAGE_COMPLETE = 0x0500,
    BIO_MSG_PIPELINE_ERROR,
    BIO_MSG_ENCODING_REQUEST,
    BIO_MSG_ENCODING_RESPONSE,
    BIO_MSG_SIGNAL_ROUTE_REQUEST,
    BIO_MSG_SIGNAL_ROUTE_COMPLETE,
    BIO_MSG_EVENT_BUS_PUBLISH,
    BIO_MSG_EVENT_BUS_SUBSCRIBE,

    /* Training messages (0x0600 - 0x06FF) */
    BIO_MSG_TRAINING_STEP_REQUEST = 0x0600,
    BIO_MSG_TRAINING_STEP_COMPLETE,
    BIO_MSG_LOSS_COMPUTED,
    BIO_MSG_GRADIENT_COMPUTED,
    BIO_MSG_OPTIMIZER_STEP,
    BIO_MSG_BATCH_COMPLETE,
    BIO_MSG_EPOCH_COMPLETE,
    BIO_MSG_CHECKPOINT_REQUEST,
    BIO_MSG_CHECKPOINT_COMPLETE,
    BIO_MSG_TRAINING_METRIC,        /**< Generic training metric */
    BIO_MSG_LR_CHANGED,             /**< Learning rate changed */

    /* System messages (0x0700 - 0x07FF) */
    BIO_MSG_SHUTDOWN_REQUEST = 0x0700,
    BIO_MSG_SHUTDOWN_ACK,
    BIO_MSG_HEALTH_CHECK,
    BIO_MSG_HEALTH_RESPONSE,
    BIO_MSG_CONFIG_UPDATE,
    BIO_MSG_CONFIG_ACK,
    BIO_MSG_ERROR_REPORT,
    BIO_MSG_LOG_EVENT,

    /* Security messages (0x0750 - 0x076F) */
    BIO_MSG_SECURITY_EVENT = 0x0750,        /**< General security event */
    BIO_MSG_SECURITY_ALERT,                 /**< Security alert notification */
    BIO_MSG_SECURITY_POLICY_UPDATE,         /**< Policy update notification */
    BIO_MSG_SECURITY_THREAT_DETECTED,       /**< Threat detection event */
    BIO_MSG_SECURITY_LEVEL_CHANGE,          /**< Security level changed */
    BIO_MSG_SECURITY_CONSENSUS_REQUEST,     /**< Consensus protocol request */
    BIO_MSG_SECURITY_CONSENSUS_RESPONSE,    /**< Consensus protocol response */
    BIO_MSG_SECURITY_RATE_LIMIT,            /**< Rate limiting event */
    BIO_MSG_SECURITY_AUDIT_EVENT,           /**< Security audit event */

    /* Perception messages (0x0800 - 0x08FF) */
    BIO_MSG_VISUAL_INPUT = 0x0800,
    BIO_MSG_VISUAL_FEATURE_DETECTED,
    BIO_MSG_VISUAL_ATTENTION_SHIFT,
    BIO_MSG_AUDIO_INPUT,
    BIO_MSG_AUDIO_FEATURE_DETECTED,
    BIO_MSG_AUDIO_ATTENTION_SHIFT,
    BIO_MSG_SPEECH_ONSET_DETECTED,
    BIO_MSG_PHONEME_RECOGNIZED,
    BIO_MSG_WORD_RECOGNIZED,
    BIO_MSG_MULTIMODAL_BINDING,

    /* Language messages (0x0900 - 0x09FF) */
    BIO_MSG_LEXICAL_ACCESS_REQUEST = 0x0900,
    BIO_MSG_LEXICAL_ACCESS_RESPONSE,
    BIO_MSG_SYNTAX_PARSE_REQUEST,
    BIO_MSG_SYNTAX_PARSE_RESULT,
    BIO_MSG_PHONOLOGICAL_ENCODE_REQUEST,
    BIO_MSG_PHONOLOGICAL_ENCODE_RESULT,
    BIO_MSG_MOTOR_COMMAND_REQUEST,
    BIO_MSG_MOTOR_COMMAND_RESULT,
    BIO_MSG_UTTERANCE_PRODUCTION_REQUEST,
    BIO_MSG_UTTERANCE_PRODUCTION_COMPLETE,
    BIO_MSG_SPEECH_FEEDBACK,
    BIO_MSG_LANGUAGE_ERROR,

    /* Neural Link Protocol (NLP) networking messages (0x0A00 - 0x0AFF) */
    BIO_MSG_NLP_SESSION_CONNECTED = 0x0A00,     /**< Peer session established */
    BIO_MSG_NLP_SESSION_DISCONNECTED,           /**< Peer session closed */
    BIO_MSG_NLP_SESSION_STATE_CHANGE,           /**< Session state transition */
    BIO_MSG_NLP_MESSAGE_RECEIVED,               /**< NLP message received from peer */
    BIO_MSG_NLP_MESSAGE_SENT,                   /**< NLP message sent to peer */
    BIO_MSG_NLP_MESSAGE_ACK,                    /**< Message acknowledgment */
    BIO_MSG_NLP_CRYPTO_ENCRYPT_REQUEST,         /**< Encryption request */
    BIO_MSG_NLP_CRYPTO_ENCRYPT_COMPLETE,        /**< Encryption completed */
    BIO_MSG_NLP_CRYPTO_DECRYPT_REQUEST,         /**< Decryption request */
    BIO_MSG_NLP_CRYPTO_DECRYPT_COMPLETE,        /**< Decryption completed */
    BIO_MSG_NLP_CRYPTO_KEY_EXCHANGE,            /**< Key exchange event */
    BIO_MSG_NLP_CRYPTO_ERROR,                   /**< Cryptographic error */
    BIO_MSG_NLP_COMPRESSION_REQUEST,            /**< Compression request */
    BIO_MSG_NLP_COMPRESSION_COMPLETE,           /**< Compression completed */
    BIO_MSG_NLP_PROTOCOL_MODE_CHANGE,           /**< Protocol mode changed */
    BIO_MSG_NLP_NEURAL_ENCODE_REQUEST,          /**< Neural encoding request */
    BIO_MSG_NLP_NEURAL_ENCODE_COMPLETE,         /**< Neural encoding completed */
    BIO_MSG_NLP_NEURAL_DECODE_REQUEST,          /**< Neural decoding request */
    BIO_MSG_NLP_NEURAL_DECODE_COMPLETE,         /**< Neural decoding completed */
    BIO_MSG_NLP_BRIDGE_EVENT,                   /**< Protocol bridge event */
    BIO_MSG_NLP_ERROR,                          /**< General NLP error */

    /* Swarm messages (0x0B00 - 0x0BFF) */
    BIO_MSG_SWARM_ENERGY_UPDATE = 0x0B00,       /**< Energy level update */
    BIO_MSG_SWARM_ENERGY_GOSSIP,                /**< Energy gossip protocol */
    BIO_MSG_SWARM_ENERGY_CRITICAL,              /**< Critical energy alert */
    BIO_MSG_SWARM_CASCADE_FAILURE,              /**< Cascade failure detected */
    BIO_MSG_SWARM_CASCADE_RECOVERY,             /**< Cascade recovery initiated */
    BIO_MSG_SWARM_POSITION_UPDATE,              /**< Position broadcast */
    BIO_MSG_SWARM_FORMATION_STATE,              /**< Formation state broadcast */
    BIO_MSG_SWARM_DEFORMATION_ALERT,            /**< Swarm deformation alert */
    BIO_MSG_SWARM_MEMORY_SYNC,                  /**< Memory synchronization */
    BIO_MSG_SWARM_QUORUM_VOTE,                  /**< Quorum voting message */
    BIO_MSG_SWARM_FLOCKING_UPDATE,              /**< Flocking behavior update */
    BIO_MSG_SWARM_IMMUNE_ALERT,                 /**< Immune system alert */
    BIO_MSG_SWARM_MORPHOGENESIS_SIGNAL,         /**< Morphogenesis signal */
    BIO_MSG_SWARM_PHEROMONE_RELEASE,            /**< Pheromone release event */

    /* Portia messages (0x0C00 - 0x0CFF) */
    BIO_MSG_PORTIA_PLAN_CREATED = 0x0C00,       /**< New plan created */
    BIO_MSG_PORTIA_PLAN_UPDATED,                /**< Plan state updated */
    BIO_MSG_PORTIA_PLAN_COMPLETED,              /**< Plan completed */
    BIO_MSG_PORTIA_PLAN_FAILED,                 /**< Plan failed */
    BIO_MSG_PORTIA_TIER_CHANGE,                 /**< Platform tier changed */
    BIO_MSG_PORTIA_DEGRADATION_EVENT,           /**< Degradation event */
    BIO_MSG_PORTIA_SENSOR_FUSION_UPDATE,        /**< Sensor fusion update */
    BIO_MSG_PORTIA_POWER_STATE_CHANGE,          /**< Power state changed */
    BIO_MSG_PORTIA_LEARNING_EVENT,              /**< Learning event */

    /* Sentinel */
    BIO_MSG_TYPE_COUNT
} bio_message_type_t;

/*=============================================================================
 * MESSAGE HEADER (Common to all messages)
 *============================================================================*/

/**
 * @brief Common header for all bio-async messages
 */
typedef struct {
    bio_message_type_t type;        /**< Message type identifier */
    uint32_t sequence_id;           /**< Sequence number for ordering */
    uint32_t source_module;         /**< Source module identifier */
    uint32_t target_module;         /**< Target module (0 = broadcast) */
    uint64_t timestamp_us;          /**< Timestamp in microseconds */
    nimcp_bio_channel_type_t channel; /**< Recommended channel for response */
    uint32_t payload_size;          /**< Size of payload in bytes */
    uint32_t flags;                 /**< Message flags */
} bio_message_header_t;

/** Message flags */
#define BIO_MSG_FLAG_URGENT         (1 << 0)  /**< Priority handling */
#define BIO_MSG_FLAG_REQUIRES_ACK   (1 << 1)  /**< Sender expects acknowledgment */
#define BIO_MSG_FLAG_BROADCAST      (1 << 2)  /**< Send to all modules */
#define BIO_MSG_FLAG_COMPRESSED     (1 << 3)  /**< Payload is compressed */
#define BIO_MSG_FLAG_ENCRYPTED      (1 << 4)  /**< Payload is encrypted */

/*=============================================================================
 * MODULE IDENTIFIERS
 *============================================================================*/

typedef enum {
    BIO_MODULE_UNKNOWN = 0,

    /* Core modules (0x0100-0x010F) */
    BIO_MODULE_BRAIN = 0x0100,
    BIO_MODULE_NEURON_MODEL,
    BIO_MODULE_SYNAPSE,
    BIO_MODULE_TOPOLOGY,
    BIO_MODULE_CORTICAL_COLUMN,
    BIO_MODULE_BRAIN_REGION,
    BIO_MODULE_NEURAL_LOGIC,        /**< Neural logic gate module */
    BIO_MODULE_NEURALNET,           /**< Neural network module */
    BIO_MODULE_AXON,                /**< Axon module */

    /* Neuron model submodules (0x0130-0x013F) */
    BIO_MODULE_NEURON_MODEL_IZHIKEVICH = 0x0130,
    BIO_MODULE_NEURON_MODEL_TWO_COMPARTMENT,
    BIO_MODULE_NEURON_MODEL_BIO_ASYNC,
    BIO_MODULE_SPIKE_EVENT,

    /* Cortical column submodules (0x0140-0x014F) */
    BIO_MODULE_CORTICAL_COLUMNAR_CONNECTIVITY = 0x0140,
    BIO_MODULE_CORTICAL_LAYERS,
    BIO_MODULE_CORTICAL_HYPERCOLUMNS,
    BIO_MODULE_CORTICAL_ORIENTATION,
    BIO_MODULE_CORTICAL_TOPOGRAPHIC,

    /* Topology submodules (0x0150-0x015F) */
    BIO_MODULE_TOPOLOGY_COMMUNITY = 0x0150,
    BIO_MODULE_TOPOLOGY_FRACTAL,
    BIO_MODULE_TOPOLOGY_NETWORK_BUILDER,

    /* Brain submodules (0x0110-0x012F) */
    BIO_MODULE_BRAIN_BIOLOGICAL = 0x0110,    /**< Biological subsystems */
    BIO_MODULE_BRAIN_ACCESSORS,              /**< Brain accessor methods */
    BIO_MODULE_BRAIN_OSCILLATIONS,           /**< Complex oscillations */
    BIO_MODULE_BRAIN_PROCESSING,             /**< Processing subsystem */
    BIO_MODULE_BRAIN_BROCA,                  /**< Broca's area (language) */
    BIO_MODULE_BRAIN_LEARNING,               /**< Learning subsystem */
    BIO_MODULE_BRAIN_COGNITIVE,              /**< Cognitive integration */
    BIO_MODULE_BRAIN_ANALYSIS,               /**< Topology analysis */
    BIO_MODULE_BRAIN_PRETRAINED,             /**< Pretrained models */
    BIO_MODULE_BRAIN_INFORMATION,            /**< Information theory */
    BIO_MODULE_BRAIN_DISTRIBUTED,            /**< Distributed brain */
    BIO_MODULE_BRAIN_STRATEGY,               /**< Strategy pattern */
    BIO_MODULE_BRAIN_FACTORY,                /**< Brain factory */
    BIO_MODULE_BRAIN_FACTORY_INIT,           /**< Factory initialization */
    BIO_MODULE_BRAIN_FACTORY_VALIDATION,     /**< Factory validation */
    BIO_MODULE_BRAIN_PERSISTENCE,            /**< Persistence layer */
    BIO_MODULE_BRAIN_INFERENCE,              /**< Inference engine */
    BIO_MODULE_BRAIN_LANGUAGE_PRODUCTION,    /**< Language production bridge */
    BIO_MODULE_BRAIN_SYNTAX,                 /**< Syntax processor */
    BIO_MODULE_BRAIN_SPEECH_MOTOR,           /**< Speech motor control */
    BIO_MODULE_BRAIN_PHONOLOGICAL,           /**< Phonological processing */
    BIO_MODULE_BRAIN_MULTIMODAL,             /**< Multimodal integrator */
    BIO_MODULE_BRAIN_SENSORY,                /**< Sensory extractor */
    BIO_MODULE_BRAIN_CIRCUIT_COMPILATION,    /**< Circuit compilation */
    BIO_MODULE_BRAIN_REASONING,              /**< Reasoning learning */
    BIO_MODULE_BRAIN_ASSOCIATION,            /**< Association learning */
    BIO_MODULE_BRAIN_RULE,                   /**< Rule learning */
    BIO_MODULE_BRAIN_RESIZE,                 /**< Brain resize operations */
    BIO_MODULE_DISTRIBUTED_COW,              /**< Distributed copy-on-write */

    /* Cognitive modules */
    BIO_MODULE_INTROSPECTION = 0x0200,
    BIO_MODULE_ETHICS,
    BIO_MODULE_SALIENCE,
    BIO_MODULE_ATTENTION,
    BIO_MODULE_WORKING_MEMORY,
    BIO_MODULE_KNOWLEDGE,
    BIO_MODULE_CURIOSITY,
    BIO_MODULE_CONSOLIDATION,
    BIO_MODULE_MIRROR_NEURONS,
    BIO_MODULE_EXECUTIVE,
    BIO_MODULE_GLOBAL_WORKSPACE,
    BIO_MODULE_WELLBEING,
    BIO_MODULE_EPISTEMIC,
    BIO_MODULE_FRACTAL_COGNITIVE,
    BIO_MODULE_NETWORK_ANALYSIS,

    /* Introspection submodules (0x0210-0x021F) */
    BIO_MODULE_INTROSPECTION_METACOGNITION = 0x0210,
    BIO_MODULE_INTROSPECTION_SELF_AWARENESS,
    BIO_MODULE_INTROSPECTION_SELF_MODEL,
    BIO_MODULE_INTROSPECTION_THEORY_OF_MIND,

    /* Attention submodules (0x0220-0x022F) */
    BIO_MODULE_ATTENTION_FAULT = 0x0220,
    BIO_MODULE_ATTENTION_REASONING,

    /* Curiosity submodules (0x0228-0x022F) */
    BIO_MODULE_CURIOSITY_REASONING = 0x0228,

    /* Executive submodules (0x0230-0x023F) */
    BIO_MODULE_EXECUTIVE_RECOVERY = 0x0230,

    /* Knowledge submodules (0x0238-0x024F) */
    BIO_MODULE_KNOWLEDGE_EXPLANATIONS = 0x0238,
    BIO_MODULE_KNOWLEDGE_SYMBOLIC_LOGIC,
    BIO_MODULE_KNOWLEDGE_META_LEARNING,
    BIO_MODULE_KNOWLEDGE_INTERFACE,
    BIO_MODULE_KNOWLEDGE_FACTORY,
    BIO_MODULE_KNOWLEDGE_INTEGRATION,

    /* Memory submodules (0x0250-0x025F) */
    BIO_MODULE_MEMORY_AUTOBIOGRAPHICAL = 0x0250,
    BIO_MODULE_MEMORY_EPISODIC_RECOVERY,
    BIO_MODULE_MEMORY_ENGRAM,
    BIO_MODULE_MEMORY_SEMANTIC,

    /* Working memory submodules (0x0260-0x026F) */
    BIO_MODULE_WORKING_MEMORY_FAULT = 0x0260,
    BIO_MODULE_WORKING_MEMORY_TRANSFER,

    /* Consolidation submodules (0x0268-0x026F) */
    BIO_MODULE_CONSOLIDATION_RECOVERY = 0x0268,
    BIO_MODULE_CONSOLIDATION_SYSTEMS,
    BIO_MODULE_CONSOLIDATION_SLEEP,

    /* Mirror neuron submodules (0x0270-0x027F) */
    BIO_MODULE_MIRROR_NEURONS_STDP = 0x0270,

    /* Wellbeing submodules (0x0278-0x027F) */
    BIO_MODULE_WELLBEING_MENTAL_HEALTH = 0x0278,

    /* Emotion modules (0x0320 - 0x032F) */
    BIO_MODULE_EMOTIONS = 0x0320,
    BIO_MODULE_EMOTION_RECOGNITION,
    BIO_MODULE_EMPATHETIC_RESPONSE,
    BIO_MODULE_GRIEF,
    BIO_MODULE_JOY,
    BIO_MODULE_REMORSE,
    BIO_MODULE_EMOTIONAL_TAGGING,

    /* Emotion submodules (0x0330 - 0x034F) */
    BIO_MODULE_EMOTIONS_BIAS = 0x0330,
    BIO_MODULE_EMOTIONS_PERSONALITY,
    BIO_MODULE_EMOTIONS_SHADOW,
    BIO_MODULE_EMOTIONS_SOCIAL,

    /* Glial modules */
    BIO_MODULE_ASTROCYTE = 0x0300,
    BIO_MODULE_ASTROCYTES = 0x0300,  /* Alias for compatibility */
    BIO_MODULE_MICROGLIA,
    BIO_MODULE_OLIGODENDROCYTE,
    BIO_MODULE_OLIGODENDROCYTES = BIO_MODULE_OLIGODENDROCYTE,  /* Alias */
    BIO_MODULE_MYELIN,
    BIO_MODULE_GLIAL_INTEGRATION,

    /* Plasticity modules */
    BIO_MODULE_PLASTICITY = 0x0400,      /**< General plasticity module */
    BIO_MODULE_STDP = 0x0401,
    BIO_MODULE_STP = 0x0402,
    BIO_MODULE_HOMEOSTATIC = 0x0403,
    BIO_MODULE_BCM = 0x0404,
    BIO_MODULE_DENDRITIC = 0x0405,
    BIO_MODULE_ADAPTIVE = 0x0406,
    BIO_MODULE_ATTENTION_PLASTICITY = 0x0407,
    BIO_MODULE_PREDICTIVE_CODING = 0x0408,
    BIO_MODULE_NEUROMODULATOR = 0x0409,
    BIO_MODULE_PINK_NOISE = 0x040A,
    BIO_MODULE_ELIGIBILITY_TRACE = 0x040B,

    /* Neuromodulator submodules (0x040C-0x041F) */
    BIO_MODULE_NEUROMODULATOR_SPATIAL = 0x040C,
    BIO_MODULE_NEUROMODULATOR_PINK_NOISE = 0x040D,
    BIO_MODULE_NEUROMODULATOR_METABOLIC = 0x040E,
    BIO_MODULE_NEUROMODULATOR_RECEPTOR = 0x040F,
    BIO_MODULE_NEUROMODULATOR_PHASIC_TONIC = 0x0410,
    BIO_MODULE_NEUROMODULATOR_VESICLE = 0x0411,

    /* Middleware modules */
    BIO_MODULE_PIPELINE = 0x0500,
    BIO_MODULE_ENCODING,
    BIO_MODULE_EVENT_BUS,
    BIO_MODULE_SIGNAL_ROUTER,
    BIO_MODULE_TRAINING,

    /* Training submodules (0x0506-0x050F) */
    BIO_MODULE_TRAINING_INTEGRATION = 0x0506,
    BIO_MODULE_TRAINING_EVENT_DRIVEN,
    BIO_MODULE_TRAINING_ADAPTERS,
    BIO_MODULE_TRAINING_ADAPTERS_LR,
    BIO_MODULE_TRAINING_ADAPTERS_LOSS,
    BIO_MODULE_TRAINING_PLASTICITY_BRIDGE,
    BIO_MODULE_TRAINING_REGULARIZATION,
    BIO_MODULE_TRAINING_OPTIMIZER,
    BIO_MODULE_TRAINING_LOSS,
    BIO_MODULE_TRAINING_LR_SCHEDULER,
    BIO_MODULE_TRAINING_GRADIENT_MANAGER,
    BIO_MODULE_TRAINING_CALLBACKS,
    BIO_MODULE_TRAINING_MODULE,

    /* Middleware integration submodules (0x0510-0x051F) */
    BIO_MODULE_MIDDLEWARE_FLOW_TRACKER = 0x0510,
    BIO_MODULE_MIDDLEWARE_CONTROLLER,
    BIO_MODULE_MIDDLEWARE_SHANNON,
    BIO_MODULE_MIDDLEWARE_EXEC_ADAPTER,
    BIO_MODULE_MIDDLEWARE_QUANTUM_PROPAGATOR,

    /* System modules */
    BIO_MODULE_SYSTEM = 0x0600,
    BIO_MODULE_MEMORY,
    BIO_MODULE_LOGGING,
    BIO_MODULE_SECURITY,
    BIO_MODULE_CAPABILITY,
    BIO_MODULE_CFI,
    BIO_MODULE_SECURITY_AUDIT,

    /* System submodules (0x0608-0x061F) */
    BIO_MODULE_SYSTEM_FAILURE_PREDICTION = 0x0608,
    BIO_MODULE_SYSTEM_GPU_NEURON,
    BIO_MODULE_SYSTEM_SHANNON,

    /* Networking modules (0x0680 - 0x06FF) */
    BIO_MODULE_DISTRIBUTED = 0x0680,
    BIO_MODULE_NETWORK_EVENTS,
    BIO_MODULE_P2P,
    BIO_MODULE_PROTOCOL,
    BIO_MODULE_REPLICATION,

    /* Perception modules (0x0700 - 0x07FF) */
    BIO_MODULE_VISUAL_CORTEX = 0x0700,
    BIO_MODULE_AUDIO_CORTEX,
    BIO_MODULE_SPEECH_CORTEX,
    BIO_MODULE_MULTIMODAL,

    /* Language modules (0x0800 - 0x08FF) */
    BIO_MODULE_NLP = 0x0800,
    BIO_MODULE_BROCA,
    BIO_MODULE_WERNICKE,
    BIO_MODULE_ANGULAR_GYRUS,
    BIO_MODULE_ARCUATE_FASCICULUS,

    /* Swarm modules (0x0B00 - 0x0BFF) */
    BIO_MODULE_SWARM_ENERGY_GOSSIP = 0x0B00,
    BIO_MODULE_SWARM_CASCADE,
    BIO_MODULE_SWARM_PROPRIOCEPTION,
    BIO_MODULE_SWARM_MEMORY,
    BIO_MODULE_SWARM_QUORUM,
    BIO_MODULE_SWARM_FLOCKING,
    BIO_MODULE_SWARM_IMMUNE,
    BIO_MODULE_SWARM_MORPHOGENESIS,
    BIO_MODULE_SWARM_MULTI,
    BIO_MODULE_SWARM_PHEROMONE,

    /* Portia modules (0x0C00 - 0x0CFF) */
    BIO_MODULE_PORTIA = 0x0C00,
    BIO_MODULE_PORTIA_PLANNING,
    BIO_MODULE_PORTIA_SENSOR_FUSION,
    BIO_MODULE_PORTIA_TIER_SWITCH,
    BIO_MODULE_PORTIA_LEARNING,
    BIO_MODULE_PORTIA_DEGRADATION,
    BIO_MODULE_PORTIA_ATTENTION,
    BIO_MODULE_PORTIA_POWER,
    BIO_MODULE_PORTIA_ACCELERATOR,
    BIO_MODULE_PORTIA_CLASSIFICATION,
    BIO_MODULE_PORTIA_DECEPTION,

    /* Special values */
    BIO_MODULE_ALL = 0xFFFF,        /**< Broadcast to all modules */

    BIO_MODULE_COUNT
} bio_module_id_t;

/*=============================================================================
 * BRAIN MESSAGES
 *============================================================================*/

/**
 * @brief Query brain state (via acetylcholine - fast query)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t query_flags;           /**< What state to query */
    uint32_t region_id;             /**< Specific region (0 = global) */
} bio_msg_brain_state_query_t;

#define BIO_BRAIN_QUERY_NEURON_COUNT    (1 << 0)
#define BIO_BRAIN_QUERY_SYNAPSE_COUNT   (1 << 1)
#define BIO_BRAIN_QUERY_ACTIVE_REGIONS  (1 << 2)
#define BIO_BRAIN_QUERY_NEUROMODULATORS (1 << 3)
#define BIO_BRAIN_QUERY_ENERGY_STATE    (1 << 4)

/**
 * @brief Brain state response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t neuron_count;
    uint32_t synapse_count;
    uint32_t active_region_count;
    float dopamine_level;
    float serotonin_level;
    float norepinephrine_level;
    float acetylcholine_level;
    float energy_level;
    float global_activity;
} bio_msg_brain_state_response_t;

/**
 * @brief Request neuron activation
 */
typedef struct {
    bio_message_header_t header;
    uint32_t neuron_id;
    float input_current;
    float duration_ms;
} bio_msg_neuron_activation_request_t;

/**
 * @brief Neuron activation response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t neuron_id;
    float membrane_potential;
    bool spiked;
    float spike_time_ms;
} bio_msg_neuron_activation_response_t;

/**
 * @brief Region configuration query
 */
typedef struct {
    bio_message_header_t header;
    uint32_t region_id;
    uint32_t query_flags;        /**< What information to query */
} bio_msg_region_config_query_t;

/**
 * @brief Region configuration response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t region_id;
    uint32_t neuron_count;
    uint32_t synapse_count;
    uint32_t active_region_count;
    float dopamine_level;
    float serotonin_level;
    float norepinephrine_level;
    float acetylcholine_level;
} bio_msg_region_config_response_t;

/*=============================================================================
 * LOGIC MESSAGES
 *============================================================================*/

/**
 * @brief Request logic gate evaluation
 */
typedef struct {
    bio_message_header_t header;
    uint32_t gate_id;               /**< Gate neuron ID */
    uint32_t gate_type;             /**< Logic gate type (AND/OR/NOT/XOR/IMPLIES) */
    float input_a;                  /**< Input A value [0,1] */
    float input_b;                  /**< Input B value [0,1] */
    float threshold;                /**< Custom threshold (0 = use default) */
} bio_msg_logic_gate_evaluate_t;

/**
 * @brief Logic gate evaluation result
 */
typedef struct {
    bio_message_header_t header;
    uint32_t gate_id;               /**< Gate neuron ID */
    uint32_t gate_type;             /**< Logic gate type */
    float output;                   /**< Output value [0,1] */
    bool spiked;                    /**< Whether gate spiked */
    uint64_t spike_time_us;         /**< Spike timestamp */
    float threshold_used;           /**< Threshold that was used */
} bio_msg_logic_gate_result_t;

/**
 * @brief Logic circuit simulation step request
 */
typedef struct {
    bio_message_header_t header;
    uint64_t timestamp_us;          /**< Current simulation time */
    uint64_t delta_t_us;            /**< Timestep duration */
    uint32_t max_iterations;        /**< Max iterations (0 = until stable) */
} bio_msg_logic_circuit_step_t;

/**
 * @brief Logic circuit step completion notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t timestamp_us;          /**< Completion timestamp */
    uint32_t spikes_generated;      /**< Number of gate spikes */
    uint32_t gates_evaluated;       /**< Number of gates updated */
    float avg_eval_time_us;         /**< Average evaluation time */
    bool circuit_stable;            /**< Whether outputs stabilized */
} bio_msg_logic_circuit_complete_t;

/**
 * @brief Logic spike event (broadcast when gate fires)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t gate_id;               /**< Firing gate ID */
    uint32_t gate_type;             /**< Gate type */
    float output;                   /**< Output value */
    uint64_t spike_time_us;         /**< Spike timestamp */
    uint32_t propagation_count;     /**< Number of downstream targets */
} bio_msg_logic_spike_event_t;

/*=============================================================================
 * PLASTICITY MESSAGES
 *============================================================================*/

/**
 * @brief Request weight update (via dopamine - reward signal)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t synapse_id;
    uint32_t pre_neuron_id;
    uint32_t post_neuron_id;
    float weight_delta;
    float learning_rate;
    float eligibility_trace;
    bool clamp_to_bounds;
    float min_weight;
    float max_weight;
} bio_msg_weight_update_request_t;

/**
 * @brief Weight update response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t synapse_id;
    float old_weight;
    float new_weight;
    bool clamped;
    nimcp_error_t error;
} bio_msg_weight_update_response_t;

/**
 * @brief STDP spike timing event
 */
typedef struct {
    bio_message_header_t header;
    uint32_t pre_neuron_id;
    uint32_t post_neuron_id;
    float pre_spike_time_ms;
    float post_spike_time_ms;
    float delta_t_ms;               /**< post - pre timing */
} bio_msg_stdp_event_t;

/**
 * @brief Batched STDP events for efficiency
 */
typedef struct {
    bio_message_header_t header;
    uint32_t event_count;
    uint32_t max_events;
    /* Followed by event_count * bio_msg_stdp_event_t */
} bio_msg_stdp_batch_t;

/**
 * @brief Learning rate update request/response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t synapse_id;
    float base_learning_rate;
    float modulated_learning_rate;
    float dopamine_level;
    float serotonin_level;
} bio_msg_learning_rate_update_t;

/**
 * @brief Neuromodulator release event
 */
typedef struct {
    bio_message_header_t header;
    nimcp_bio_channel_type_t neuromodulator;
    uint32_t source_region;
    float release_amount;
    float current_concentration;
    float diffusion_radius_um;
} bio_msg_neuromodulator_release_t;

/**
 * @brief Eligibility trace update event
 */
typedef struct {
    bio_message_header_t header;
    uint32_t synapse_id;
    float trace_value;
    float reward_signal;
    float dopamine_level;
    uint64_t update_time_us;
} bio_msg_eligibility_trace_update_t;

/*=============================================================================
 * COGNITIVE MESSAGES
 *============================================================================*/

/**
 * @brief Introspection query (via acetylcholine - attention)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t query_type;
    uint32_t target_pattern_id;
    float confidence_threshold;
} bio_msg_introspection_query_t;

#define BIO_INTRO_QUERY_PATTERN_MATCH   0x01
#define BIO_INTRO_QUERY_SELF_STATE      0x02
#define BIO_INTRO_QUERY_COGNITIVE_LOAD  0x03
#define BIO_INTRO_QUERY_EMOTIONAL_STATE 0x04

/**
 * @brief Introspection response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t query_type;
    float confidence;
    float cognitive_load;
    float emotional_valence;
    float arousal;
    uint32_t matched_pattern_count;
} bio_msg_introspection_response_t;

/**
 * @brief Ethics evaluation request (via serotonin - deliberative)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t action_id;
    uint32_t context_id;
    float urgency;
    uint32_t stakeholder_count;
    /* Followed by stakeholder impact data */
} bio_msg_ethics_request_t;

/**
 * @brief Ethics evaluation response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t action_id;
    float ethical_score;            /**< [-1, 1] negative=harmful, positive=beneficial */
    float confidence;
    bool veto;                      /**< true = action should be blocked */
    uint32_t primary_concern;       /**< Ethics framework triggered */
    char explanation[256];
} bio_msg_ethics_response_t;

/**
 * @brief Salience evaluation (via norepinephrine - alerting)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t stimulus_id;
    float raw_intensity;
    float novelty;
    float relevance;
} bio_msg_salience_query_t;

/**
 * @brief Salience response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t stimulus_id;
    float salience_score;           /**< [0, 1] combined salience */
    float attention_priority;
    bool requires_immediate_attention;
} bio_msg_salience_response_t;

/**
 * @brief Attention shift command
 */
typedef struct {
    bio_message_header_t header;
    uint32_t target_id;
    float attention_weight;
    uint32_t duration_ms;
    bool preemptive;
} bio_msg_attention_shift_t;

/**
 * @brief Working memory store request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t slot_id;
    uint32_t data_size;
    float priority;
    uint32_t decay_ms;
    /* Followed by data */
} bio_msg_wm_store_t;

/**
 * @brief Working memory retrieve request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t slot_id;
    float min_confidence;
} bio_msg_wm_retrieve_t;

/**
 * @brief Curiosity signal (intrinsic motivation)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t target_id;
    float curiosity_intensity;
    float information_gain_estimate;
    float exploration_bonus;
} bio_msg_curiosity_signal_t;

/**
 * @brief Knowledge query request
 */
typedef struct {
    bio_message_header_t header;
    char query_str[256];            /**< Query string */
    uint32_t max_results;           /**< Maximum results to return */
    float min_confidence;           /**< Minimum confidence threshold */
} bio_msg_knowledge_query_t;

/**
 * @brief Knowledge query response
 */
typedef struct {
    bio_message_header_t header;
    bool success;                   /**< Query succeeded */
    uint32_t num_matches;           /**< Number of matches found */
    char matches[10][256];          /**< Up to 10 matching results */
    float confidence[10];           /**< Confidence for each match */
} bio_msg_knowledge_response_t;

/**
 * @brief Decision request
 */
typedef struct {
    bio_message_header_t header;
    char decision_context[256];     /**< Context for the decision */
    uint32_t option_count;          /**< Number of options */
    float urgency;                  /**< How urgent the decision is */
} bio_msg_decision_request_t;

/**
 * @brief Decision response
 */
typedef struct {
    bio_message_header_t header;
    bool approved;                  /**< Decision approved */
    float confidence;               /**< Confidence in decision */
    char reasoning[512];            /**< Explanation of decision */
    uint32_t selected_option;       /**< Selected option index */
} bio_msg_decision_response_t;

/*=============================================================================
 * GLIAL MESSAGES
 *============================================================================*/

/**
 * @brief Astrocyte calcium wave initiation
 */
typedef struct {
    bio_message_header_t header;
    uint32_t source_region;
    float initial_calcium_um;
    float propagation_speed_um_s;
    nimcp_wave_mode_t mode;
} bio_msg_astrocyte_wave_t;

/**
 * @brief Microglia alert (immune/maintenance)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t alert_region;
    uint32_t alert_type;
    float severity;
    uint32_t affected_synapse_count;
} bio_msg_microglia_alert_t;

#define BIO_MICROGLIA_ALERT_DAMAGE      0x01
#define BIO_MICROGLIA_ALERT_INFECTION   0x02
#define BIO_MICROGLIA_ALERT_PRUNE_NEEDED 0x03
#define BIO_MICROGLIA_ALERT_DEBRIS      0x04

/**
 * @brief Oligodendrocyte myelination request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t axon_id;
    float target_thickness;
    float priority;
    uint32_t segment_start;
    uint32_t segment_length;
} bio_msg_oligodendrocyte_myelinate_t;

/**
 * @brief Metabolic demand signal
 */
typedef struct {
    bio_message_header_t header;
    uint32_t region_id;
    float glucose_demand;
    float oxygen_demand;
    float atp_deficit;
    float urgency;
} bio_msg_metabolic_demand_t;

/*=============================================================================
 * MIDDLEWARE MESSAGES
 *============================================================================*/

/**
 * @brief Pipeline stage completion
 */
typedef struct {
    bio_message_header_t header;
    uint32_t pipeline_id;
    uint32_t stage_id;
    float processing_time_ms;
    nimcp_error_t status;
    uint32_t output_size;
} bio_msg_pipeline_complete_t;

/**
 * @brief Pipeline error notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t pipeline_id;
    uint32_t stage_id;
    uint32_t error_code;
    char error_message[64];
} bio_msg_pipeline_error_t;

/**
 * @brief Gradient computation result
 */
typedef struct {
    bio_message_header_t header;
    uint32_t layer_id;
    float gradient_norm;
    uint32_t num_parameters;
    bool gradient_clipped;
} bio_msg_gradient_computed_t;

/**
 * @brief Encoding request (spike encoding)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t encoding_type;
    uint32_t input_size;
    uint32_t output_neurons;
    float time_window_ms;
    /* Followed by input data */
} bio_msg_encoding_request_t;

#define BIO_ENCODING_RATE       0x01
#define BIO_ENCODING_TEMPORAL   0x02
#define BIO_ENCODING_POPULATION 0x03
#define BIO_ENCODING_RANK_ORDER 0x04

/**
 * @brief Signal routing request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t source_id;
    uint32_t target_id;
    uint32_t signal_type;
    float bandwidth_required;
    uint32_t latency_max_ms;
} bio_msg_route_request_t;

/*=============================================================================
 * TRAINING MESSAGES
 *============================================================================*/

/**
 * @brief Training step request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t batch_id;
    uint32_t batch_size;
    float learning_rate;
    uint32_t optimizer_type;
} bio_msg_training_step_t;

/**
 * @brief Loss computation result
 */
typedef struct {
    bio_message_header_t header;
    uint32_t batch_id;
    float loss_value;
    float loss_gradient;
    uint32_t loss_type;
} bio_msg_loss_computed_t;

/**
 * @brief Checkpoint request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t checkpoint_id;
    char path[256];
    bool include_optimizer_state;
    bool compress;
} bio_msg_checkpoint_request_t;

/**
 * @brief Optimizer step notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t step_number;           /**< Current step number */
    float learning_rate;            /**< Current learning rate */
    float gradient_norm;            /**< Gradient L2 norm */
    uint32_t weight_updates;        /**< Number of weights updated */
} bio_msg_optimizer_step_t;

/**
 * @brief Generic training metric
 */
typedef struct {
    bio_message_header_t header;
    uint32_t metric_type;           /**< Type of metric (0=loss, 1=accuracy, 2=grad_norm, 3=lr) */
    float metric_value;             /**< Metric value */
    uint32_t step_number;           /**< Associated step */
    char metric_name[32];           /**< Optional name */
} bio_msg_training_metric_t;

/*=============================================================================
 * SYSTEM MESSAGES
 *============================================================================*/

/**
 * @brief Health check request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t check_flags;
} bio_msg_health_check_t;

#define BIO_HEALTH_CHECK_MEMORY     (1 << 0)
#define BIO_HEALTH_CHECK_CPU        (1 << 1)
#define BIO_HEALTH_CHECK_THREADS    (1 << 2)
#define BIO_HEALTH_CHECK_QUEUES     (1 << 3)
#define BIO_HEALTH_CHECK_ALL        0xFFFFFFFF

/**
 * @brief Health check response
 */
typedef struct {
    bio_message_header_t header;
    bool healthy;
    float memory_usage_percent;
    float cpu_usage_percent;
    uint32_t active_threads;
    uint32_t pending_messages;
    float avg_latency_ms;
} bio_msg_health_response_t;

/**
 * @brief Error report
 */
typedef struct {
    bio_message_header_t header;
    nimcp_error_t error_code;
    uint32_t severity;
    uint32_t source_line;
    char source_file[64];
    char message[256];
} bio_msg_error_report_t;

#define BIO_ERROR_SEVERITY_DEBUG    0
#define BIO_ERROR_SEVERITY_INFO     1
#define BIO_ERROR_SEVERITY_WARNING  2
#define BIO_ERROR_SEVERITY_ERROR    3
#define BIO_ERROR_SEVERITY_CRITICAL 4

/*=============================================================================
 * PERCEPTION MESSAGES
 *============================================================================*/

/**
 * @brief Visual feature detection notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t feature_id;            /**< Detected feature type */
    float x_position;               /**< X position in visual field (normalized) */
    float y_position;               /**< Y position in visual field (normalized) */
    float confidence;               /**< Detection confidence (0-1) */
    float salience;                 /**< Feature salience (0-1) */
    uint32_t layer;                 /**< Visual processing layer (V1, V2, etc.) */
} bio_msg_visual_feature_detected_t;

/**
 * @brief Visual attention shift request
 */
typedef struct {
    bio_message_header_t header;
    float target_x;                 /**< Target X position */
    float target_y;                 /**< Target Y position */
    float urgency;                  /**< Shift urgency (0-1) */
    uint32_t reason;                /**< Reason for attention shift */
} bio_msg_visual_attention_shift_t;

/**
 * @brief Audio feature detection notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t feature_id;            /**< Feature type (formant, pitch, etc.) */
    float frequency_hz;             /**< Center frequency */
    float amplitude;                /**< Normalized amplitude */
    float onset_time_ms;            /**< Feature onset time */
    float duration_ms;              /**< Feature duration */
    uint8_t channel;                /**< Audio channel (left=0, right=1) */
} bio_msg_audio_feature_detected_t;

/**
 * @brief Phoneme recognition notification
 */
typedef struct {
    bio_message_header_t header;
    uint8_t phoneme_id;             /**< IPA phoneme identifier */
    char phoneme_symbol[8];         /**< IPA symbol (UTF-8) */
    float confidence;               /**< Recognition confidence */
    float onset_time_ms;            /**< Phoneme onset in audio stream */
    float duration_ms;              /**< Phoneme duration */
    float formants[4];              /**< Formant frequencies (F1-F4) */
} bio_msg_phoneme_recognized_t;

/**
 * @brief Word recognition notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t word_id;               /**< Lexicon word ID */
    char word[32];                  /**< Recognized word string */
    float confidence;               /**< Recognition confidence */
    float onset_time_ms;            /**< Word onset time */
    float duration_ms;              /**< Word duration */
    uint8_t phoneme_count;          /**< Number of phonemes */
    uint8_t phonemes[16];           /**< Phoneme sequence */
} bio_msg_word_recognized_t;

/**
 * @brief Multimodal binding event
 */
typedef struct {
    bio_message_header_t header;
    uint32_t binding_id;            /**< Unique binding identifier */
    uint32_t visual_feature_id;     /**< Associated visual feature */
    uint32_t audio_feature_id;      /**< Associated audio feature */
    float binding_strength;         /**< Binding strength (0-1) */
    float temporal_alignment;       /**< Temporal alignment quality */
} bio_msg_multimodal_binding_t;

/*=============================================================================
 * LANGUAGE PRODUCTION MESSAGES (Broca's Region)
 *============================================================================*/

/**
 * @brief Lexical access request (word lookup)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t word_id;               /**< Word ID (0 = use string) */
    char word[32];                  /**< Word to look up */
    uint8_t pos_hint;               /**< Part of speech hint */
    uint32_t context_id;            /**< Syntactic context */
} bio_msg_lexical_access_request_t;

/**
 * @brief Lexical access response
 */
typedef struct {
    bio_message_header_t header;
    uint32_t word_id;               /**< Found word ID */
    bool found;                     /**< Whether word was found */
    uint8_t phonemes[16];           /**< Phoneme sequence */
    uint8_t phoneme_count;          /**< Number of phonemes */
    uint8_t pos;                    /**< Part of speech */
    float frequency;                /**< Usage frequency */
    float activation;               /**< Lexical activation level */
} bio_msg_lexical_access_response_t;

/**
 * @brief Syntax parsing request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t word_ids[16];          /**< Word sequence to parse */
    uint8_t word_count;             /**< Number of words */
    uint8_t parse_mode;             /**< Parse mode (incremental/full) */
} bio_msg_syntax_parse_request_t;

/**
 * @brief Syntax parse result
 */
typedef struct {
    bio_message_header_t header;
    bool valid;                     /**< Parse successful */
    uint8_t structure_type;         /**< Sentence structure type */
    uint8_t constituent_count;      /**< Number of constituents */
    float complexity;               /**< Syntactic complexity score */
    uint8_t error_position;         /**< Position of error (if any) */
    uint8_t error_type;             /**< Type of syntax error */
} bio_msg_syntax_parse_result_t;

/**
 * @brief Phonological encoding request
 */
typedef struct {
    bio_message_header_t header;
    uint8_t phonemes[32];           /**< Input phoneme sequence */
    uint8_t phoneme_count;          /**< Number of phonemes */
    float speaking_rate;            /**< Target speaking rate */
    bool enable_coarticulation;     /**< Apply coarticulation */
    bool enable_prosody;            /**< Apply prosodic patterns */
} bio_msg_phonological_encode_request_t;

/**
 * @brief Phonological encoding result
 */
typedef struct {
    bio_message_header_t header;
    uint8_t syllables[16];          /**< Syllable structure */
    uint8_t syllable_count;         /**< Number of syllables */
    float durations[32];            /**< Phoneme durations (ms) */
    float pitches[32];              /**< Pitch contour (F0) */
    float intensities[32];          /**< Intensity contour */
    bool success;                   /**< Encoding successful */
} bio_msg_phonological_encode_result_t;

/**
 * @brief Motor command request
 */
typedef struct {
    bio_message_header_t header;
    uint8_t phoneme;                /**< Target phoneme */
    float duration_ms;              /**< Target duration */
    float pitch_hz;                 /**< Target pitch */
    float intensity;                /**< Target intensity */
    uint8_t coarticulation_context; /**< Surrounding phoneme context */
} bio_msg_motor_command_request_t;

/**
 * @brief Motor command result (articulator positions)
 */
typedef struct {
    bio_message_header_t header;
    float lip_aperture;             /**< Lip opening (0-1) */
    float lip_protrusion;           /**< Lip protrusion (0-1) */
    float tongue_height;            /**< Tongue height (0-1) */
    float tongue_advance;           /**< Tongue frontness (0-1) */
    float jaw_opening;              /**< Jaw opening (0-1) */
    float velum_opening;            /**< Velum opening for nasality (0-1) */
    float larynx_tension;           /**< Vocal fold tension (0-1) */
    float timestamp_ms;             /**< Command timestamp */
} bio_msg_motor_command_result_t;

/**
 * @brief Full utterance production request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t word_ids[16];          /**< Words to produce */
    uint8_t word_count;             /**< Number of words */
    float speaking_rate;            /**< Target speaking rate */
    uint8_t emotion;                /**< Emotional coloring */
    uint8_t emphasis_mask;          /**< Word emphasis flags */
} bio_msg_utterance_production_request_t;

/**
 * @brief Utterance production complete notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t utterance_id;          /**< Unique utterance identifier */
    bool success;                   /**< Production successful */
    float duration_ms;              /**< Total duration */
    uint32_t command_count;         /**< Number of motor commands */
    float avg_confidence;           /**< Average processing confidence */
} bio_msg_utterance_production_complete_t;

/**
 * @brief Speech feedback (for self-monitoring)
 */
typedef struct {
    bio_message_header_t header;
    uint32_t utterance_id;          /**< Related utterance */
    uint8_t phoneme_produced;       /**< What was produced */
    uint8_t phoneme_intended;       /**< What was intended */
    float error_magnitude;          /**< Error size */
    bool requires_correction;       /**< Needs speech repair */
} bio_msg_speech_feedback_t;

/*=============================================================================
 * NEURAL LINK PROTOCOL (NLP) NETWORKING MESSAGES
 *============================================================================*/

/**
 * @brief NLP session connected notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Connected peer ID */
    uint64_t session_id;            /**< Session identifier */
    uint8_t protocol_mode;          /**< 0=Standard, 1=Tactical, 2=Stealth */
    uint8_t encryption_level;       /**< Encryption strength level */
    bool authenticated;             /**< Whether peer is authenticated */
} bio_msg_nlp_session_connected_t;

/**
 * @brief NLP session disconnected notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Disconnected peer ID */
    uint64_t session_id;            /**< Session identifier */
    uint8_t reason;                 /**< Disconnect reason code */
    bool graceful;                  /**< Graceful vs forced disconnect */
} bio_msg_nlp_session_disconnected_t;

/**
 * @brief NLP session state change notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Peer ID */
    uint64_t session_id;            /**< Session identifier */
    uint8_t old_state;              /**< Previous session state */
    uint8_t new_state;              /**< New session state */
    uint64_t state_change_time_us;  /**< When state changed */
} bio_msg_nlp_session_state_change_t;

/**
 * @brief NLP message received notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Sending peer ID */
    uint32_t message_type;          /**< NLP message type */
    uint32_t message_size;          /**< Size of received message */
    uint32_t sequence_num;          /**< NLP sequence number */
    bool encrypted;                 /**< Was message encrypted */
    bool compressed;                /**< Was message compressed */
} bio_msg_nlp_message_received_t;

/**
 * @brief NLP message sent notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Destination peer ID */
    uint32_t message_type;          /**< NLP message type */
    uint32_t message_size;          /**< Size of sent message */
    uint32_t sequence_num;          /**< NLP sequence number */
    bool encrypted;                 /**< Was message encrypted */
    bool reliable;                  /**< Requires acknowledgment */
} bio_msg_nlp_message_sent_t;

/**
 * @brief NLP crypto operation request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t operation_id;          /**< Operation identifier */
    uint32_t data_size;             /**< Size of data to process */
    uint8_t algorithm;              /**< Crypto algorithm (AES-GCM, ChaCha20, etc.) */
    uint8_t key_id;                 /**< Key identifier */
    bool async;                     /**< Asynchronous operation */
} bio_msg_nlp_crypto_request_t;

/**
 * @brief NLP crypto operation complete notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t operation_id;          /**< Completed operation ID */
    uint32_t output_size;           /**< Size of output data */
    bool success;                   /**< Operation successful */
    uint8_t error_code;             /**< Error code if failed */
    float processing_time_us;       /**< Time taken */
} bio_msg_nlp_crypto_complete_t;

/**
 * @brief NLP crypto error notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Related peer (if applicable) */
    uint32_t operation_id;          /**< Failed operation ID */
    uint8_t error_code;             /**< Crypto error code */
    char error_message[64];         /**< Error description */
    bool is_security_violation;     /**< Possible attack detected */
} bio_msg_nlp_crypto_error_t;

/**
 * @brief NLP compression request/complete notification
 */
typedef struct {
    bio_message_header_t header;
    uint32_t operation_id;          /**< Operation identifier */
    uint32_t input_size;            /**< Original size */
    uint32_t output_size;           /**< Compressed/decompressed size */
    uint8_t algorithm;              /**< Compression algorithm */
    float compression_ratio;        /**< Achieved ratio */
    bool success;                   /**< Operation successful */
} bio_msg_nlp_compression_t;

/**
 * @brief NLP protocol mode change notification
 */
typedef struct {
    bio_message_header_t header;
    uint8_t old_mode;               /**< Previous mode (0=Standard, 1=Tactical, 2=Stealth) */
    uint8_t new_mode;               /**< New mode */
    uint64_t peer_id;               /**< Affected peer (0 = all) */
    uint32_t reason;                /**< Reason for mode change */
} bio_msg_nlp_mode_change_t;

/**
 * @brief NLP neural encoding/decoding request
 */
typedef struct {
    bio_message_header_t header;
    uint32_t context_id;            /**< Neural language context ID */
    uint32_t data_size;             /**< Data size to encode/decode */
    uint8_t encoding_type;          /**< Encoding type */
    bool bidirectional;             /**< Two-way encoding */
} bio_msg_nlp_neural_request_t;

/**
 * @brief NLP neural encoding/decoding complete
 */
typedef struct {
    bio_message_header_t header;
    uint32_t context_id;            /**< Neural language context ID */
    uint32_t output_size;           /**< Output size */
    float confidence;               /**< Encoding confidence */
    bool success;                   /**< Operation successful */
    uint32_t neurons_activated;     /**< Neural representation size */
} bio_msg_nlp_neural_complete_t;

/**
 * @brief NLP protocol bridge event
 */
typedef struct {
    bio_message_header_t header;
    uint32_t bridge_id;             /**< Bridge identifier */
    uint8_t event_type;             /**< 0=created, 1=message, 2=destroyed */
    uint32_t source_protocol;       /**< Source protocol ID */
    uint32_t target_protocol;       /**< Target protocol ID */
    uint32_t messages_bridged;      /**< Messages bridged count */
} bio_msg_nlp_bridge_event_t;

/**
 * @brief NLP general error notification
 */
typedef struct {
    bio_message_header_t header;
    uint64_t peer_id;               /**< Related peer (0 if N/A) */
    uint32_t error_code;            /**< Error code */
    uint8_t severity;               /**< 0=info, 1=warning, 2=error, 3=critical */
    char module_name[32];           /**< Module that generated error */
    char error_message[128];        /**< Error description */
} bio_msg_nlp_error_t;

/*=============================================================================
 * MESSAGE UTILITIES
 *============================================================================*/

/**
 * @brief Initialize message header with defaults
 */
static inline void bio_msg_init_header(bio_message_header_t* header,
                                        bio_message_type_t type,
                                        bio_module_id_t source,
                                        bio_module_id_t target,
                                        size_t payload_size) {
    header->type = type;
    header->sequence_id = 0;  /* Set by sender */
    header->source_module = source;
    header->target_module = target;
    header->timestamp_us = 0;  /* Set by sender */
    header->channel = BIO_CHANNEL_DOPAMINE;  /* Default */
    header->payload_size = (uint32_t)payload_size;
    header->flags = 0;
}

/**
 * @brief Get recommended channel for message type
 */
static inline nimcp_bio_channel_type_t bio_msg_recommended_channel(bio_message_type_t type) {
    /* Plasticity/reward messages → Dopamine */
    if (type >= 0x0200 && type < 0x0300) {
        return BIO_CHANNEL_DOPAMINE;
    }
    /* Ethics/slow cognitive → Serotonin */
    if (type == BIO_MSG_ETHICS_EVALUATION_REQUEST ||
        type == BIO_MSG_ETHICS_EVALUATION_RESPONSE ||
        type == BIO_MSG_CONSOLIDATION_TRIGGER) {
        return BIO_CHANNEL_SEROTONIN;
    }
    /* Alerts/salience → Norepinephrine */
    if (type == BIO_MSG_SALIENCE_QUERY ||
        type == BIO_MSG_SALIENCE_RESPONSE ||
        type == BIO_MSG_MICROGLIA_ALERT ||
        type == BIO_MSG_ERROR_REPORT) {
        return BIO_CHANNEL_NOREPINEPHRINE;
    }
    /* Fast queries/attention → Acetylcholine */
    if (type == BIO_MSG_BRAIN_STATE_QUERY ||
        type == BIO_MSG_INTROSPECTION_QUERY ||
        type == BIO_MSG_ATTENTION_SHIFT ||
        type == BIO_MSG_WORKING_MEMORY_RETRIEVE) {
        return BIO_CHANNEL_ACETYLCHOLINE;
    }
    /* Default to dopamine */
    return BIO_CHANNEL_DOPAMINE;
}

/**
 * @brief Get message type name for logging
 */
const char* bio_msg_type_name(bio_message_type_t type);

/**
 * @brief Get module name for logging
 */
const char* bio_module_name(bio_module_id_t module);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_BIO_MESSAGES_H */
