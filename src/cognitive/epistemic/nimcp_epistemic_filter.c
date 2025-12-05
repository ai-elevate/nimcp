//=============================================================================
// nimcp_epistemic_filter.c - Epistemic Hygiene Implementation
//=============================================================================

#include "cognitive/epistemic/nimcp_epistemic_filter.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"

#define LOG_MODULE "epistemic"

//=============================================================================
// Configuration Constants
//=============================================================================

#define MAX_SOURCES 256
#define MAX_CLAIM_PATTERNS 32
#define MAX_TEXT_LENGTH 100000  // Maximum text length to analyze (100KB)

// Skepticism thresholds
#define SKEPTICISM_DEFAULT 0.6f
#define SKEPTICISM_MIN 0.0f
#define SKEPTICISM_MAX 1.0f

// Conspiracy pattern weights
#define CONSPIRACY_WEIGHT_NARRATIVE 0.25f
#define CONSPIRACY_WEIGHT_UNFALSIFIABLE 0.3f
#define CONSPIRACY_WEIGHT_PATTERN_SEEKING 0.2f
#define CONSPIRACY_WEIGHT_MAINSTREAM_REJECTION 0.25f
#define CONSPIRACY_WEIGHT_AD_HOC 0.15f
#define CONSPIRACY_WEIGHT_CERTAINTY 0.1f

// Bias detection thresholds
#define BIAS_THRESHOLD_DUNNING_KRUGER 0.5f
#define BIAS_THRESHOLD_CONFIRMATION 0.5f
#define BIAS_THRESHOLD_BANDWAGON 0.5f

// Evidence weights
#define EVIDENCE_WEIGHT_SAGAN 0.35f
#define EVIDENCE_WEIGHT_SOURCE 0.25f
#define EVIDENCE_WEIGHT_CONSENSUS 0.25f
#define EVIDENCE_WEIGHT_LOGIC 0.25f

// Acceptance thresholds
#define ACCEPTANCE_THRESHOLD_DEFAULT 0.6f
#define ACCEPTANCE_THRESHOLD_EXTRAORDINARY 0.8f
#define ACCEPTANCE_THRESHOLD_IMPOSSIBLE 0.85f

/**
 * @brief Epistemic filter engine structure
 */
struct epistemic_filter_struct {
    // Configuration
    float skepticism_level;        /**< Base skepticism (0-1) */
    float consensus_weight;        /**< How much to weight consensus */
    float source_weight;           /**< How much to weight source reliability */

    // Source tracking
    source_reliability_t sources[MAX_SOURCES];
    uint32_t num_sources;

    // Statistics
    uint64_t claims_assessed;
    uint64_t claims_accepted;
    uint64_t claims_rejected;
    uint64_t biases_detected;

    // Bio-async integration
    bio_module_context_t bio_ctx;  /**< Bio-async module context */
    bool bio_async_enabled;        /**< Bio-async registration status */
};

//=============================================================================
// Conspiracy Pattern Detection
//=============================================================================

/**
 * @brief Check for conspiracy thinking patterns in text
 */
static float detect_conspiracy_patterns(const char* text) {
    if (!text) return 0.0f;

    // Validate text length
    size_t text_len = strnlen(text, MAX_TEXT_LENGTH + 1);
    if (text_len > MAX_TEXT_LENGTH) {
        return 0.0f;  // Reject suspiciously long text
    }

    float conspiracy_score = 0.0f;
    uint32_t pattern_count = 0;

    // Convert to lowercase for pattern matching (bounded)
    char* lower_text = nimcp_malloc(text_len + 1);
    if (!lower_text) return 0.0f;

    for (size_t i = 0; i < text_len; i++) {
        lower_text[i] = tolower((unsigned char)text[i]);
    }
    lower_text[text_len] = '\0';

    // Pattern 1: "They don't want you to know" narratives
    if (strstr(lower_text, "they don't want") ||
        strstr(lower_text, "they're hiding") ||
        strstr(lower_text, "cover up") ||
        strstr(lower_text, "wake up") ||
        strstr(lower_text, "do your own research")) {
        conspiracy_score += CONSPIRACY_WEIGHT_NARRATIVE;
        pattern_count++;
    }

    // Pattern 2: Unfalsifiable claims
    if (strstr(lower_text, "can't prove it didn't") ||
        strstr(lower_text, "absence of evidence") ||
        strstr(lower_text, "you can't disprove")) {
        conspiracy_score += CONSPIRACY_WEIGHT_PATTERN_SEEKING;
        pattern_count++;
    }

    // Pattern 3: Pattern-seeking ("everything is connected")
    if (strstr(lower_text, "it's all connected") ||
        strstr(lower_text, "no coincidence") ||
        strstr(lower_text, "connect the dots")) {
        conspiracy_score += CONSPIRACY_WEIGHT_PATTERN_SEEKING;
        pattern_count++;
    }

    // Pattern 4: Rejection of mainstream evidence
    if (strstr(lower_text, "mainstream media") ||
        strstr(lower_text, "msm lies") ||
        strstr(lower_text, "fake news") ||
        strstr(lower_text, "sheeple")) {
        conspiracy_score += CONSPIRACY_WEIGHT_MAINSTREAM_REJECTION;
        pattern_count++;
    }

    // Pattern 5: Ad-hoc explanations
    if (strstr(lower_text, "actually means") ||
        strstr(lower_text, "real reason") ||
        strstr(lower_text, "what they're really")) {
        conspiracy_score += CONSPIRACY_WEIGHT_AD_HOC;
        pattern_count++;
    }

    // Pattern 6: Certainty without evidence
    if (strstr(lower_text, "obviously") ||
        strstr(lower_text, "clearly") ||
        strstr(lower_text, "everyone knows")) {
        conspiracy_score += CONSPIRACY_WEIGHT_CERTAINTY;
        pattern_count++;
    }

    nimcp_free(lower_text);

    // Normalize by number of patterns found
    if (pattern_count > 0) {
        conspiracy_score = fminf(1.0f, conspiracy_score);
    }

    return conspiracy_score;
}

//=============================================================================
// Bias Detection Algorithms
//=============================================================================

/**
 * @brief Detect confirmation bias (only seeking confirming evidence)
 */
static float detect_confirmation_bias(
    const claim_evidence_t* evidence,
    float prior_belief)
{
    // High prior belief + low evidence diversity = confirmation bias
    if (prior_belief > 0.7f && evidence->num_sources < 2) {
        return 0.6f;
    }

    // Single source + high certainty = possible confirmation bias
    if (evidence->num_sources == 1 && evidence->evidence_strength > 0.8f) {
        return 0.5f;
    }

    return 0.0f;
}

/**
 * @brief Detect Dunning-Kruger effect (overconfidence with low knowledge)
 */
static float detect_dunning_kruger(
    float confidence,
    float evidence_quality)
{
    // High confidence + low evidence = Dunning-Kruger
    if (confidence > 0.8f && evidence_quality < 0.3f) {
        return 0.7f;
    }

    return 0.0f;
}

/**
 * @brief Detect bandwagon effect (following popular opinion)
 */
static float detect_bandwagon_effect(
    const claim_evidence_t* evidence)
{
    // High public consensus + low evidence = bandwagon
    if (evidence->public_consensus > 0.7f &&
        evidence->evidence_quality < EVIDENCE_MODERATE) {
        return 0.6f;
    }

    return 0.0f;
}

//=============================================================================
// API Implementation
//=============================================================================

epistemic_filter_t epistemic_filter_create(float skepticism_level) {
    epistemic_filter_t filter = nimcp_calloc(1, sizeof(struct epistemic_filter_struct));
    if (!filter) {
        return NULL;
    }

    // Set configuration
    filter->skepticism_level = fminf(fmaxf(skepticism_level, 0.0f), 1.0f);
    filter->consensus_weight = 0.3f;   // Moderate weight on consensus
    filter->source_weight = 0.4f;      // Higher weight on source reliability

    filter->num_sources = 0;
    filter->claims_assessed = 0;
    filter->claims_accepted = 0;
    filter->claims_rejected = 0;
    filter->biases_detected = 0;

    // Initialize bio-async fields
    filter->bio_ctx = NULL;
    filter->bio_async_enabled = false;

    // Register with bio-async router if available
    NIMCP_LOGGING_DEBUG("epistemic: Checking bio-async router initialization...");
    if (bio_router_is_initialized()) {
        NIMCP_LOGGING_DEBUG("epistemic: Bio-router initialized, registering module (id=%d, inbox_capacity=32)...",
                           BIO_MODULE_EPISTEMIC);
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_EPISTEMIC,
            .module_name = "epistemic",
            .inbox_capacity = 32,
            .user_data = filter
        };
        filter->bio_ctx = bio_router_register_module(&bio_info);
        if (filter->bio_ctx) {
            filter->bio_async_enabled = true;
            NIMCP_LOGGING_INFO("epistemic: Bio-async communication enabled (module_id=%d)",
                              BIO_MODULE_EPISTEMIC);
        } else {
            NIMCP_LOGGING_WARN("epistemic: Bio-async registration failed - module will operate without async messaging");
        }
    } else {
        NIMCP_LOGGING_DEBUG("epistemic: Bio-router not initialized, skipping async registration");
    }

    return filter;
}

void epistemic_filter_destroy(epistemic_filter_t filter) {
    if (!filter) return;

    // Unregister from bio-async router
    if (filter->bio_async_enabled && filter->bio_ctx) {
        bio_router_unregister_module(filter->bio_ctx);
        filter->bio_ctx = NULL;
        filter->bio_async_enabled = false;
        NIMCP_LOGGING_INFO("Bio-async communication disabled for epistemic");
    }

    nimcp_free(filter);
}

void epistemic_evidence_init(claim_evidence_t* evidence) {
    if (!evidence) return;

    memset(evidence, 0, sizeof(claim_evidence_t));

    evidence->evidence_quality = EVIDENCE_NONE;
    evidence->plausibility = PLAUSIBLE_NEUTRAL;
    evidence->evidence_strength = 0.0f;
    evidence->logical_consistency = 1.0f;  // Assume consistent unless proven otherwise
    evidence->num_sources = 0;
    evidence->source_reliability_avg = 0.5f;  // Neutral reliability
    evidence->has_primary_sources = false;
    evidence->expert_consensus = 0.5f;  // Unknown
    evidence->public_consensus = 0.5f;  // Unknown
    evidence->is_extraordinary_claim = false;
    evidence->is_falsifiable = true;  // Assume falsifiable unless proven otherwise
    evidence->has_contradictions = false;
}

void epistemic_assessment_init(epistemic_assessment_t* assessment) {
    if (!assessment) return;

    memset(assessment, 0, sizeof(epistemic_assessment_t));

    assessment->epistemic_quality = 0.5f;
    assessment->skepticism_score = 0.5f;
    assessment->credibility_score = 0.5f;
    assessment->should_accept = false;
    assessment->requires_verification = true;
    assessment->num_biases_detected = 0;
    assessment->logical_coherence = 1.0f;
    assessment->prior_compatibility = 0.5f;

    epistemic_evidence_init(&assessment->evidence);
}

float epistemic_apply_sagan_standard(
    claim_plausibility_t prior_plausibility,
    evidence_quality_t evidence_quality)
{
    // "Extraordinary claims require extraordinary evidence" - Carl Sagan

    // Map plausibility and evidence to scores
    float plausibility_score = 0.0f;
    switch (prior_plausibility) {
        case PLAUSIBLE_IMPOSSIBLE:     plausibility_score = 0.0f; break;
        case PLAUSIBLE_EXTRAORDINARY:  plausibility_score = 0.1f; break;
        case PLAUSIBLE_UNLIKELY:       plausibility_score = 0.3f; break;
        case PLAUSIBLE_NEUTRAL:        plausibility_score = 0.5f; break;
        case PLAUSIBLE_LIKELY:         plausibility_score = 0.7f; break;
        case PLAUSIBLE_ESTABLISHED:    plausibility_score = 0.9f; break;
    }

    float evidence_score = 0.0f;
    switch (evidence_quality) {
        case EVIDENCE_NONE:        evidence_score = 0.0f; break;
        case EVIDENCE_ANECDOTAL:   evidence_score = 0.2f; break;
        case EVIDENCE_WEAK:        evidence_score = 0.3f; break;
        case EVIDENCE_MODERATE:    evidence_score = 0.5f; break;
        case EVIDENCE_STRONG:      evidence_score = 0.7f; break;
        case EVIDENCE_SCIENTIFIC:  evidence_score = 0.9f; break;
        case EVIDENCE_CONSENSUS:   evidence_score = 1.0f; break;
    }

    // For extraordinary claims (low plausibility), require high evidence
    // For ordinary claims (high plausibility), lower evidence suffices
    float required_evidence = 1.0f - plausibility_score;
    float evidence_deficit = required_evidence - evidence_score;

    // Credibility decreases with evidence deficit
    float credibility = 1.0f - fmaxf(0.0f, evidence_deficit);

    // Apply penalties
    if (prior_plausibility == PLAUSIBLE_IMPOSSIBLE) {
        credibility = 0.0f;  // Impossible claims rejected
    } else if (prior_plausibility == PLAUSIBLE_EXTRAORDINARY &&
               evidence_quality < EVIDENCE_SCIENTIFIC) {
        credibility *= 0.3f;  // Extraordinary claims need extraordinary evidence
    }

    return fminf(1.0f, fmaxf(0.0f, credibility));
}

float epistemic_check_conspiracy_pattern(
    epistemic_filter_t filter,
    const char* claim_text,
    const claim_evidence_t* evidence)
{
    if (!filter || !claim_text || !evidence) {
        return 0.0f;
    }

    float conspiracy_score = 0.0f;

    // Check text patterns
    float text_score = detect_conspiracy_patterns(claim_text);
    conspiracy_score += text_score * 0.4f;

    // Check evidence patterns
    if (!evidence->is_falsifiable) {
        conspiracy_score += 0.3f;  // Unfalsifiable claims are conspiracy-like
    }

    if (evidence->num_sources == 0 || evidence->num_sources == 1) {
        conspiracy_score += 0.15f;  // Single source
    }

    if (evidence->expert_consensus < 0.2f && evidence->public_consensus > 0.5f) {
        conspiracy_score += 0.15f;  // Experts disagree but public believes
    }

    return fminf(1.0f, conspiracy_score);
}

uint32_t epistemic_detect_biases(
    epistemic_filter_t filter,
    const float* reasoning_features,
    uint32_t num_features,
    bias_detection_t* biases,
    uint32_t max_biases)
{
    if (!filter || !reasoning_features || !biases || max_biases == 0) {
        return 0;
    }

    uint32_t bias_count = 0;

    // For this implementation, we'll use some of the reasoning features
    // In a full implementation, these would be learned patterns

    // Feature interpretation (example):
    // features[0] = confidence
    // features[1] = evidence_quality
    // features[2] = prior_belief
    // features[3] = num_sources
    // features[4] = public_consensus

    if (num_features >= 3) {
        float confidence = reasoning_features[0];
        float evidence_quality = reasoning_features[1];
        float prior_belief = reasoning_features[2];

        // Build temporary evidence structure for bias detection
        claim_evidence_t temp_evidence;
        epistemic_evidence_init(&temp_evidence);
        temp_evidence.evidence_quality = (evidence_quality_t)(int)(evidence_quality * (EVIDENCE_CONSENSUS + 1));

        if (num_features >= 4) {
            temp_evidence.num_sources = (uint32_t)reasoning_features[3];
        }
        if (num_features >= 5) {
            temp_evidence.public_consensus = reasoning_features[4];
        }
        temp_evidence.evidence_strength = evidence_quality;

        // Detect Dunning-Kruger
        float dk_score = detect_dunning_kruger(confidence, evidence_quality);
        if (dk_score > 0.5f && bias_count < max_biases) {
            biases[bias_count].bias_type = BIAS_DUNNING_KRUGER;
            biases[bias_count].confidence = dk_score;
            biases[bias_count].severity = dk_score;
            snprintf(biases[bias_count].description, sizeof(biases[bias_count].description),
                     "High confidence (%.0f%%) with low evidence (%.0f%%) suggests overconfidence",
                     confidence * 100.0f, evidence_quality * 100.0f);
            bias_count++;
            filter->biases_detected++;
        }

        // Detect Confirmation Bias
        float conf_bias_score = detect_confirmation_bias(&temp_evidence, prior_belief);
        if (conf_bias_score > 0.4f && bias_count < max_biases) {
            biases[bias_count].bias_type = BIAS_CONFIRMATION;
            biases[bias_count].confidence = conf_bias_score;
            biases[bias_count].severity = conf_bias_score;
            snprintf(biases[bias_count].description, sizeof(biases[bias_count].description),
                     "High prior belief (%.0f%%) with limited sources (%d) suggests confirmation bias",
                     prior_belief * 100.0f, temp_evidence.num_sources);
            bias_count++;
            filter->biases_detected++;
        }

        // Detect Bandwagon Effect
        if (num_features >= 5) {
            float bandwagon_score = detect_bandwagon_effect(&temp_evidence);
            if (bandwagon_score > 0.4f && bias_count < max_biases) {
                biases[bias_count].bias_type = BIAS_BANDWAGON;
                biases[bias_count].confidence = bandwagon_score;
                biases[bias_count].severity = bandwagon_score;
                snprintf(biases[bias_count].description, sizeof(biases[bias_count].description),
                         "High public consensus (%.0f%%) with low evidence quality suggests bandwagon effect",
                         temp_evidence.public_consensus * 100.0f);
                bias_count++;
                filter->biases_detected++;
            }
        }
    }

    return bias_count;
}

bool epistemic_assess_claim(
    epistemic_filter_t filter,
    const char* claim_text,
    float prior_probability,
    const claim_evidence_t* evidence,
    epistemic_assessment_t* assessment)
{
    if (!filter || !claim_text || !evidence || !assessment) {
        return false;
    }

    filter->claims_assessed++;

    // Initialize assessment
    epistemic_assessment_init(assessment);
    memcpy(&assessment->evidence, evidence, sizeof(claim_evidence_t));

    // =========================================================================
    // STEP 1: Apply Sagan Standard (extraordinary claims need extraordinary evidence)
    // =========================================================================

    float sagan_credibility = epistemic_apply_sagan_standard(
        evidence->plausibility,
        evidence->evidence_quality
    );

    // =========================================================================
    // STEP 2: Check for conspiracy patterns
    // =========================================================================

    float conspiracy_score = epistemic_check_conspiracy_pattern(
        filter,
        claim_text,
        evidence
    );

    // High conspiracy score dramatically reduces credibility
    if (conspiracy_score > 0.6f) {
        sagan_credibility *= (1.0f - conspiracy_score);
    }

    // =========================================================================
    // STEP 3: Evaluate source reliability
    // =========================================================================

    float source_credibility = evidence->source_reliability_avg;

    // Bonus for multiple independent sources
    if (evidence->num_sources >= 3) {
        source_credibility = fminf(1.0f, source_credibility * 1.2f);
    }

    // Penalty for single source
    if (evidence->num_sources == 1) {
        source_credibility *= 0.7f;
    }

    // Bonus for primary sources
    if (evidence->has_primary_sources) {
        source_credibility = fminf(1.0f, source_credibility * 1.1f);
    }

    // =========================================================================
    // STEP 4: Consider expert consensus
    // =========================================================================

    float consensus_credibility = evidence->expert_consensus;

    // Strong expert consensus increases credibility
    if (evidence->expert_consensus > 0.8f) {
        consensus_credibility = fminf(1.0f, consensus_credibility * 1.15f);
    }

    // Public consensus without expert consensus is red flag
    if (evidence->public_consensus > 0.7f && evidence->expert_consensus < 0.3f) {
        consensus_credibility *= 0.5f;  // Possible bandwagon/misinformation
    }

    // =========================================================================
    // STEP 5: Check logical consistency
    // =========================================================================

    float logic_credibility = evidence->logical_consistency;

    // Contradictions severely reduce credibility
    if (evidence->has_contradictions) {
        logic_credibility *= 0.3f;
    }

    // Unfalsifiable claims are suspicious
    if (!evidence->is_falsifiable) {
        logic_credibility *= 0.5f;
    }

    // =========================================================================
    // STEP 6: Combine all factors
    // =========================================================================

    // Weighted combination
    assessment->credibility_score =
        sagan_credibility * 0.35f +              // Sagan standard
        source_credibility * filter->source_weight +  // Source reliability
        consensus_credibility * filter->consensus_weight +  // Expert consensus
        logic_credibility * 0.25f;               // Logical consistency

    // Apply skepticism modifier
    assessment->skepticism_score = filter->skepticism_level;

    // Adjust for skepticism (higher skepticism = higher bar)
    assessment->credibility_score *= (1.0f - filter->skepticism_level * 0.3f);

    // =========================================================================
    // STEP 7: Make recommendation
    // =========================================================================

    // Decision threshold depends on claim extraordinariness
    float acceptance_threshold = 0.6f;  // Default

    if (evidence->is_extraordinary_claim) {
        acceptance_threshold = 0.8f;  // Higher bar for extraordinary claims
    }

    if (evidence->plausibility == PLAUSIBLE_EXTRAORDINARY) {
        acceptance_threshold = 0.85f;
    }

    if (evidence->plausibility == PLAUSIBLE_IMPOSSIBLE) {
        assessment->should_accept = false;
        assessment->credibility_score = 0.0f;
        snprintf(assessment->reasoning, sizeof(assessment->reasoning),
                 "Claim violates established laws of nature. Rejected.");
        snprintf(assessment->recommendation, sizeof(assessment->recommendation),
                 "REJECT: Impossible claim");
        filter->claims_rejected++;
        return true;
    }

    // Make decision
    assessment->should_accept = (assessment->credibility_score >= acceptance_threshold);
    assessment->requires_verification = (assessment->credibility_score < 0.8f);

    // Calculate epistemic quality (how well-supported is this belief)
    assessment->epistemic_quality = assessment->credibility_score;
    assessment->logical_coherence = evidence->logical_consistency;
    assessment->prior_compatibility = (float)evidence->plausibility / 5.0f;

    // =========================================================================
    // STEP 8: Generate reasoning and recommendation
    // =========================================================================

    if (conspiracy_score > 0.6f) {
        snprintf(assessment->reasoning, sizeof(assessment->reasoning),
                 "Claim exhibits conspiracy-theory patterns (score: %.0f%%). "
                 "Evidence quality: %d/6. Credibility: %.0f%%",
                 conspiracy_score * 100.0f,
                 evidence->evidence_quality,
                 assessment->credibility_score * 100.0f);
    } else {
        snprintf(assessment->reasoning, sizeof(assessment->reasoning),
                 "Evidence quality: %d/6. Sources: %u. Expert consensus: %.0f%%. "
                 "Credibility: %.0f%%",
                 evidence->evidence_quality,
                 evidence->num_sources,
                 evidence->expert_consensus * 100.0f,
                 assessment->credibility_score * 100.0f);
    }

    if (assessment->should_accept) {
        snprintf(assessment->recommendation, sizeof(assessment->recommendation),
                 "ACCEPT: Sufficient evidence (%.0f%% credibility)",
                 assessment->credibility_score * 100.0f);
        filter->claims_accepted++;
    } else if (assessment->credibility_score > 0.3f) {
        snprintf(assessment->recommendation, sizeof(assessment->recommendation),
                 "UNCERTAIN: Requires verification (%.0f%% credibility)",
                 assessment->credibility_score * 100.0f);
    } else {
        snprintf(assessment->recommendation, sizeof(assessment->recommendation),
                 "REJECT: Insufficient evidence (%.0f%% credibility)",
                 assessment->credibility_score * 100.0f);
        filter->claims_rejected++;
    }

    return true;
}

bool epistemic_update_source(
    epistemic_filter_t filter,
    const char* source_id,
    bool was_correct)
{
    if (!filter || !source_id) {
        return false;
    }

    // Find existing source
    for (uint32_t i = 0; i < filter->num_sources; i++) {
        if (strcmp(filter->sources[i].source_id, source_id) == 0) {
            if (was_correct) {
                filter->sources[i].correct_count++;
            } else {
                filter->sources[i].incorrect_count++;
            }

            // Update reliability
            uint32_t total = filter->sources[i].correct_count +
                           filter->sources[i].incorrect_count +
                           filter->sources[i].unverified_count;
            if (total > 0) {
                filter->sources[i].reliability =
                    (float)filter->sources[i].correct_count / total;
            }

            return true;
        }
    }

    // Add new source
    if (filter->num_sources < MAX_SOURCES) {
        uint32_t idx = filter->num_sources++;
        strncpy(filter->sources[idx].source_id, source_id,
                sizeof(filter->sources[idx].source_id) - 1);
        filter->sources[idx].correct_count = was_correct ? 1 : 0;
        filter->sources[idx].incorrect_count = was_correct ? 0 : 1;
        filter->sources[idx].unverified_count = 0;
        filter->sources[idx].reliability = was_correct ? 1.0f : 0.0f;
        filter->sources[idx].is_primary_source = false;
        return true;
    }

    return false;
}

float epistemic_get_source_reliability(
    epistemic_filter_t filter,
    const char* source_id)
{
    if (!filter || !source_id) {
        return -1.0f;
    }

    for (uint32_t i = 0; i < filter->num_sources; i++) {
        if (strcmp(filter->sources[i].source_id, source_id) == 0) {
            return filter->sources[i].reliability;
        }
    }

    return -1.0f;  // Unknown source
}
