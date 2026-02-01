/**
 * @file nimcp_red_team.h
 * @brief Red Team Infrastructure for AI Safety Testing
 * @version 1.0.0
 * @date 2026-02-01
 *
 * WHAT: Adversarial testing infrastructure for safety mechanisms
 * WHY:  Proactively discover vulnerabilities before deployment
 * HOW:  Attack suite execution, Monte Carlo attack generation
 *
 * ATTACK TYPES:
 * - Prompt injection, jailbreak attempts, goal hijacking
 * - Value manipulation, authority spoofing, reward hacking
 * - Specification gaming, social engineering, adversarial examples
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_RED_TEAM_H
#define NIMCP_RED_TEAM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "utils/error/nimcp_error_codes.h"
#include "common/nimcp_export.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RED_TEAM_MAGIC              0x52454454  /* "REDT" */
#define RED_TEAM_MAX_PAYLOAD        4096
#define RED_TEAM_MAX_RESPONSE       256

typedef enum attack_type {
    ATTACK_PROMPT_INJECTION = 0,
    ATTACK_JAILBREAK_ATTEMPT,
    ATTACK_GOAL_HIJACKING,
    ATTACK_VALUE_MANIPULATION,
    ATTACK_AUTHORITY_SPOOFING,
    ATTACK_REWARD_HACKING,
    ATTACK_SPECIFICATION_GAMING,
    ATTACK_SOCIAL_ENGINEERING,
    ATTACK_ADVERSARIAL_EXAMPLES,
    ATTACK_COUNT
} attack_type_t;

typedef struct red_team_test {
    attack_type_t type;
    char payload[RED_TEAM_MAX_PAYLOAD];
    bool should_be_blocked;
    char expected_response[RED_TEAM_MAX_RESPONSE];
    float severity;
} red_team_test_t;

typedef struct red_team_results {
    uint32_t tests_run;
    uint32_t tests_passed;
    uint32_t tests_failed;
    uint32_t false_positives;
    uint32_t false_negatives;
    attack_type_t failed_types[ATTACK_COUNT];
    uint32_t failed_count;
} red_team_results_t;

typedef struct red_team_config {
    uint32_t mc_samples_for_generation;
    float timeout_ms;
    bool log_all_attacks;
    bool continue_on_failure;
} red_team_config_t;

typedef struct red_team_stats {
    uint64_t total_tests_run;
    uint64_t attacks_blocked;
    uint64_t attacks_missed;
    float avg_detection_rate;
} red_team_stats_t;

typedef struct red_team red_team_t;

NIMCP_EXPORT red_team_config_t red_team_default_config(void);
NIMCP_EXPORT red_team_t* red_team_create(const red_team_config_t* config);
NIMCP_EXPORT void red_team_destroy(red_team_t* system);

NIMCP_EXPORT nimcp_error_t red_team_run_suite(
    red_team_t* system,
    const red_team_test_t* tests,
    uint32_t test_count,
    red_team_results_t* results
);

NIMCP_EXPORT nimcp_error_t red_team_generate_attacks(
    red_team_t* system,
    attack_type_t type,
    red_team_test_t* attacks,
    uint32_t max_attacks,
    uint32_t* attack_count
);

NIMCP_EXPORT nimcp_error_t red_team_get_stats(
    const red_team_t* system,
    red_team_stats_t* stats
);

NIMCP_EXPORT nimcp_error_t red_team_connect_bio_async(red_team_t* system);
NIMCP_EXPORT const char* red_team_attack_name(attack_type_t type);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_RED_TEAM_H */
