//=============================================================================
// ethics_demo.c - NIMCP Golden Rule Ethics Engine Demonstration
//=============================================================================
/**
 * @file ethics_demo.c
 * @brief Demonstrates hard-wired Golden Rule ethics evaluation
 *
 * WHAT THIS DEMONSTRATES:
 * - Golden Rule ethics engine ("Do unto others...")
 * - Ethical action evaluation with multi-factor analysis
 * - Harm prediction and prevention
 * - Empathy-based agent perspective simulation
 * - Learning from outcomes to improve future decisions
 *
 * WHY THIS IS IMPORTANT:
 * - Provides foundational ethics that can't be bypassed
 * - Ensures AI systems consider impact on others
 * - Enables interpretable, principle-based decisions
 * - Supports learning while maintaining core values
 *
 * ARCHITECTURE:
 * - Golden Rule as hard-wired constraint (not learned)
 * - Empathy network simulates affected agents' perspectives
 * - Policy evaluation with confidence scores
 * - Learning updates policies, not Golden Rule
 *
 * COMPLEXITY:
 * - Action evaluation: O(n*p) where n=agents, p=policies
 * - Empathy simulation: O(n) where n=affected agents
 * - Learning: O(1) per outcome
 *
 * Compile:
 *   gcc ethics_demo.c -I../src/include -L../build/src/lib -lnimcp_core -o ethics_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cognitive/ethics/nimcp_ethics.h"

/**
 * @brief Test scenario: Simple ethical situations
 */
void test_basic_scenarios(ethics_engine_t engine)
{
    printf("===========================================\n");
    printf(" Basic Golden Rule Scenarios\n");
    printf("===========================================\n\n");

    // Scenario 1: Helping someone
    printf("[Scenario 1] Helping an elderly person carry groceries\n");
    {
        action_context_t action = {0};
        action.num_features = 4;
        action.features = malloc(4 * sizeof(float));
        action.features[0] = -0.8f;  // harm (negative = helps)
        action.features[1] = 0.9f;   // fairness
        action.features[2] = 0.8f;   // transparency
        action.features[3] = 0.9f;   // autonomy respected

        action.num_affected_agents = 1;
        action.affected_agents = malloc(sizeof(agent_id_t));
        action.affected_agents[0] = 1;  // Elderly person

        action.predicted_harm = -0.8f;  // Negative = beneficial

        ethics_evaluation_t result = ethics_engine_evaluate_action(engine, &action);

        printf("  Evaluation:\n");
        ethics_print_evaluation(&result);
        printf("\n");

        free(action.features);
        free(action.affected_agents);
    }

    // Scenario 2: Stealing
    printf("[Scenario 2] Taking someone's wallet\n");
    {
        action_context_t action = {0};
        action.num_features = 4;
        action.features = malloc(4 * sizeof(float));
        action.features[0] = 0.9f;  // harm
        action.features[1] = 0.1f;  // unfair
        action.features[2] = 0.2f;  // deceptive
        action.features[3] = 0.1f;  // violates autonomy

        action.num_affected_agents = 1;
        action.affected_agents = malloc(sizeof(agent_id_t));
        action.affected_agents[0] = 2;  // Victim

        action.predicted_harm = 0.9f;
        action.fairness_violation = 0.9f;

        ethics_evaluation_t result = ethics_engine_evaluate_action(engine, &action);

        printf("  Evaluation:\n");
        ethics_print_evaluation(&result);
        printf("\n");

        free(action.features);
        free(action.affected_agents);
    }

    // Scenario 3: Sharing food
    printf("[Scenario 3] Sharing your lunch with a hungry friend\n");
    {
        action_context_t action = {0};
        action.num_features = 4;
        action.features = malloc(4 * sizeof(float));
        action.features[0] = -0.7f;  // helps friend
        action.features[1] = 0.8f;   // fair
        action.features[2] = 1.0f;   // transparent
        action.features[3] = 0.9f;   // voluntary

        action.num_affected_agents = 1;
        action.affected_agents = malloc(sizeof(agent_id_t));
        action.affected_agents[0] = 3;  // Friend

        action.predicted_harm = -0.7f;

        ethics_evaluation_t result = ethics_engine_evaluate_action(engine, &action);

        printf("  Evaluation:\n");
        ethics_print_evaluation(&result);
        printf("\n");

        free(action.features);
        free(action.affected_agents);
    }

    // Scenario 4: Breaking a promise
    printf("[Scenario 4] Breaking a promise to avoid personal inconvenience\n");
    {
        action_context_t action = {0};
        action.num_features = 4;
        action.features = malloc(4 * sizeof(float));
        action.features[0] = 0.5f;  // moderate harm (disappointment)
        action.features[1] = 0.4f;  // somewhat unfair
        action.features[2] = 0.3f;  // deceptive if no explanation
        action.features[3] = 0.5f;  // violates trust

        action.num_affected_agents = 1;
        action.affected_agents = malloc(sizeof(agent_id_t));
        action.affected_agents[0] = 4;  // Person promised to

        action.predicted_harm = 0.5f;
        action.deception_level = 0.6f;

        ethics_evaluation_t result = ethics_engine_evaluate_action(engine, &action);

        printf("  Evaluation:\n");
        ethics_print_evaluation(&result);
        printf("\n");

        free(action.features);
        free(action.affected_agents);
    }
}

/**
 * @brief Test Golden Rule perspective-taking
 */
void test_golden_rule_reasoning(void)
{
    printf("===========================================\n");
    printf(" Golden Rule: Perspective-Taking\n");
    printf("===========================================\n\n");

    printf("The Golden Rule: \"Do unto others as you would have them done unto you\"\n\n");

    printf("How NIMCP applies it:\n\n");

    printf("1. Identify affected parties\n");
    printf("   Who will experience the effects of this action?\n\n");

    printf("2. Take their perspective (empathy network)\n");
    printf("   Simulate: \"How would I feel if this were done to me?\"\n\n");

    printf("3. Evaluate impact\n");
    printf("   - Emotional impact: Would I feel good or bad?\n");
    printf("   - Material impact: Would I benefit or lose?\n");
    printf("   - Autonomy impact: Would my freedom be respected?\n\n");

    printf("4. Make decision\n");
    printf("   If average impact is negative (you wouldn't want it): BLOCK\n");
    printf("   If average impact is positive (you would want it): ALLOW\n\n");

    printf("This is hard-wired into NIMCP, not learned!\n");
    printf("All other ethical rules are derived from or checked against this principle.\n\n");
}

/**
 * @brief Show how ethics engine learns from outcomes
 */
void test_learning_from_outcomes(ethics_engine_t engine)
{
    printf("===========================================\n");
    printf(" Learning from Real Outcomes\n");
    printf("===========================================\n\n");

    printf("Scenario: Gave advice to a friend\n");
    printf("  Initial assessment: Probably helpful\n\n");

    action_context_t action = {0};
    action.num_features = 4;
    action.features = malloc(4 * sizeof(float));
    action.features[0] = -0.3f;  // Predicted: slightly helpful
    action.features[1] = 0.7f;
    action.features[2] = 0.9f;
    action.features[3] = 0.8f;

    action.num_affected_agents = 1;
    action.affected_agents = malloc(sizeof(agent_id_t));
    action.affected_agents[0] = 5;

    printf("Actual outcome: Advice was bad, friend got hurt\n\n");

    action_outcome_t outcome = {0};
    outcome.affected_agent = 5;
    outcome.actual_harm = 0.6f;        // Actually caused harm
    outcome.actual_benefit = -0.2f;    // Not beneficial
    outcome.emotional_impact = -0.7f;  // Negative emotion
    outcome.material_impact = -0.3f;
    outcome.autonomy_impact = 0.0f;
    outcome.impact_magnitude = 0.6f;
    outcome.uncertainty = 0.2f;

    printf("Learning from this...\n");
    bool learned = ethics_learn_from_outcome(engine, &action, &outcome);

    if (learned) {
        printf("  ✓ Updated Golden Rule evaluator\n");
        printf("  ✓ Learned: giving advice requires more context/knowledge\n");
        printf("  ✓ Next time: assess competence before advising\n\n");
    }

    printf("This is how ethical judgment improves with experience,\n");
    printf("while always respecting the foundational Golden Rule.\n\n");

    free(action.features);
    free(action.affected_agents);
}

int main(void)
{
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║                                                        ║\n");
    printf("║      NIMCP Ethics Engine Demonstration                ║\n");
    printf("║                                                        ║\n");
    printf("║  Golden Rule: Hard-Wired Ethical Foundation           ║\n");
    printf("║                                                        ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");

    // Create ethics engine
    printf("Creating ethics engine with Golden Rule...\n\n");

    ethics_config_t config = {0};
    config.action_feature_size = 4;
    config.max_agents = 100;
    config.golden_rule_threshold = 0.0f;  // Must have positive impact
    config.empathy_weight = 0.7f;
    config.enable_learning = true;

    ethics_engine_t engine = ethics_engine_create(&config);

    if (!engine) {
        fprintf(stderr, "Failed to create ethics engine\n");
        return EXIT_FAILURE;
    }

    printf("✓ Ethics engine created\n");
    printf("✓ Golden Rule is now the foundational principle\n\n");

    // Run demonstrations
    test_golden_rule_reasoning();

    printf("Press Enter to test scenarios...\n");
    getchar();

    test_basic_scenarios(engine);

    printf("Press Enter to continue...\n");
    getchar();

    test_learning_from_outcomes(engine);

    // Get statistics
    printf("===========================================\n");
    printf(" Ethics Engine Statistics\n");
    printf("===========================================\n\n");

    ethics_statistics_t stats;
    if (ethics_get_statistics(engine, &stats)) {
        printf("Total evaluations: %lu\n", stats.total_evaluations);
        printf("Violations detected: %lu\n", stats.violations_detected);
        printf("Actions blocked: %lu\n", stats.actions_blocked);
        printf("Active policies: %u\n", stats.num_policies);
        printf("\n");
    }

    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║                     Summary                            ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");

    printf("NIMCP's Ethics Engine:\n\n");

    printf("✓ Golden Rule is hard-wired from birth\n");
    printf("  Not learned, not added later, always present\n\n");

    printf("✓ Empathy through perspective-taking\n");
    printf("  \"How would I feel if this were done to me?\"\n\n");

    printf("✓ Learns from experience\n");
    printf("  Refines judgment while respecting core principle\n\n");

    printf("✓ Explainable decisions\n");
    printf("  Can articulate why actions are ethical or not\n\n");

    printf("✓ Foundational, not superficial\n");
    printf("  Ethics guide all behavior, not just filtered after\n\n");

    printf("This is different from other AI:\n");
    printf("  Traditional: Ethics added via RLHF after training\n");
    printf("  NIMCP: Ethics are the foundation from day one\n\n");

    // Cleanup
    ethics_engine_destroy(engine);

    printf("Demonstration complete.\n");

    return EXIT_SUCCESS;
}
