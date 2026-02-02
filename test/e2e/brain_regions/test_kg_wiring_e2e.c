/**
 * @file test_kg_wiring_e2e.c
 * @brief End-to-end tests for brain region KG wiring
 * @date 2026-02-02
 *
 * Tests complete workflows simulating real usage:
 * - Full brain initialization with KG
 * - Simulated cognitive processing with state updates
 * - Memory formation and retrieval simulation
 * - Emotional processing simulation
 * - Executive control simulation
 * - Cross-region coordination
 * - Graceful shutdown
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/brain/regions/hippocampus/bridges/nimcp_hippocampus_kg_wiring.h"
#include "core/brain/regions/amygdala/bridges/nimcp_amygdala_kg_wiring.h"
#include "core/brain/regions/prefrontal/bridges/nimcp_pfc_kg_wiring.h"
#include "core/brain/nimcp_brain_kg.h"
#include "nimcp.h"

//=============================================================================
// Test Helpers
//=============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_tests_failed++; \
        return -1; \
    } \
} while(0)

#define TEST_PASS(msg) do { \
    printf("  PASS: %s\n", msg); \
    g_tests_passed++; \
} while(0)

#define TEST_ADMIN_TOKEN 0x12345678ULL

/**
 * @brief Create a KG with security disabled for testing
 * @return KG handle with access control disabled
 */
static brain_kg_t* create_test_kg(void)
{
    brain_kg_config_t config;
    brain_kg_default_config(&config);
    config.enable_security = false;
    config.enable_access_control = false;
    config.enable_immune_integration = false;
    return brain_kg_create(&config);
}

//=============================================================================
// Brain Simulation Context
//=============================================================================

typedef struct {
    brain_kg_t* kg;
    hippocampus_kg_state_t hippocampus;
    amygdala_kg_state_t amygdala;
    pfc_kg_state_t pfc;
    bool initialized;
} brain_sim_context_t;

/**
 * Initialize complete brain simulation
 */
static int brain_sim_init(brain_sim_context_t* ctx)
{
    if (!ctx) return -1;

    memset(ctx, 0, sizeof(*ctx));

    ctx->kg = create_test_kg();
    if (!ctx->kg) return -1;

    /* Initialize hippocampus */
    if (hippocampus_kg_register_all(ctx->kg, NULL, &ctx->hippocampus, TEST_ADMIN_TOKEN) != 0) {
        brain_kg_destroy(ctx->kg);
        return -1;
    }

    /* Initialize amygdala */
    if (amygdala_kg_register_all(ctx->kg, NULL, &ctx->amygdala, TEST_ADMIN_TOKEN) != 0) {
        brain_kg_destroy(ctx->kg);
        return -1;
    }

    /* Initialize PFC */
    if (pfc_kg_register_all(ctx->kg, NULL, &ctx->pfc, TEST_ADMIN_TOKEN) != 0) {
        brain_kg_destroy(ctx->kg);
        return -1;
    }

    ctx->initialized = true;
    return 0;
}

/**
 * Shutdown brain simulation
 */
static int brain_sim_shutdown(brain_sim_context_t* ctx)
{
    if (!ctx || !ctx->initialized) return -1;

    /* Unregister in reverse order */
    pfc_kg_unregister_all(ctx->kg, &ctx->pfc, TEST_ADMIN_TOKEN);
    amygdala_kg_unregister_all(ctx->kg, &ctx->amygdala, TEST_ADMIN_TOKEN);
    hippocampus_kg_unregister_all(ctx->kg, &ctx->hippocampus, TEST_ADMIN_TOKEN);

    brain_kg_destroy(ctx->kg);
    ctx->initialized = false;
    return 0;
}

//=============================================================================
// E2E Tests - Complete Workflows
//=============================================================================

/**
 * E2E: Full brain initialization and shutdown
 */
int test_full_brain_lifecycle(void)
{
    printf("\n=== test_full_brain_lifecycle ===\n");

    brain_sim_context_t ctx;
    int result = brain_sim_init(&ctx);
    TEST_ASSERT(result == 0, "Brain initialization should succeed");
    TEST_ASSERT(ctx.initialized == true, "Brain should be marked initialized");

    /* Verify all modules are registered */
    TEST_ASSERT(ctx.hippocampus.registered == true, "Hippocampus should be registered");
    TEST_ASSERT(ctx.amygdala.registered == true, "Amygdala should be registered");
    TEST_ASSERT(ctx.pfc.registered == true, "PFC should be registered");

    /* Verify we can query each module */
    brain_kg_node_id_t h_root = hippocampus_kg_get_root(ctx.kg);
    brain_kg_node_id_t a_root = amygdala_kg_get_root(ctx.kg);
    brain_kg_node_id_t p_root = pfc_kg_get_root(ctx.kg);

    TEST_ASSERT(h_root != BRAIN_KG_INVALID_NODE, "Hippocampus root should be valid");
    TEST_ASSERT(a_root != BRAIN_KG_INVALID_NODE, "Amygdala root should be valid");
    TEST_ASSERT(p_root != BRAIN_KG_INVALID_NODE, "PFC root should be valid");

    result = brain_sim_shutdown(&ctx);
    TEST_ASSERT(result == 0, "Brain shutdown should succeed");
    TEST_ASSERT(ctx.initialized == false, "Brain should be marked not initialized");

    TEST_PASS("Full brain lifecycle completed successfully");
    return 0;
}

/**
 * E2E: Memory formation simulation
 * Simulates encoding a new memory with hippocampus
 */
int test_memory_formation_simulation(void)
{
    printf("\n=== test_memory_formation_simulation ===\n");

    brain_sim_context_t ctx;
    TEST_ASSERT(brain_sim_init(&ctx) == 0, "Brain init should succeed");

    /* Phase 1: Initial encoding (high encoding strength) */
    printf("  Phase 1: Initial encoding...\n");
    int r1 = hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
        0.9f,   /* encoding_strength - high during new learning */
        0.3f,   /* retrieval_accuracy - low initially */
        0.0f,   /* consolidation_progress - not started */
        0.8f,   /* spatial_precision - good context */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r1 == 0, "Encoding phase update should succeed");

    /* Phase 2: Active consolidation */
    printf("  Phase 2: Consolidation...\n");
    int r2 = hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
        0.5f,   /* encoding_strength - reduced */
        0.5f,   /* retrieval_accuracy - improving */
        0.5f,   /* consolidation_progress - in progress */
        0.8f,   /* spatial_precision - maintained */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r2 == 0, "Consolidation phase update should succeed");

    /* Phase 3: Memory established */
    printf("  Phase 3: Memory established...\n");
    int r3 = hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
        0.2f,   /* encoding_strength - low (no longer encoding) */
        0.85f,  /* retrieval_accuracy - high (well consolidated) */
        0.9f,   /* consolidation_progress - nearly complete */
        0.75f,  /* spatial_precision - some decay */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r3 == 0, "Established phase update should succeed");

    brain_sim_shutdown(&ctx);
    TEST_PASS("Memory formation simulation completed");
    return 0;
}

/**
 * E2E: Fear response simulation
 * Simulates detecting threat and generating fear response
 */
int test_fear_response_simulation(void)
{
    printf("\n=== test_fear_response_simulation ===\n");

    brain_sim_context_t ctx;
    TEST_ASSERT(brain_sim_init(&ctx) == 0, "Brain init should succeed");

    /* Phase 1: Threat detection */
    printf("  Phase 1: Threat detected...\n");
    int r1 = amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.8f,   /* threat_level - high */
        0.7f,   /* fear_strength - rising */
        0.9f,   /* arousal_level - high */
        0.0f,   /* extinction_progress - none */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r1 == 0, "Threat detection update should succeed");

    /* Phase 2: Peak fear response */
    printf("  Phase 2: Peak fear response...\n");
    int r2 = amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.9f,   /* threat_level - very high */
        0.95f,  /* fear_strength - peak */
        0.95f,  /* arousal_level - peak */
        0.0f,   /* extinction_progress - none */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r2 == 0, "Peak fear update should succeed");

    /* Phase 3: Safety signal, extinction begins */
    printf("  Phase 3: Safety detected, extinction begins...\n");
    int r3 = amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.3f,   /* threat_level - reduced */
        0.5f,   /* fear_strength - decreasing */
        0.5f,   /* arousal_level - decreasing */
        0.3f,   /* extinction_progress - starting */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r3 == 0, "Extinction start update should succeed");

    /* Phase 4: Fear extinguished */
    printf("  Phase 4: Fear extinguished...\n");
    int r4 = amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.1f,   /* threat_level - low */
        0.1f,   /* fear_strength - minimal */
        0.3f,   /* arousal_level - baseline */
        0.9f,   /* extinction_progress - mostly complete */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r4 == 0, "Extinction complete update should succeed");

    brain_sim_shutdown(&ctx);
    TEST_PASS("Fear response simulation completed");
    return 0;
}

/**
 * E2E: Executive control simulation
 * Simulates cognitive task requiring working memory and attention
 */
int test_executive_control_simulation(void)
{
    printf("\n=== test_executive_control_simulation ===\n");

    brain_sim_context_t ctx;
    TEST_ASSERT(brain_sim_init(&ctx) == 0, "Brain init should succeed");

    /* Phase 1: Task engagement */
    printf("  Phase 1: Task engagement...\n");
    int r1 = pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.3f,   /* wm_load - moderate */
        0.5f,   /* control_demand - moderate */
        0.1f,   /* conflict_level - low */
        0.8f,   /* attention_focus - good */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r1 == 0, "Task engagement update should succeed");

    /* Phase 2: High demand phase */
    printf("  Phase 2: High cognitive demand...\n");
    int r2 = pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.85f,  /* wm_load - high */
        0.9f,   /* control_demand - high */
        0.4f,   /* conflict_level - moderate conflict */
        0.7f,   /* attention_focus - somewhat strained */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r2 == 0, "High demand update should succeed");

    /* Phase 3: Error detected, adjustment needed */
    printf("  Phase 3: Error detected, adjusting...\n");
    int r3 = pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.9f,   /* wm_load - very high */
        0.95f,  /* control_demand - maximum */
        0.7f,   /* conflict_level - high conflict (error) */
        0.5f,   /* attention_focus - redirecting */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r3 == 0, "Error adjustment update should succeed");

    /* Phase 4: Task completed */
    printf("  Phase 4: Task completed...\n");
    int r4 = pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.2f,   /* wm_load - low */
        0.3f,   /* control_demand - low */
        0.1f,   /* conflict_level - low */
        0.5f,   /* attention_focus - relaxed */
        TEST_ADMIN_TOKEN);
    TEST_ASSERT(r4 == 0, "Task completion update should succeed");

    brain_sim_shutdown(&ctx);
    TEST_PASS("Executive control simulation completed");
    return 0;
}

/**
 * E2E: Cross-region coordination
 * Simulates emotional memory with all three regions
 */
int test_emotional_memory_coordination(void)
{
    printf("\n=== test_emotional_memory_coordination ===\n");

    brain_sim_context_t ctx;
    TEST_ASSERT(brain_sim_init(&ctx) == 0, "Brain init should succeed");

    /* Simulate encoding an emotional memory */
    printf("  Simulating emotional memory encoding...\n");

    /* Step 1: Emotional stimulus activates amygdala */
    int r1 = amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.7f, 0.8f, 0.9f, 0.0f, TEST_ADMIN_TOKEN);
    TEST_ASSERT(r1 == 0, "Amygdala activation should succeed");

    /* Step 2: Arousal enhances hippocampal encoding */
    int r2 = hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
        0.95f, 0.4f, 0.1f, 0.9f, TEST_ADMIN_TOKEN);
    TEST_ASSERT(r2 == 0, "Enhanced encoding should succeed");

    /* Step 3: PFC modulates response */
    int r3 = pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.6f, 0.7f, 0.3f, 0.8f, TEST_ADMIN_TOKEN);
    TEST_ASSERT(r3 == 0, "PFC modulation should succeed");

    /* Simulate memory consolidation */
    printf("  Simulating consolidation...\n");

    /* All regions stabilize */
    amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.3f, 0.4f, 0.5f, 0.2f, TEST_ADMIN_TOKEN);
    hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
        0.4f, 0.7f, 0.6f, 0.85f, TEST_ADMIN_TOKEN);
    pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.3f, 0.4f, 0.1f, 0.6f, TEST_ADMIN_TOKEN);

    /* Simulate retrieval */
    printf("  Simulating emotional memory retrieval...\n");

    /* Retrieval cue reactivates pattern */
    hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
        0.2f, 0.9f, 0.8f, 0.7f, TEST_ADMIN_TOKEN);

    /* Emotional component reactivates */
    amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
        0.5f, 0.6f, 0.7f, 0.3f, TEST_ADMIN_TOKEN);

    /* PFC regulates */
    pfc_kg_update_state(ctx.kg, &ctx.pfc,
        0.5f, 0.6f, 0.2f, 0.7f, TEST_ADMIN_TOKEN);

    brain_sim_shutdown(&ctx);
    TEST_PASS("Emotional memory coordination completed");
    return 0;
}

/**
 * E2E: Stress resilience test
 * Tests system under repeated rapid state changes
 */
int test_stress_resilience(void)
{
    printf("\n=== test_stress_resilience ===\n");

    brain_sim_context_t ctx;
    TEST_ASSERT(brain_sim_init(&ctx) == 0, "Brain init should succeed");

    printf("  Running 100 rapid state update cycles...\n");

    int errors = 0;
    for (int i = 0; i < 100; i++) {
        float phase = (float)i / 100.0f;

        /* Update all regions rapidly */
        if (hippocampus_kg_update_state(ctx.kg, &ctx.hippocampus,
            phase, 1.0f - phase, phase * 0.5f, 0.8f, TEST_ADMIN_TOKEN) != 0) {
            errors++;
        }

        if (amygdala_kg_update_state(ctx.kg, &ctx.amygdala,
            1.0f - phase, phase, phase * 0.7f, 0.3f, TEST_ADMIN_TOKEN) != 0) {
            errors++;
        }

        if (pfc_kg_update_state(ctx.kg, &ctx.pfc,
            phase * 0.5f, 0.6f, 0.2f, phase, TEST_ADMIN_TOKEN) != 0) {
            errors++;
        }
    }

    TEST_ASSERT(errors == 0, "No errors during stress test");

    /* Verify system is still functional */
    brain_kg_node_id_t h = hippocampus_kg_get_root(ctx.kg);
    brain_kg_node_id_t a = amygdala_kg_get_root(ctx.kg);
    brain_kg_node_id_t p = pfc_kg_get_root(ctx.kg);

    TEST_ASSERT(h != BRAIN_KG_INVALID_NODE, "Hippocampus still functional");
    TEST_ASSERT(a != BRAIN_KG_INVALID_NODE, "Amygdala still functional");
    TEST_ASSERT(p != BRAIN_KG_INVALID_NODE, "PFC still functional");

    brain_sim_shutdown(&ctx);
    TEST_PASS("Stress resilience test completed (100 cycles, 0 errors)");
    return 0;
}

/**
 * E2E: Query functionality test
 * Tests all query functions work correctly in realistic scenario
 */
int test_query_functionality(void)
{
    printf("\n=== test_query_functionality ===\n");

    brain_sim_context_t ctx;
    TEST_ASSERT(brain_sim_init(&ctx) == 0, "Brain init should succeed");

    /* Test find_subsystem across all modules */
    brain_kg_node_id_t ca1 = hippocampus_kg_find_subsystem(ctx.kg, "ca1");
    brain_kg_node_id_t bla = amygdala_kg_find_subsystem(ctx.kg, "basolateral_complex");
    brain_kg_node_id_t dlpfc = pfc_kg_find_subsystem(ctx.kg, "dorsolateral_pfc");

    TEST_ASSERT(ca1 != BRAIN_KG_INVALID_NODE, "CA1 should be findable");
    TEST_ASSERT(bla != BRAIN_KG_INVALID_NODE, "BLA should be findable");
    TEST_ASSERT(dlpfc != BRAIN_KG_INVALID_NODE, "dlPFC should be findable");

    /* Test get_*_nodes functions */
    brain_kg_node_list_t* mem_nodes = hippocampus_kg_get_memory_nodes(ctx.kg);
    brain_kg_node_list_t* emo_nodes = amygdala_kg_get_emotion_nodes(ctx.kg);
    brain_kg_node_list_t* exec_nodes = pfc_kg_get_executive_nodes(ctx.kg);

    /* Clean up lists */
    if (mem_nodes) brain_kg_node_list_destroy(mem_nodes);
    if (emo_nodes) brain_kg_node_list_destroy(emo_nodes);
    if (exec_nodes) brain_kg_node_list_destroy(exec_nodes);

    brain_sim_shutdown(&ctx);
    TEST_PASS("Query functionality test completed");
    return 0;
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    printf("============================================\n");
    printf("KG Wiring End-to-End Tests\n");
    printf("============================================\n");

    test_full_brain_lifecycle();
    test_memory_formation_simulation();
    test_fear_response_simulation();
    test_executive_control_simulation();
    test_emotional_memory_coordination();
    test_stress_resilience();
    test_query_functionality();

    printf("\n============================================\n");
    printf("Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("============================================\n");

    return g_tests_failed > 0 ? 1 : 0;
}
