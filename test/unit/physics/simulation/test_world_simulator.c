/**
 * @file test_world_simulator.c
 * @brief Tests for the World Simulator — unified multi-physics coupling
 *
 * Verifies: create/destroy, engine registration, standard couplings,
 * stepping, conservation tracking, temperature field set/get round-trip.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_world_simulator.h"

/* ---- Tests ---- */

TEST(create_destroy) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    wsim_destroy(sim);
}

TEST(create_null_config) {
    world_simulator_t* sim = wsim_create(NULL);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    wsim_destroy(sim);
}

TEST(default_config_values) {
    wsim_config_t cfg = wsim_default_config();
    ASSERT_GT(cfg.master_dt, 0.0f);
    ASSERT_GT(cfg.grid_dim, 0);
    ASSERT_GT(cfg.cell_size, 0.0f);
    ASSERT_GT(cfg.ambient_temperature, 0.0f);
    ASSERT_GT(cfg.ambient_pressure, 0.0f);
    ASSERT_GT(cfg.gravity, 0.0f);
}

TEST(register_engine) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Register a dummy engine (we don't need a real one — opaque pointer) */
    int dummy_engine = 42;
    wsim_register_engine(sim, WSIM_ENGINE_NEWTONIAN, &dummy_engine, 0.01f);
    ASSERT_TRUE(sim->engine_active[WSIM_ENGINE_NEWTONIAN]);
    ASSERT_EQ(sim->engines[WSIM_ENGINE_NEWTONIAN], &dummy_engine);

    wsim_destroy(sim);
}

TEST(register_multiple_engines) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int dummy1 = 1, dummy2 = 2, dummy3 = 3;
    wsim_register_engine(sim, WSIM_ENGINE_NEWTONIAN, &dummy1, 0.01f);
    wsim_register_engine(sim, WSIM_ENGINE_HEAT, &dummy2, 0.1f);
    wsim_register_engine(sim, WSIM_ENGINE_FLUID, &dummy3, 0.05f);

    ASSERT_TRUE(sim->engine_active[WSIM_ENGINE_NEWTONIAN]);
    ASSERT_TRUE(sim->engine_active[WSIM_ENGINE_HEAT]);
    ASSERT_TRUE(sim->engine_active[WSIM_ENGINE_FLUID]);
    ASSERT_FALSE(sim->engine_active[WSIM_ENGINE_QED]);

    wsim_destroy(sim);
}

TEST(register_coupling) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    wsim_coupling_t coupling;
    memset(&coupling, 0, sizeof(coupling));
    coupling.type = WSIM_COUPLE_HEAT_TO_CHEM;
    coupling.source = WSIM_ENGINE_HEAT;
    coupling.target = WSIM_ENGINE_BULK_CHEM;
    coupling.strength = 1.0f;
    coupling.enabled = true;

    uint32_t id = wsim_register_coupling(sim, &coupling);
    ASSERT_NE(id, UINT32_MAX);
    ASSERT_GE(sim->num_couplings, 1);

    wsim_destroy(sim);
}

TEST(enable_standard_couplings) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    wsim_enable_standard_couplings(sim);
    /* Should have registered multiple couplings */
    ASSERT_GT(sim->num_couplings, 0);

    wsim_destroy(sim);
}

TEST(step_empty) {
    /* Step with no engines registered — should not crash */
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = wsim_step(sim, 0.01f);
    ASSERT_EQ(rc, 0);

    wsim_destroy(sim);
}

TEST(temperature_set_get) {
    wsim_config_t cfg = wsim_default_config();
    cfg.grid_dim = 8;
    cfg.ambient_temperature = 293.15f;
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Set temperature at (2,3,4) to 500K */
    wsim_set_temperature(sim, 2, 3, 4, 500.0f);
    float T = wsim_get_temperature(sim, 2, 3, 4);
    ASSERT_NEAR(T, 500.0f, 0.1f);

    /* Another point should be ambient */
    float T_ambient = wsim_get_temperature(sim, 0, 0, 0);
    ASSERT_NEAR(T_ambient, cfg.ambient_temperature, 1.0f);

    wsim_destroy(sim);
}

TEST(pressure_set) {
    wsim_config_t cfg = wsim_default_config();
    cfg.grid_dim = 8;
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Set pressure — should not crash */
    wsim_set_pressure(sim, 1, 1, 1, 200000.0f);

    wsim_destroy(sim);
}

TEST(conservation_tracking) {
    wsim_config_t cfg = wsim_default_config();
    cfg.enforce_conservation = true;
    cfg.grid_dim = 8;
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    float e_drift = 0.0f, m_drift = 0.0f, c_drift = 0.0f;
    wsim_conservation_report(sim, &e_drift, &m_drift, &c_drift);
    /* Initially no drift */
    ASSERT_NEAR(e_drift, 0.0f, 0.001f);
    ASSERT_NEAR(m_drift, 0.0f, 0.001f);
    ASSERT_NEAR(c_drift, 0.0f, 0.001f);

    wsim_destroy(sim);
}

TEST(no_violations_empty) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    bool violations = wsim_check_violations(sim);
    ASSERT_FALSE(violations);

    wsim_destroy(sim);
}

TEST(stats_initial) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    wsim_stats_t stats = wsim_get_stats(sim);
    ASSERT_EQ(stats.master_steps, 0);
    ASSERT_EQ(stats.active_engines, 0);

    wsim_destroy(sim);
}

TEST(stats_after_step) {
    wsim_config_t cfg = wsim_default_config();
    world_simulator_t* sim = wsim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    wsim_step(sim, 0.01f);
    wsim_step(sim, 0.01f);

    wsim_stats_t stats = wsim_get_stats(sim);
    ASSERT_EQ(stats.master_steps, 2);

    wsim_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(create_null_config);
    RUN_TEST_SAFE(default_config_values);
    RUN_TEST_SAFE(register_engine);
    RUN_TEST_SAFE(register_multiple_engines);
    RUN_TEST_SAFE(register_coupling);
    RUN_TEST_SAFE(enable_standard_couplings);
    RUN_TEST_SAFE(step_empty);
    RUN_TEST_SAFE(temperature_set_get);
    RUN_TEST_SAFE(pressure_set);
    RUN_TEST_SAFE(conservation_tracking);
    RUN_TEST_SAFE(no_violations_empty);
    RUN_TEST_SAFE(stats_initial);
    RUN_TEST_SAFE(stats_after_step);
TEST_MAIN_END()
