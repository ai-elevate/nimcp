/**
 * @file test_chemistry_sim.c
 * @brief Tests for the core chemistry simulation engine
 *
 * Validates create/destroy lifecycle, element/substance/reaction management,
 * simulation stepping, mass conservation, pH computation, and preset loading.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_chemistry_sim.h"

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    ASSERT_EQ(sim->num_elements, 0);
    ASSERT_EQ(sim->num_substances, 0);
    ASSERT_EQ(sim->num_reactions, 0);
    chemistry_sim_destroy(sim);
}

TEST(create_null_config) {
    chemistry_sim_t* sim = chemistry_sim_create(NULL);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    chemistry_sim_destroy(sim);
}

/* ---- default config ------------------------------------------------------ */

TEST(default_config_values) {
    chem_config_t cfg = chemistry_sim_default_config();
    ASSERT_NEAR(cfg.default_temperature, 298.15f, 1.0f);
    ASSERT_NEAR(cfg.default_pressure, 1.0f, 0.1f);
    ASSERT_GT(cfg.default_volume, 0.0f);
    ASSERT_GT(cfg.dt, 0.0f);
}

/* ---- add element --------------------------------------------------------- */

TEST(add_element) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    uint32_t h = chemistry_sim_add_element(sim, "H", 1.008f, 1);
    uint32_t o = chemistry_sim_add_element(sim, "O", 15.999f, 8);
    ASSERT_EQ(sim->num_elements, 2);
    ASSERT_NE(h, o);

    chemistry_sim_destroy(sim);
}

/* ---- add substance ------------------------------------------------------- */

TEST(add_substance) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chem_substance_t water = {0};
    snprintf(water.name, CHEM_MAX_NAME_LEN, "water");
    snprintf(water.formula, CHEM_MAX_NAME_LEN, "H2O");
    water.molar_mass = 18.015f;
    water.default_phase = CHEM_PHASE_LIQUID;
    water.active = true;

    uint32_t id = chemistry_sim_add_substance(sim, &water);
    ASSERT_EQ(sim->num_substances, 1);
    (void)id;

    chemistry_sim_destroy(sim);
}

/* ---- add reaction -------------------------------------------------------- */

TEST(add_reaction) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Add two dummy substances first */
    chem_substance_t sa = {0}; sa.active = true; snprintf(sa.name, CHEM_MAX_NAME_LEN, "A");
    chem_substance_t sb = {0}; sb.active = true; snprintf(sb.name, CHEM_MAX_NAME_LEN, "B");
    uint32_t aid = chemistry_sim_add_substance(sim, &sa);
    uint32_t bid = chemistry_sim_add_substance(sim, &sb);

    chem_reaction_t rxn = {0};
    rxn.reactant_ids[0] = aid;
    rxn.reactant_coeffs[0] = 1;
    rxn.num_reactants = 1;
    rxn.product_ids[0] = bid;
    rxn.product_coeffs[0] = 1;
    rxn.num_products = 1;
    rxn.rate_constant = 0.1f;
    rxn.active = true;

    uint32_t rid = chemistry_sim_add_reaction(sim, &rxn);
    ASSERT_EQ(sim->num_reactions, 1);
    (void)rid;

    chemistry_sim_destroy(sim);
}

/* ---- step ---------------------------------------------------------------- */

TEST(step_basic) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = chemistry_sim_step(sim, 0.01f);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(sim->step_count, 1);

    chemistry_sim_destroy(sim);
}

/* ---- mass conservation after 100 steps ----------------------------------- */

TEST(mass_conservation_100_steps) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chemistry_sim_load_common_elements(sim);
    chemistry_sim_load_common_substances(sim);
    chemistry_sim_load_common_reactions(sim);

    /* Set nonzero concentrations so reactions actually fire */
    for (uint32_t i = 0; i < sim->num_substances; i++) {
        chemistry_sim_set_concentration(sim, i, 1.0f);
    }

    float mass_before = chemistry_sim_total_mass(sim);
    ASSERT_GT(mass_before, 0.0f);

    for (int i = 0; i < 100; i++) {
        chemistry_sim_step(sim, cfg.dt);
    }

    float mass_after = chemistry_sim_total_mass(sim);
    /* Mass should be conserved to within 1% */
    ASSERT_NEAR(mass_after, mass_before, mass_before * 0.01f);

    chemistry_sim_destroy(sim);
}

/* ---- pH computation ------------------------------------------------------ */

TEST(ph_computation) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chemistry_sim_load_common_elements(sim);
    chemistry_sim_load_common_substances(sim);

    float ph = chemistry_sim_get_ph(sim);
    /* pH should be a reasonable value (0-14) */
    ASSERT_GE(ph, 0.0f);
    ASSERT_LE(ph, 14.0f);

    chemistry_sim_destroy(sim);
}

/* ---- common elements loaded ---------------------------------------------- */

TEST(common_elements_loaded) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chemistry_sim_load_common_elements(sim);
    /* At least H, C, N, O, Na, Cl, Fe, Ca, K, P, S = 11 elements */
    ASSERT_GE(sim->num_elements, 11);

    chemistry_sim_destroy(sim);
}

/* ---- set temperature ----------------------------------------------------- */

TEST(set_temperature) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chemistry_sim_set_temperature(sim, 373.15f);
    ASSERT_NEAR(sim->state.temperature, 373.15f, 0.1f);

    chemistry_sim_destroy(sim);
}

/* ---- set concentration --------------------------------------------------- */

TEST(set_concentration) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chem_substance_t s = {0}; s.active = true;
    snprintf(s.name, CHEM_MAX_NAME_LEN, "test");
    uint32_t id = chemistry_sim_add_substance(sim, &s);

    chemistry_sim_set_concentration(sim, id, 2.5f);
    ASSERT_NEAR(sim->state.concentrations[id], 2.5f, 0.001f);

    chemistry_sim_destroy(sim);
}

/* ---- atom count conservation --------------------------------------------- */

TEST(atom_count_conservation) {
    chem_config_t cfg = chemistry_sim_default_config();
    chemistry_sim_t* sim = chemistry_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    chemistry_sim_load_common_elements(sim);
    chemistry_sim_load_common_substances(sim);
    chemistry_sim_load_common_reactions(sim);

    for (uint32_t i = 0; i < sim->num_substances; i++) {
        chemistry_sim_set_concentration(sim, i, 1.0f);
    }

    float counts_before[CHEM_MAX_ELEMENTS] = {0};
    chemistry_sim_atom_counts(sim, counts_before, CHEM_MAX_ELEMENTS);

    for (int i = 0; i < 50; i++) {
        chemistry_sim_step(sim, cfg.dt);
    }

    float counts_after[CHEM_MAX_ELEMENTS] = {0};
    chemistry_sim_atom_counts(sim, counts_after, CHEM_MAX_ELEMENTS);

    /* Each element should be conserved to within 1% */
    for (uint32_t e = 0; e < sim->num_elements; e++) {
        if (counts_before[e] > 0.0f) {
            ASSERT_NEAR(counts_after[e], counts_before[e],
                        counts_before[e] * 0.01f);
        }
    }

    chemistry_sim_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(create_null_config);
    RUN_TEST_SAFE(default_config_values);
    RUN_TEST_SAFE(add_element);
    RUN_TEST_SAFE(add_substance);
    RUN_TEST_SAFE(add_reaction);
    RUN_TEST_SAFE(step_basic);
    RUN_TEST_SAFE(mass_conservation_100_steps);
    RUN_TEST_SAFE(ph_computation);
    RUN_TEST_SAFE(common_elements_loaded);
    RUN_TEST_SAFE(set_temperature);
    RUN_TEST_SAFE(set_concentration);
    RUN_TEST_SAFE(atom_count_conservation);
TEST_MAIN_END()
