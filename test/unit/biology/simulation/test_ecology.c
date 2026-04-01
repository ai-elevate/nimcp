/**
 * @file test_ecology.c
 * @brief Tests for the ecology simulation engine
 *
 * Validates Shannon index, species-area power law, nutrient cycling
 * conservation, diversity computations, food web management.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_ecology.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    ecology_config_t cfg = ecology_default_config();
    ecology_sim_t* sim = ecology_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    ecology_destroy(sim);
}

/* ---- Shannon index of uniform distribution = ln(N) ----------------------- */

TEST(shannon_uniform_distribution) {
    /* 4 species with equal abundance -> H = ln(4) ~ 1.386 */
    float abundances[] = {100.0f, 100.0f, 100.0f, 100.0f};
    float H = ecology_shannon_index(abundances, 4);
    ASSERT_NEAR(H, logf(4.0f), 0.05f);
}

TEST(shannon_single_species) {
    /* 1 species -> H = 0 (no diversity) */
    float abundances[] = {100.0f};
    float H = ecology_shannon_index(abundances, 1);
    ASSERT_NEAR(H, 0.0f, 0.01f);
}

TEST(shannon_unequal_distribution) {
    /* Unequal distribution should have lower H than equal */
    float equal[] = {100.0f, 100.0f, 100.0f};
    float unequal[] = {280.0f, 10.0f, 10.0f};
    float H_eq = ecology_shannon_index(equal, 3);
    float H_uneq = ecology_shannon_index(unequal, 3);
    ASSERT_GT(H_eq, H_uneq);
}

/* ---- Simpson index ------------------------------------------------------- */

TEST(simpson_uniform) {
    /* 4 equal species: D = 1 - 4*(0.25^2) = 1 - 0.25 = 0.75 */
    float abundances[] = {100.0f, 100.0f, 100.0f, 100.0f};
    float D = ecology_simpson_index(abundances, 4);
    ASSERT_NEAR(D, 0.75f, 0.05f);
}

/* ---- species-area power law: S = c * A^z -------------------------------- */

TEST(species_area_power_law) {
    /* c=10, A=100, z=0.25 -> S = 10 * 100^0.25 = 10 * 3.162 = 31.62 */
    float S = ecology_species_area(10.0f, 100.0f, 0.25f);
    float expected = 10.0f * powf(100.0f, 0.25f);
    ASSERT_NEAR(S, expected, 0.5f);
}

TEST(species_area_larger_area) {
    /* Larger area -> more species */
    float S1 = ecology_species_area(10.0f, 10.0f, 0.25f);
    float S2 = ecology_species_area(10.0f, 1000.0f, 0.25f);
    ASSERT_GT(S2, S1);
}

/* ---- nutrient cycling conserves total N ---------------------------------- */

TEST(nutrient_cycling_conserves_nitrogen) {
    ecology_config_t cfg = ecology_default_config();
    cfg.enable_nutrient_cycling = true;
    cfg.dt = 1.0f;
    ecology_sim_t* sim = ecology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Set initial nitrogen pools */
    sim->nutrients[ECOL_NUTRIENT_N_ATMOSPHERIC].amount = 1000.0f;
    sim->nutrients[ECOL_NUTRIENT_N_AMMONIUM].amount = 50.0f;
    sim->nutrients[ECOL_NUTRIENT_N_NITRATE].amount = 30.0f;
    sim->nutrients[ECOL_NUTRIENT_N_ORGANIC].amount = 20.0f;

    float total_N_before = 0.0f;
    for (int i = ECOL_NUTRIENT_N_ATMOSPHERIC; i <= ECOL_NUTRIENT_N_ORGANIC; i++) {
        total_N_before += sim->nutrients[i].amount;
    }

    for (int i = 0; i < 100; i++) {
        ecology_step_nitrogen_cycle(sim, 1.0f);
    }

    float total_N_after = 0.0f;
    for (int i = ECOL_NUTRIENT_N_ATMOSPHERIC; i <= ECOL_NUTRIENT_N_ORGANIC; i++) {
        total_N_after += sim->nutrients[i].amount;
    }

    /* Total N should be conserved within 1% */
    ASSERT_NEAR(total_N_after, total_N_before, total_N_before * 0.01f);

    ecology_destroy(sim);
}

/* ---- add species and link ------------------------------------------------ */

TEST(add_species_and_link) {
    ecology_config_t cfg = ecology_default_config();
    ecology_sim_t* sim = ecology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    ecol_species_t sp1 = {0};
    snprintf(sp1.name, ECOL_MAX_NAME_LEN, "grass");
    sp1.trophic_level = ECOL_TROPHIC_PRODUCER;
    sp1.abundance = 1000.0f;
    sp1.carrying_capacity = 5000.0f;
    sp1.growth_rate = 0.5f;
    sp1.active = true;
    ecology_add_species(sim, &sp1);

    ecol_species_t sp2 = {0};
    snprintf(sp2.name, ECOL_MAX_NAME_LEN, "deer");
    sp2.trophic_level = ECOL_TROPHIC_PRIMARY_CONSUMER;
    sp2.abundance = 50.0f;
    sp2.carrying_capacity = 200.0f;
    sp2.growth_rate = 0.1f;
    sp2.active = true;
    ecology_add_species(sim, &sp2);

    ecol_food_web_link_t link = {0};
    link.species_a = 1;
    link.species_b = 0;
    link.type = ECOL_INTERACTION_PREDATION;
    link.strength = 0.01f;
    ecology_add_link(sim, &link);

    ASSERT_EQ(sim->num_species, 2);
    ASSERT_EQ(sim->num_links, 1);

    ecology_destroy(sim);
}

/* ---- May's stability criterion ------------------------------------------- */

TEST(may_stability) {
    /* sqrt(S*C) * sigma < 1 for stability */
    /* S=10, C=0.1, sigma=0.1: sqrt(1)*0.1 = 0.1 < 1 (stable) */
    float m = ecology_may_stability(10, 0.1f, 0.1f);
    ASSERT_LT(m, 1.0f);

    /* S=100, C=0.5, sigma=0.5: sqrt(50)*0.5 = 3.54 > 1 (unstable) */
    float m2 = ecology_may_stability(100, 0.5f, 0.5f);
    ASSERT_GT(m2, 1.0f);
}

/* ---- load temperate forest ----------------------------------------------- */

TEST(load_temperate_forest) {
    ecology_config_t cfg = ecology_default_config();
    ecology_sim_t* sim = ecology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    ecology_load_temperate_forest(sim);
    ASSERT_GE(sim->num_species, 3);

    ecology_destroy(sim);
}

/* ---- step basic ---------------------------------------------------------- */

TEST(step_basic) {
    ecology_config_t cfg = ecology_default_config();
    ecology_sim_t* sim = ecology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = ecology_step(sim, 1.0f);
    ASSERT_EQ(rc, 0);

    ecology_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(shannon_uniform_distribution);
    RUN_TEST_SAFE(shannon_single_species);
    RUN_TEST_SAFE(shannon_unequal_distribution);
    RUN_TEST_SAFE(simpson_uniform);
    RUN_TEST_SAFE(species_area_power_law);
    RUN_TEST_SAFE(species_area_larger_area);
    RUN_TEST_SAFE(nutrient_cycling_conserves_nitrogen);
    RUN_TEST_SAFE(add_species_and_link);
    RUN_TEST_SAFE(may_stability);
    RUN_TEST_SAFE(load_temperate_forest);
    RUN_TEST_SAFE(step_basic);
TEST_MAIN_END()
