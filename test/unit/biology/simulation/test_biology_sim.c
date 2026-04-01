/**
 * @file test_biology_sim.c
 * @brief Tests for the core biology simulation engine
 *
 * Validates create/destroy, species management, logistic growth,
 * predator-prey oscillation stability, biodiversity index, ecosystem loading.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_biology_sim.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    bio_config_t cfg = biology_sim_default_config();
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    biology_sim_destroy(sim);
}

TEST(create_null_config) {
    biology_sim_t* sim = biology_sim_create(NULL);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    biology_sim_destroy(sim);
}

/* ---- add species --------------------------------------------------------- */

TEST(add_species) {
    bio_config_t cfg = biology_sim_default_config();
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    bio_species_t sp = {0};
    snprintf(sp.name, BIO_MAX_NAME_LEN, "rabbit");
    sp.kingdom = BIO_KINGDOM_ANIMAL;
    sp.trophic_level = BIO_TROPHIC_PRIMARY;
    sp.population = 100.0f;
    sp.carrying_capacity = 1000.0f;
    sp.growth_rate = 0.5f;
    sp.active = true;

    uint32_t id = biology_sim_add_species(sim, &sp);
    ASSERT_EQ(sim->num_species, 1);
    ASSERT_NEAR(sim->species[id].population, 100.0f, 0.1f);

    biology_sim_destroy(sim);
}

/* ---- logistic growth ----------------------------------------------------- */

TEST(logistic_growth) {
    bio_config_t cfg = biology_sim_default_config();
    cfg.dt = 0.1f;
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    bio_species_t sp = {0};
    snprintf(sp.name, BIO_MAX_NAME_LEN, "bacteria");
    sp.kingdom = BIO_KINGDOM_BACTERIA;
    sp.trophic_level = BIO_TROPHIC_DECOMPOSER;
    sp.population = 10.0f;
    sp.carrying_capacity = 1000.0f;
    sp.growth_rate = 1.0f;
    sp.active = true;
    uint32_t id = biology_sim_add_species(sim, &sp);

    /* Run for many steps -- population should approach carrying capacity */
    for (int i = 0; i < 500; i++) {
        biology_sim_step(sim, cfg.dt);
    }

    float pop = sim->species[id].population;
    /* Should be near K but not exceed it by much */
    ASSERT_GT(pop, 900.0f);
    ASSERT_LT(pop, 1100.0f);

    biology_sim_destroy(sim);
}

/* ---- population starts below K, stays bounded ---------------------------- */

TEST(population_bounded) {
    bio_config_t cfg = biology_sim_default_config();
    cfg.dt = 0.1f;
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    bio_species_t sp = {0};
    snprintf(sp.name, BIO_MAX_NAME_LEN, "elk");
    sp.population = 500.0f;
    sp.carrying_capacity = 500.0f;
    sp.growth_rate = 0.3f;
    sp.active = true;
    uint32_t id = biology_sim_add_species(sim, &sp);

    /* At K, growth should be near zero */
    for (int i = 0; i < 100; i++) {
        biology_sim_step(sim, cfg.dt);
    }

    float pop = sim->species[id].population;
    ASSERT_NEAR(pop, 500.0f, 50.0f);

    biology_sim_destroy(sim);
}

/* ---- predator-prey oscillation ------------------------------------------- */

TEST(predator_prey_oscillation) {
    bio_config_t cfg = biology_sim_default_config();
    cfg.dt = 0.01f;  /* small dt for stability */
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Prey */
    bio_species_t prey = {0};
    snprintf(prey.name, BIO_MAX_NAME_LEN, "hare");
    prey.trophic_level = BIO_TROPHIC_PRIMARY;
    prey.population = 100.0f;
    prey.carrying_capacity = 500.0f;
    prey.growth_rate = 0.5f;
    prey.active = true;
    uint32_t prey_id = biology_sim_add_species(sim, &prey);

    /* Predator */
    bio_species_t pred = {0};
    snprintf(pred.name, BIO_MAX_NAME_LEN, "lynx");
    pred.trophic_level = BIO_TROPHIC_SECONDARY;
    pred.population = 20.0f;
    pred.carrying_capacity = 100.0f;
    pred.growth_rate = 0.1f;
    pred.active = true;
    uint32_t pred_id = biology_sim_add_species(sim, &pred);

    /* Predation interaction */
    bio_interaction_t inter = {0};
    inter.species_a = pred_id;
    inter.species_b = prey_id;
    inter.type = BIO_INTERACT_PREDATION;
    inter.strength = 0.01f;
    inter.efficiency = 0.1f;
    inter.active = true;
    biology_sim_add_interaction(sim, &inter);

    /* Run for 1000 steps */
    for (int i = 0; i < 1000; i++) {
        biology_sim_step(sim, cfg.dt);
    }

    /* Neither should go to zero or infinity */
    float prey_pop = sim->species[prey_id].population;
    float pred_pop = sim->species[pred_id].population;
    ASSERT_GT(prey_pop, 0.1f);
    ASSERT_LT(prey_pop, 10000.0f);
    ASSERT_GT(pred_pop, 0.1f);
    ASSERT_LT(pred_pop, 10000.0f);

    biology_sim_destroy(sim);
}

/* ---- biodiversity index -------------------------------------------------- */

TEST(biodiversity_index) {
    bio_config_t cfg = biology_sim_default_config();
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Add 3 equal-abundance species */
    for (int i = 0; i < 3; i++) {
        bio_species_t sp = {0};
        snprintf(sp.name, BIO_MAX_NAME_LEN, "sp_%d", i);
        sp.population = 100.0f;
        sp.carrying_capacity = 1000.0f;
        sp.active = true;
        biology_sim_add_species(sim, &sp);
    }

    float H = biology_sim_biodiversity(sim);
    /* Shannon index for 3 equal species = ln(3) ~ 1.099 */
    ASSERT_NEAR(H, logf(3.0f), 0.1f);

    biology_sim_destroy(sim);
}

/* ---- total biomass ------------------------------------------------------- */

TEST(total_biomass) {
    bio_config_t cfg = biology_sim_default_config();
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    bio_species_t sp = {0};
    sp.population = 50.0f;
    sp.active = true;
    biology_sim_add_species(sim, &sp);

    float biomass = biology_sim_total_biomass(sim);
    ASSERT_GT(biomass, 0.0f);

    biology_sim_destroy(sim);
}

/* ---- load grassland ecosystem -------------------------------------------- */

TEST(load_grassland) {
    bio_config_t cfg = biology_sim_default_config();
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    biology_sim_load_grassland(sim);
    ASSERT_GE(sim->num_species, 3);
    ASSERT_GE(sim->num_interactions, 1);

    biology_sim_destroy(sim);
}

/* ---- step basic ---------------------------------------------------------- */

TEST(step_basic) {
    bio_config_t cfg = biology_sim_default_config();
    biology_sim_t* sim = biology_sim_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = biology_sim_step(sim, 0.1f);
    ASSERT_EQ(rc, 0);

    biology_sim_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(create_null_config);
    RUN_TEST_SAFE(add_species);
    RUN_TEST_SAFE(logistic_growth);
    RUN_TEST_SAFE(population_bounded);
    RUN_TEST_SAFE(predator_prey_oscillation);
    RUN_TEST_SAFE(biodiversity_index);
    RUN_TEST_SAFE(total_biomass);
    RUN_TEST_SAFE(load_grassland);
    RUN_TEST_SAFE(step_basic);
TEST_MAIN_END()
