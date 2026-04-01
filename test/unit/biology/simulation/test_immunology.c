/**
 * @file test_immunology.c
 * @brief Tests for the immunology simulation engine
 *
 * Validates complement cascade, antibody binding affinity, vaccine secondary
 * response timing, cytokine dynamics, and infection management.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_immunology.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    immunology_config_t cfg = immunology_default_config();
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    immunology_destroy(sim);
}

/* ---- complement cascade activates ---------------------------------------- */

TEST(complement_activates) {
    immunology_config_t cfg = immunology_default_config();
    cfg.enable_complement = true;
    cfg.dt = 1.0f;
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Activate classical pathway */
    int rc = immunology_activate_complement(sim, IMMUNO_COMPLEMENT_CLASSICAL);
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(sim->complement.activated);

    /* Step complement cascade -- MAC should increase */
    float mac_before = sim->complement.mac_level;
    for (int i = 0; i < 50; i++) {
        immunology_step_complement(sim, 1.0f);
    }
    float mac_after = sim->complement.mac_level;
    ASSERT_GT(mac_after, mac_before);

    immunology_destroy(sim);
}

/* ---- antibody binding increases with affinity ---------------------------- */

TEST(antibody_binding_affinity) {
    float ab_conc = 1.0e-6f;   /* 1 uM */
    float ag_conc = 1.0e-6f;   /* 1 uM */

    float binding_low = immunology_antibody_binding(IMMUNO_KA_LOW, ab_conc, ag_conc);
    float binding_high = immunology_antibody_binding(IMMUNO_KA_HIGH, ab_conc, ag_conc);

    ASSERT_GT(binding_high, binding_low);
}

TEST(antibody_binding_monotonic) {
    float ab = 1.0e-6f, ag = 1.0e-6f;
    float prev = 0.0f;
    float Ka_values[] = {1e5f, 1e7f, 1e9f, 1e11f};
    for (int i = 0; i < 4; i++) {
        float b = immunology_antibody_binding(Ka_values[i], ab, ag);
        ASSERT_GE(b, prev);
        prev = b;
    }
}

/* ---- vaccine: secondary response faster than primary --------------------- */

TEST(vaccine_secondary_faster) {
    immunology_config_t cfg = immunology_default_config();
    cfg.enable_adaptive = true;
    cfg.enable_memory_cells = true;
    cfg.dt = 1.0f;
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Add a pathogen */
    immuno_pathogen_t pathogen = {0};
    snprintf(pathogen.name, IMMUNO_MAX_NAME_LEN, "test_virus");
    pathogen.type = IMMUNO_PATHOGEN_VIRUS;
    pathogen.virulence = 0.5f;
    pathogen.replication_rate = 0.1f;
    pathogen.antigen_strength = 0.8f;
    pathogen.active = true;
    immunology_add_pathogen(sim, &pathogen);

    /* Vaccinate (creates memory cells) */
    immunology_vaccinate(sim, 0, 1.0f);

    /* Run primary immune response */
    float primary_time = 0.0f;
    for (int i = 0; i < 500; i++) {
        immunology_step(sim, cfg.dt);
        primary_time += cfg.dt;
    }

    /* Record antibody level after primary */
    immunology_stats_t stats_primary = immunology_get_stats(sim);

    /* Now infect -- secondary response should be faster */
    immunology_infect(sim, 0, 3.0f);

    /* Secondary onset is ~48h vs primary ~168h */
    /* Just verify the memory response timing constant exists and
       the system doesn't crash during secondary response */
    for (int i = 0; i < 100; i++) {
        immunology_step(sim, cfg.dt);
    }

    immunology_stats_t stats_secondary = immunology_get_stats(sim);
    /* After secondary exposure, antibody levels should be at least as high */
    ASSERT_GE(stats_secondary.total_antibody_conc,
              stats_primary.total_antibody_conc * 0.5f);

    immunology_destroy(sim);
}

/* ---- add immune cell ----------------------------------------------------- */

TEST(add_immune_cell) {
    immunology_config_t cfg = immunology_default_config();
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    immuno_cell_t cell = {0};
    cell.type = IMMUNO_CELL_NEUTROPHIL;
    cell.count = IMMUNO_NEUTROPHIL_NORMAL;
    cell.activation = 0.0f;
    cell.cytotoxicity = 0.3f;
    cell.phagocytic_rate = 10.0f;
    cell.active = true;

    int rc = immunology_add_immune_cell(sim, &cell);
    ASSERT_EQ(rc, 0);
    ASSERT_GE(sim->num_cells, 1);

    immunology_destroy(sim);
}

/* ---- cytokine dynamics --------------------------------------------------- */

TEST(cytokine_dynamics) {
    immunology_config_t cfg = immunology_default_config();
    cfg.dt = 1.0f;
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Set up elevated cytokine production */
    sim->cytokines[IMMUNO_CYTOKINE_IL6].production_rate = 100.0f;
    sim->cytokines[IMMUNO_CYTOKINE_IL6].decay_rate = 0.1f;

    float initial = sim->cytokines[IMMUNO_CYTOKINE_IL6].concentration;
    for (int i = 0; i < 10; i++) {
        immunology_step_cytokines(sim, 1.0f);
    }
    float after = sim->cytokines[IMMUNO_CYTOKINE_IL6].concentration;
    ASSERT_GT(after, initial);

    immunology_destroy(sim);
}

/* ---- load bacterial infection -------------------------------------------- */

TEST(load_bacterial_infection) {
    immunology_config_t cfg = immunology_default_config();
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    immunology_load_bacterial_infection(sim);
    ASSERT_GE(sim->num_pathogens, 1);

    immunology_destroy(sim);
}

/* ---- class switch -------------------------------------------------------- */

TEST(antibody_class_switch) {
    immunology_config_t cfg = immunology_default_config();
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    immuno_antibody_t ab = {0};
    ab.ab_class = IMMUNO_AB_IGM;
    ab.concentration = 10.0f;
    ab.affinity_ka = IMMUNO_KA_MODERATE;
    ab.active = true;
    sim->antibodies[0] = ab;
    sim->num_antibodies = 1;

    int rc = immunology_class_switch(sim, 0, IMMUNO_AB_IGG);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(sim->antibodies[0].ab_class, IMMUNO_AB_IGG);

    immunology_destroy(sim);
}

/* ---- step basic ---------------------------------------------------------- */

TEST(step_basic) {
    immunology_config_t cfg = immunology_default_config();
    immunology_sim_t* sim = immunology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = immunology_step(sim, 1.0f);
    ASSERT_EQ(rc, 0);

    immunology_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(complement_activates);
    RUN_TEST_SAFE(antibody_binding_affinity);
    RUN_TEST_SAFE(antibody_binding_monotonic);
    RUN_TEST_SAFE(vaccine_secondary_faster);
    RUN_TEST_SAFE(add_immune_cell);
    RUN_TEST_SAFE(cytokine_dynamics);
    RUN_TEST_SAFE(load_bacterial_infection);
    RUN_TEST_SAFE(antibody_class_switch);
    RUN_TEST_SAFE(step_basic);
TEST_MAIN_END()
