/**
 * @file test_analytical_chemistry.c
 * @brief Tests for analytical chemistry: pH, buffers, titration, Beer-Lambert
 *
 * Validates strong acid pH, Henderson-Hasselbalch, titration equivalence,
 * Van Deemter, calibration, and Nernst equation.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_analytical_chemistry.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    analytical_chemistry_config_t cfg = analytical_chemistry_default_config();
    analytical_chemistry_sim_t* sim = analytical_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    analytical_chemistry_destroy(sim);
}

/* ---- strong acid pH: pH = -log10([H+]) ---------------------------------- */

TEST(strong_acid_ph_01M) {
    /* 0.1 M HCl -> pH = -log10(0.1) = 1.0 */
    float ph = analytical_chemistry_strong_acid_pH(0.1f);
    ASSERT_NEAR(ph, 1.0f, 0.05f);
}

TEST(strong_acid_ph_001M) {
    /* 0.01 M HCl -> pH = 2.0 */
    float ph = analytical_chemistry_strong_acid_pH(0.01f);
    ASSERT_NEAR(ph, 2.0f, 0.05f);
}

TEST(strong_base_ph) {
    /* 0.1 M NaOH -> pOH = 1 -> pH = 13.0 */
    float ph = analytical_chemistry_strong_base_pH(0.1f);
    ASSERT_NEAR(ph, 13.0f, 0.1f);
}

/* ---- Henderson-Hasselbalch: pH = pKa + log([A-]/[HA]) ------------------- */

TEST(henderson_hasselbalch_equal) {
    /* When [A-] = [HA], pH = pKa */
    float ph = analytical_chemistry_henderson_hasselbalch(4.76f, 0.1f, 0.1f);
    ASSERT_NEAR(ph, 4.76f, 0.01f);
}

TEST(henderson_hasselbalch_excess_base) {
    /* [A-] = 10*[HA] -> pH = pKa + 1 */
    float ph = analytical_chemistry_henderson_hasselbalch(4.76f, 1.0f, 0.1f);
    ASSERT_NEAR(ph, 5.76f, 0.05f);
}

TEST(henderson_hasselbalch_excess_acid) {
    /* [A-] = 0.1*[HA] -> pH = pKa - 1 */
    float ph = analytical_chemistry_henderson_hasselbalch(4.76f, 0.1f, 1.0f);
    ASSERT_NEAR(ph, 3.76f, 0.05f);
}

/* ---- titration equivalence point ----------------------------------------- */

TEST(titration_equivalence_point) {
    analytical_chemistry_config_t cfg = analytical_chemistry_default_config();
    analytical_chemistry_sim_t* sim = analytical_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Analyte: 0.1M HCl, 50 mL */
    anachem_solution_t analyte = {0};
    snprintf(analyte.name, ANACHEM_MAX_NAME, "HCl");
    analyte.type = ANACHEM_ACID_STRONG;
    analyte.concentration = 0.1f;
    analyte.volume = 0.050f; /* 50 mL */
    analyte.active = true;
    uint32_t aid = analytical_chemistry_add_solution(sim, &analyte);

    /* Titrant: 0.1M NaOH */
    anachem_solution_t titrant = {0};
    snprintf(titrant.name, ANACHEM_MAX_NAME, "NaOH");
    titrant.type = ANACHEM_BASE_STRONG;
    titrant.concentration = 0.1f;
    titrant.volume = 0.100f;
    titrant.active = true;
    uint32_t tid = analytical_chemistry_add_solution(sim, &titrant);

    anachem_titration_t tit = {0};
    tit.type = ANACHEM_TITRATION_SA_SB;
    tit.analyte_idx = aid;
    tit.titrant_idx = tid;
    tit.active = true;
    analytical_chemistry_add_titration(sim, &tit);

    /* At equivalence: 50 mL of 0.1M titrant needed */
    /* pH of strong acid/strong base at equivalence = 7.0 */
    float ph_eq = analytical_chemistry_titration_pH(&sim->titrations[0],
        &sim->solutions[aid], &sim->solutions[tid], 50.0f);
    ASSERT_NEAR(ph_eq, 7.0f, 0.5f);

    analytical_chemistry_destroy(sim);
}

/* ---- Beer-Lambert -------------------------------------------------------- */

TEST(beer_lambert_absorbance) {
    /* epsilon=50, l=2 cm, c=0.01 M -> A = 1.0 */
    anachem_beer_lambert_t bl = analytical_chemistry_beer_lambert(50.0f, 2.0f, 0.01f);
    ASSERT_NEAR(bl.absorbance, 1.0f, 0.001f);
    /* T = 10^(-A) = 0.1 */
    ASSERT_NEAR(bl.transmittance, 0.1f, 0.01f);
}

/* ---- Van Deemter: H = A + B/u + C*u ------------------------------------ */

TEST(van_deemter_value) {
    float H = analytical_chemistry_van_deemter(0.1f, 1.0f, 0.01f, 10.0f);
    /* H = 0.1 + 1.0/10 + 0.01*10 = 0.1 + 0.1 + 0.1 = 0.3 */
    ASSERT_NEAR(H, 0.3f, 0.01f);
}

TEST(van_deemter_optimal_flow) {
    /* u_opt = sqrt(B/C) */
    float u_opt = analytical_chemistry_optimal_flow(1.0f, 0.01f);
    ASSERT_NEAR(u_opt, 10.0f, 0.5f);
}

/* ---- Nernst equation ----------------------------------------------------- */

TEST(nernst_standard_conditions) {
    /* E0=0.34V, n=2, Q=1, T=298.15 -> E = E0 */
    float E = analytical_chemistry_nernst(0.34f, 2, 1.0f, 298.15f);
    ASSERT_NEAR(E, 0.34f, 0.01f);
}

/* ---- buffer capacity ----------------------------------------------------- */

TEST(buffer_capacity_maximum) {
    /* Buffer capacity is maximal when pH = pKa (i.e., [H+] = Ka) */
    /* For acetic acid: Ka = 10^(-4.76) = 1.738e-5, total_conc = 0.1 */
    float Ka = 1.738e-5f;
    float H_at_pka = Ka;
    float beta = analytical_chemistry_buffer_capacity(0.1f, Ka, H_at_pka);
    ASSERT_GT(beta, 0.0f);

    /* Buffer capacity should be lower away from pKa */
    float H_off = Ka * 10.0f;  /* one pH unit away */
    float beta_off = analytical_chemistry_buffer_capacity(0.1f, Ka, H_off);
    ASSERT_GT(beta, beta_off);
}

/* ---- load common buffers ------------------------------------------------- */

TEST(load_common_buffers) {
    analytical_chemistry_config_t cfg = analytical_chemistry_default_config();
    analytical_chemistry_sim_t* sim = analytical_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = analytical_chemistry_load_common_buffers(sim);
    ASSERT_EQ(rc, 0);
    ASSERT_GE(sim->num_solutions, 6);  /* phosphate, Tris, HEPES, acetate, carbonate, citrate */

    analytical_chemistry_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(strong_acid_ph_01M);
    RUN_TEST_SAFE(strong_acid_ph_001M);
    RUN_TEST_SAFE(strong_base_ph);
    RUN_TEST_SAFE(henderson_hasselbalch_equal);
    RUN_TEST_SAFE(henderson_hasselbalch_excess_base);
    RUN_TEST_SAFE(henderson_hasselbalch_excess_acid);
    RUN_TEST_SAFE(titration_equivalence_point);
    RUN_TEST_SAFE(beer_lambert_absorbance);
    RUN_TEST_SAFE(van_deemter_value);
    RUN_TEST_SAFE(van_deemter_optimal_flow);
    RUN_TEST_SAFE(nernst_standard_conditions);
    RUN_TEST_SAFE(buffer_capacity_maximum);
    RUN_TEST_SAFE(load_common_buffers);
TEST_MAIN_END()
