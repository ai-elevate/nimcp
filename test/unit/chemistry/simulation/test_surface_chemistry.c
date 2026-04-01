/**
 * @file test_surface_chemistry.c
 * @brief Tests for surface chemistry: isotherms, Arrhenius, Nernst, Butler-Volmer
 *
 * Validates Langmuir isotherm, Arrhenius rate, Nernst electrode potential,
 * Butler-Volmer kinetics, Pt catalyst loading, and corrosion computations.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_surface_chemistry.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    schem_config_t cfg = surface_chemistry_default_config();
    surface_chemistry_sim_t* sim = surface_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    surface_chemistry_destroy(sim);
}

/* ---- Langmuir isotherm: theta = KP/(1+KP) ------------------------------- */

TEST(langmuir_known_value) {
    /* K=0.01, P=100 -> theta = 0.01*100 / (1 + 0.01*100) = 1/2 = 0.5 */
    float theta = surface_chemistry_langmuir(0.01f, 100.0f);
    ASSERT_NEAR(theta, 0.5f, 0.001f);
}

TEST(langmuir_low_pressure) {
    /* K=0.01, P=1 -> theta = 0.01/(1.01) ~ 0.0099 (linear regime) */
    float theta = surface_chemistry_langmuir(0.01f, 1.0f);
    ASSERT_NEAR(theta, 0.0099f, 0.001f);
}

TEST(langmuir_high_pressure) {
    /* K=0.01, P=100000 -> theta ~ 1.0 (saturation) */
    float theta = surface_chemistry_langmuir(0.01f, 100000.0f);
    ASSERT_GT(theta, 0.99f);
    ASSERT_LE(theta, 1.0f);
}

/* ---- Arrhenius: k = A * exp(-Ea/RT) ------------------------------------- */

TEST(arrhenius_at_500K) {
    float A = 1.0e10f;
    float Ea = 50.0f;  /* kJ/mol */
    float T = 500.0f;
    float k = surface_chemistry_arrhenius(A, Ea, T);
    /* k = 1e10 * exp(-50000/(8.314*500)) = 1e10 * exp(-12.027) */
    double expected = 1.0e10 * exp(-50000.0 / (8.314 * 500.0));
    ASSERT_NEAR(k, (float)expected, (float)(expected * 0.05));
}

TEST(arrhenius_temperature_dependence) {
    float A = 1.0e10f;
    float Ea = 50.0f;
    /* Higher T should give higher rate */
    float k_low = surface_chemistry_arrhenius(A, Ea, 300.0f);
    float k_high = surface_chemistry_arrhenius(A, Ea, 600.0f);
    ASSERT_GT(k_high, k_low);
}

/* ---- Nernst: E = E0 - (RT/nF) * ln(Q) ---------------------------------- */

TEST(nernst_cu_standard) {
    /* Cu2+/Cu: E0=0.34V, T=298.15K, n=2, Q=1 -> E = E0 - 0 = 0.34V */
    float E = surface_chemistry_nernst(0.34f, 298.15f, 2, 1.0f);
    ASSERT_NEAR(E, 0.34f, 0.01f);
}

TEST(nernst_nonunit_Q) {
    /* E0=0.34V, T=298.15, n=2, Q=0.01 */
    /* E = 0.34 - (8.314*298.15/(2*96485)) * ln(0.01) */
    /* = 0.34 - 0.01285 * (-4.605) = 0.34 + 0.0592 ~ 0.399 */
    float E = surface_chemistry_nernst(0.34f, 298.15f, 2, 0.01f);
    ASSERT_NEAR(E, 0.399f, 0.02f);
}

/* ---- Butler-Volmer: sign test ------------------------------------------- */

TEST(butler_volmer_positive_eta) {
    /* Positive overpotential should give net positive (anodic) current */
    float j = surface_chemistry_butler_volmer(1.0f, 0.5f, 1, 0.1f, 298.15f);
    ASSERT_GT(j, 0.0f);
}

TEST(butler_volmer_negative_eta) {
    /* Negative overpotential -> net negative (cathodic) current */
    float j = surface_chemistry_butler_volmer(1.0f, 0.5f, 1, -0.1f, 298.15f);
    ASSERT_LT(j, 0.0f);
}

TEST(butler_volmer_zero_eta) {
    /* Zero overpotential -> zero net current */
    float j = surface_chemistry_butler_volmer(1.0f, 0.5f, 1, 0.0f, 298.15f);
    ASSERT_NEAR(j, 0.0f, 0.001f);
}

/* ---- Pt catalyst loads and steps ----------------------------------------- */

TEST(pt_catalyst_load_and_step) {
    schem_config_t cfg = surface_chemistry_default_config();
    cfg.enable_catalysis = true;
    surface_chemistry_sim_t* sim = surface_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    surface_chemistry_load_pt_catalyst(sim);
    ASSERT_GE(sim->num_surfaces, 1);
    ASSERT_GE(sim->num_reactions, 1);

    int rc = surface_chemistry_step(sim, 0.01f);
    ASSERT_EQ(rc, 0);

    surface_chemistry_destroy(sim);
}

/* ---- Tafel slope --------------------------------------------------------- */

TEST(tafel_slope_value) {
    /* b = 2.303 * RT / (alpha * n * F) */
    /* alpha=0.5, n=1, T=298.15: b = 2.303*8.314*298.15/(0.5*1*96485) = 0.1183 V */
    float b = surface_chemistry_tafel_slope(0.5f, 1, 298.15f);
    ASSERT_NEAR(b, 0.1183f, 0.005f);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(langmuir_known_value);
    RUN_TEST_SAFE(langmuir_low_pressure);
    RUN_TEST_SAFE(langmuir_high_pressure);
    RUN_TEST_SAFE(arrhenius_at_500K);
    RUN_TEST_SAFE(arrhenius_temperature_dependence);
    RUN_TEST_SAFE(nernst_cu_standard);
    RUN_TEST_SAFE(nernst_nonunit_Q);
    RUN_TEST_SAFE(butler_volmer_positive_eta);
    RUN_TEST_SAFE(butler_volmer_negative_eta);
    RUN_TEST_SAFE(butler_volmer_zero_eta);
    RUN_TEST_SAFE(pt_catalyst_load_and_step);
    RUN_TEST_SAFE(tafel_slope_value);
TEST_MAIN_END()
