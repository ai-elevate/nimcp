/**
 * @file test_qft.c
 * @brief Tests for the QFT Engine — quantum field theory, Standard Model, RMT
 *
 * Verifies: alpha_s(M_Z), asymptotic freedom, Higgs mechanism masses,
 * Goldstone theorem, RMT eigenvalue generation, lattice Metropolis.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_qft.h"

/* ---- Tests ---- */

TEST(create_destroy) {
    qft_config_t cfg = qft_default_config();
    qft_sim_t* sim = qft_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    qft_destroy(sim);
}

TEST(alpha_s_at_mz) {
    /* alpha_s(M_Z = 91.2 GeV) ~ 0.118 */
    float alpha_s = qft_alpha_s(91.2f);
    ASSERT_NEAR(alpha_s, 0.118f, 0.015f);
}

TEST(asymptotic_freedom) {
    /* alpha_s should DECREASE at higher energies (asymptotic freedom) */
    float alpha_low  = qft_alpha_s(1.0f);    /* 1 GeV */
    float alpha_high = qft_alpha_s(91.2f);   /* M_Z */

    ASSERT_GT(alpha_low, 0.0f);
    ASSERT_GT(alpha_high, 0.0f);
    ASSERT_GT(alpha_low, alpha_high);  /* stronger at lower energy */
}

TEST(alpha_em_running) {
    /* alpha_EM increases at higher energies (anti-screening) */
    float a_low  = qft_alpha_em(1.0f);     /* ~1/137 at low energy */
    float a_high = qft_alpha_em(91.2f);    /* ~1/128 at M_Z */

    ASSERT_GT(a_low, 0.0f);
    ASSERT_GT(a_high, 0.0f);
    ASSERT_GT(a_high, a_low);  /* alpha grows with energy */
}

TEST(higgs_mechanism_masses) {
    /* v = 246 GeV, gives M_W ~ 80 GeV, M_Z ~ 91 GeV */
    qft_config_t cfg = qft_default_config();
    qft_sim_t* sim = qft_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qft_higgs_mechanism(sim, 246.0f, 0.13f);

    ASSERT_NEAR(sim->higgs.W_mass, 80.4f, 3.0f);
    ASSERT_NEAR(sim->higgs.Z_mass, 91.2f, 3.0f);
    ASSERT_TRUE(sim->higgs.broken);
    ASSERT_GT(sim->higgs.higgs_mass, 0.0f);

    qft_destroy(sim);
}

TEST(higgs_weinberg_angle) {
    qft_config_t cfg = qft_default_config();
    qft_sim_t* sim = qft_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qft_higgs_mechanism(sim, 246.0f, 0.13f);

    /* sin^2(theta_W) ~ 0.231 */
    ASSERT_NEAR(sim->higgs.weinberg_angle, 0.231f, 0.03f);

    qft_destroy(sim);
}

TEST(goldstone_count) {
    /* SU(2) x U(1) -> U(1)_EM
     * dim(SU(2)xU(1)) = 3+1 = 4, dim(U(1)) = 1
     * Goldstone bosons = 4 - 1 = 3 (eaten by W+, W-, Z) */
    uint32_t count = qft_goldstone_count(4, 1);
    ASSERT_EQ(count, 3);
}

TEST(goldstone_no_breaking) {
    /* If group dims are equal, no breaking => 0 Goldstones */
    uint32_t count = qft_goldstone_count(3, 3);
    ASSERT_EQ(count, 0);
}

TEST(beta_0_qcd) {
    /* QCD: b0 = -(11*N_c - 2*N_f) / (48*pi^2)
     * For N_c=3, N_f=6: 11*3 - 2*6 = 33 - 12 = 21
     * The beta_0 should be negative (asymptotic freedom) */
    float b0 = qft_beta_0_sun(3, 6);
    ASSERT_LT(b0, 0.0f);  /* negative => asymptotic freedom */
}

TEST(rmt_generate_eigenvalues) {
    qft_config_t cfg = qft_default_config();
    cfg.rmt_matrix_dim = 16;
    cfg.rmt_samples = 1;
    qft_sim_t* sim = qft_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qft_rmt_generate(sim, QFT_RMT_GUE);
    qft_rmt_compute_eigenvalues(sim);

    /* Should have eigenvalues */
    ASSERT_NOT_NULL(sim->rmt.eigenvalues);
    ASSERT_EQ(sim->rmt.matrix_dim, 16);

    /* Eigenvalues should be real numbers (not NaN) */
    for (uint32_t i = 0; i < sim->rmt.matrix_dim; i++) {
        ASSERT_FALSE(isnan(sim->rmt.eigenvalues[i]));
    }

    /* Eigenvalues should be sorted */
    for (uint32_t i = 1; i < sim->rmt.matrix_dim; i++) {
        ASSERT_GE(sim->rmt.eigenvalues[i], sim->rmt.eigenvalues[i - 1]);
    }

    qft_destroy(sim);
}

TEST(rmt_wigner_surmise) {
    /* GUE Wigner surmise at s=1: p(1) = (32/pi^2)*1^2*exp(-4/pi)
     * = 32/pi^2 * exp(-4/pi) ~ 3.243 * 0.2790 ~ 0.905 */
    double p_gue = qft_rmt_wigner_surmise(QFT_RMT_GUE, 1.0);
    ASSERT_GT(p_gue, 0.0);
    double expected = (32.0 / (M_PI * M_PI)) * exp(-4.0 / M_PI);
    ASSERT_NEAR(p_gue, expected, 0.1);
}

TEST(lattice_metropolis) {
    qft_config_t cfg = qft_default_config();
    cfg.lattice_L = 4;
    cfg.lattice_dim = 2;
    qft_sim_t* sim = qft_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qft_lattice_init(sim, true);  /* hot start */

    int rc = qft_lattice_metropolis_sweep(sim);
    ASSERT_EQ(rc, 0);

    qft_lattice_measure(sim);
    /* Action should be finite */
    ASSERT_FALSE(isnan(sim->lattice.action));
    ASSERT_FALSE(isinf(sim->lattice.action));

    qft_destroy(sim);
}

TEST(load_standard_model) {
    qft_config_t cfg = qft_default_config();
    qft_sim_t* sim = qft_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qft_load_standard_model(sim);
    /* Should have fields (quarks, leptons, gauge bosons, Higgs) */
    ASSERT_GT(sim->num_fields, 0);
    /* Should have couplings */
    ASSERT_GT(sim->num_couplings, 0);

    qft_destroy(sim);
}

TEST(run_coupling) {
    /* Run a coupling g from mu1 to mu2 with positive beta_0 (QED-like).
     * g should grow at higher scale. */
    float g_low = 0.3f;
    float b0 = 1.0f / (12.0f * (float)M_PI * (float)M_PI);  /* positive */
    float g_high = qft_run_coupling(g_low, 1.0f, 100.0f, b0);
    ASSERT_GT(g_high, g_low);  /* coupling grows for positive beta */
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(alpha_s_at_mz);
    RUN_TEST_SAFE(asymptotic_freedom);
    RUN_TEST_SAFE(alpha_em_running);
    RUN_TEST_SAFE(higgs_mechanism_masses);
    RUN_TEST_SAFE(higgs_weinberg_angle);
    RUN_TEST_SAFE(goldstone_count);
    RUN_TEST_SAFE(goldstone_no_breaking);
    RUN_TEST_SAFE(beta_0_qcd);
    RUN_TEST_SAFE(rmt_generate_eigenvalues);
    RUN_TEST_SAFE(rmt_wigner_surmise);
    RUN_TEST_SAFE(lattice_metropolis);
    RUN_TEST_SAFE(load_standard_model);
    RUN_TEST_SAFE(run_coupling);
TEST_MAIN_END()
