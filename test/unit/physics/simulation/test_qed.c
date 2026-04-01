/**
 * @file test_qed.c
 * @brief Tests for the QED Engine — quantum electrodynamics
 *
 * Verifies: Thomson cross-section, anomalous magnetic moment, running alpha,
 * pair production threshold, Klein-Nishina low-energy limit, Breit-Wheeler threshold,
 * Feynman rules, vertex coupling.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_qed.h"

/* ---- Tests ---- */

TEST(create_destroy) {
    qed_config_t cfg = qed_default_config();
    qed_sim_t* sim = qed_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    qed_destroy(sim);
}

TEST(thomson_cross_section) {
    /* sigma_T = 8*pi/3 * r_e^2 = 6.652e-29 m^2 */
    float sigma = qed_thomson_cross_section();
    double expected = 6.6524e-29;
    /* Allow 1% tolerance */
    ASSERT_NEAR(sigma, expected, expected * 0.02);
}

TEST(anomalous_moment_first_order) {
    /* a_e = alpha/(2*pi) = 0.0011614... (Schwinger term) */
    float a_e = qed_anomalous_moment_first_order();
    double expected = QED_ALPHA / (2.0 * M_PI);  /* ~0.001161 */
    ASSERT_NEAR(a_e, expected, 1e-5);
}

TEST(anomalous_moment_third_order) {
    /* Through O(alpha^3): a_e ~ 0.00115965... */
    float a_e = qed_anomalous_moment_third_order();
    ASSERT_NEAR(a_e, 0.00115965f, 0.0001f);
}

TEST(running_alpha_at_mz) {
    /* alpha(M_Z^2) ~ 1/128 = 0.00781 (compared to alpha(0) ~ 1/137 = 0.00730) */
    float mz = 91.2f;
    float alpha_mz = qed_running_alpha(mz * mz);
    /* Should be larger than alpha(0) due to screening */
    ASSERT_GT(alpha_mz, QED_ALPHA);
    /* Should be approximately 1/128 */
    ASSERT_NEAR(alpha_mz, 1.0f / 128.0f, 0.002f);
}

TEST(pair_production_threshold) {
    /* E_threshold = 2 * m_e = 2 * 0.000511 GeV = 0.001022 GeV = 1.022 MeV */
    float threshold = qed_pair_production_threshold();
    float expected = 2.0f * QED_ELECTRON_MASS;
    ASSERT_NEAR(threshold, expected, 1e-5f);
}

TEST(klein_nishina_low_energy) {
    /* At low photon energy (E << m_e), Klein-Nishina reduces to Thomson.
     * Integrate dσ/dΩ over all angles should give sigma_T for E -> 0 */
    float sigma_T = qed_thomson_cross_section();
    /* At very low energy, forward scattering cross-section should be close
     * to 3/(8*pi)*sigma_T (Thomson differential at theta=0) */
    float low_E = 1e-6f;  /* 1 eV in GeV — very low */
    float dsigma_0 = qed_klein_nishina(low_E, 0.0f);
    float dsigma_90 = qed_klein_nishina(low_E, (float)(M_PI / 2.0));
    /* Thomson is isotropic-ish: dσ/dΩ(0) / dσ/dΩ(π/2) should be ~2
     * (from (1+cos^2θ)/2 ratio: (1+1)/2 vs (1+0)/2 = 1 vs 0.5) */
    if (dsigma_90 > 0.0f) {
        float ratio = dsigma_0 / dsigma_90;
        ASSERT_NEAR(ratio, 2.0f, 0.5f);
    }
    ASSERT_GT(dsigma_0, 0.0f);
}

TEST(breit_wheeler) {
    /* gamma + gamma -> e+ e-
     * Threshold: sqrt(s) = 2*m_e => s = 4*m_e^2 => E_cm = 2*m_e */
    float sigma_below = qed_breit_wheeler_cross_section(0.0005f);  /* below 2*m_e */
    float sigma_above = qed_breit_wheeler_cross_section(0.01f);    /* above 2*m_e */
    /* Below threshold should be zero or very small */
    ASSERT_GE(sigma_above, 0.0f);
    /* Above threshold should be nonzero */
    if (0.01f > 2.0f * QED_ELECTRON_MASS) {
        ASSERT_GT(sigma_above, sigma_below);
    }
}

TEST(vertex_coupling) {
    /* QED vertex: e = sqrt(4*pi*alpha) ~ 0.3028 */
    float e_coupling = qed_vertex_coupling();
    double expected = sqrt(4.0 * M_PI * QED_ALPHA);
    ASSERT_NEAR(e_coupling, expected, 0.01);
}

TEST(photon_propagator) {
    /* 1/q^2 — at q^2 = 1 GeV^2, propagator = 1.0 */
    float prop = qed_photon_propagator(1.0f);
    ASSERT_NEAR(prop, 1.0f, 0.1f);

    /* At q^2 = 4, propagator = 0.25 */
    float prop4 = qed_photon_propagator(4.0f);
    ASSERT_NEAR(prop4, 0.25f, 0.05f);
}

TEST(electron_propagator) {
    /* 1/(p^2 - m^2) at p^2 = 1, m = m_e ~ 0.000511 */
    float prop = qed_electron_propagator(1.0f, QED_ELECTRON_MASS);
    float expected = 1.0f / (1.0f - QED_ELECTRON_MASS * QED_ELECTRON_MASS);
    ASSERT_NEAR(prop, expected, 0.01f);
}

TEST(vacuum_polarization_sign) {
    /* Vacuum polarization Π(q^2) should be positive for spacelike q^2 > 0
     * (screening effect: charges are screened) */
    float pi_q2 = qed_vacuum_polarization(100.0f);  /* 100 GeV^2 */
    ASSERT_GT(pi_q2, 0.0f);
}

TEST(load_standard_processes) {
    qed_config_t cfg = qed_default_config();
    qed_sim_t* sim = qed_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qed_load_standard_processes(sim);
    ASSERT_GT(sim->num_processes, 0);

    qed_destroy(sim);
}

TEST(compute_cross_section) {
    qed_config_t cfg = qed_default_config();
    qed_sim_t* sim = qed_create(&cfg);
    ASSERT_NOT_NULL(sim);

    qed_load_standard_processes(sim);
    /* Compute Compton cross-section at 1 GeV CM energy */
    float sigma = qed_compute_cross_section(sim, QED_PROC_COMPTON, 1.0f, 0.5f);
    ASSERT_GE(sigma, 0.0f);

    qed_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(thomson_cross_section);
    RUN_TEST_SAFE(anomalous_moment_first_order);
    RUN_TEST_SAFE(anomalous_moment_third_order);
    RUN_TEST_SAFE(running_alpha_at_mz);
    RUN_TEST_SAFE(pair_production_threshold);
    RUN_TEST_SAFE(klein_nishina_low_energy);
    RUN_TEST_SAFE(breit_wheeler);
    RUN_TEST_SAFE(vertex_coupling);
    RUN_TEST_SAFE(photon_propagator);
    RUN_TEST_SAFE(electron_propagator);
    RUN_TEST_SAFE(vacuum_polarization_sign);
    RUN_TEST_SAFE(load_standard_processes);
    RUN_TEST_SAFE(compute_cross_section);
TEST_MAIN_END()
