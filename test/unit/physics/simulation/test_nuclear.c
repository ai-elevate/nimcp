/**
 * @file test_nuclear.c
 * @brief Tests for the Nuclear Physics Engine — binding energy, decay, fission/fusion
 *
 * Verifies: SEMF binding energy for Fe-56, half-life/decay-constant round-trip,
 * U-235 fissile, radioactive decay law, nuclear radius, Coulomb barrier.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_nuclear_physics.h"

/* ---- Tests ---- */

TEST(create_destroy) {
    np_config_t cfg = np_default_config();
    nuclear_physics_sim_t* sim = np_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    np_destroy(sim);
}

TEST(semf_fe56) {
    /* Fe-56: Z=26, A=56
     * Binding energy per nucleon ~ 8.79 MeV/nucleon (the peak of the curve) */
    float BE_per_A = np_binding_energy_per_nucleon(26, 56);
    ASSERT_NEAR(BE_per_A, 8.79f, 0.2f);
}

TEST(semf_total_fe56) {
    /* Total binding energy ~ 8.79 * 56 ~ 492 MeV */
    float BE = np_binding_energy_semf(26, 56);
    ASSERT_NEAR(BE, 492.0f, 15.0f);
}

TEST(semf_he4) {
    /* He-4 (alpha particle): Z=2, A=4
     * B/A ~ 7.07 MeV/nucleon (SEMF is less accurate for light nuclei) */
    float BE_per_A = np_binding_energy_per_nucleon(2, 4);
    ASSERT_GT(BE_per_A, 4.0f);
    ASSERT_LT(BE_per_A, 10.0f);
}

TEST(semf_heavy_nucleus) {
    /* U-238: Z=92, A=238
     * B/A ~ 7.57 MeV/nucleon — less than Fe-56 (past the peak) */
    float BE_per_A = np_binding_energy_per_nucleon(92, 238);
    ASSERT_NEAR(BE_per_A, 7.57f, 0.3f);
    /* Should be less than Fe-56 */
    float fe56 = np_binding_energy_per_nucleon(26, 56);
    ASSERT_LT(BE_per_A, fe56);
}

TEST(decay_constant_roundtrip) {
    /* lambda = ln(2) / t_half, t_half = ln(2) / lambda */
    float t_half = 100.0f;  /* 100 seconds */
    float lambda = np_decay_constant(t_half);
    ASSERT_NEAR(lambda, NP_LN2 / 100.0f, 1e-5f);

    float t_half_back = np_half_life(lambda);
    ASSERT_NEAR(t_half_back, t_half, 0.001f);
}

TEST(decay_constant_value) {
    /* ln(2) / 1.0 = 0.693147... */
    float lambda = np_decay_constant(1.0f);
    ASSERT_NEAR(lambda, 0.693147f, 1e-4f);
}

TEST(radioactive_decay_law) {
    /* N(t) = N0 * exp(-lambda * t) */
    double N0 = 1.0e6;
    float lambda = np_decay_constant(10.0f);  /* t_half = 10s */
    float t = 10.0f;  /* one half-life */
    double N = np_decay_population(N0, lambda, t);
    /* After one half-life, N ~ N0/2 */
    ASSERT_NEAR(N, N0 / 2.0, N0 * 0.01);
}

TEST(decay_two_halflives) {
    double N0 = 1.0e6;
    float lambda = np_decay_constant(5.0f);  /* t_half = 5s */
    double N = np_decay_population(N0, lambda, 10.0f);  /* 2 half-lives */
    /* N ~ N0/4 */
    ASSERT_NEAR(N, N0 / 4.0, N0 * 0.01);
}

TEST(activity) {
    /* A = lambda * N */
    float lambda = 0.1f;
    double N = 1e6;
    double A = np_activity(lambda, N);
    ASSERT_NEAR(A, 1e5, 1.0);
}

TEST(nuclear_radius) {
    /* R = r0 * A^(1/3), r0 ~ 1.25 fm
     * For A=56: R ~ 1.25 * 56^(1/3) = 1.25 * 3.826 = 4.78 fm */
    float R = np_nuclear_radius(56);
    ASSERT_NEAR(R, 4.78f, 0.3f);
}

TEST(u235_fission_energy) {
    /* Fission of U-235 releases ~200 MeV */
    float E = np_fission_energy(92, 235);
    ASSERT_NEAR(E, 200.0f, 40.0f);
}

TEST(load_common_nuclides) {
    np_config_t cfg = np_default_config();
    nuclear_physics_sim_t* sim = np_create(&cfg);
    ASSERT_NOT_NULL(sim);

    np_load_common_nuclides(sim);
    ASSERT_GT(sim->num_nuclides, 0);

    /* U-235 should be present */
    bool found_u235 = false;
    for (uint32_t i = 0; i < sim->num_nuclides; i++) {
        if (sim->nuclides[i].Z == 92 && sim->nuclides[i].A == 235) {
            found_u235 = true;
            /* U-235 is fissile — it should have a finite half-life */
            ASSERT_GT(sim->nuclides[i].half_life, 0.0f);
            break;
        }
    }
    ASSERT_TRUE(found_u235);

    np_destroy(sim);
}

TEST(coulomb_barrier) {
    /* Coulomb barrier between two protons at nuclear distance
     * V = k_e * e^2 / r ~ 1.44 MeV*fm / r(fm)
     * For r ~ 1.25 fm: V ~ 1.15 MeV */
    float V = np_coulomb_barrier(1, 1, 1.25f);
    ASSERT_GT(V, 0.5f);
    ASSERT_LT(V, 3.0f);
}

TEST(step_runs) {
    np_config_t cfg = np_default_config();
    nuclear_physics_sim_t* sim = np_create(&cfg);
    ASSERT_NOT_NULL(sim);

    np_load_common_nuclides(sim);
    int rc = np_step(sim, 1.0f);
    ASSERT_EQ(rc, 0);

    np_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(semf_fe56);
    RUN_TEST_SAFE(semf_total_fe56);
    RUN_TEST_SAFE(semf_he4);
    RUN_TEST_SAFE(semf_heavy_nucleus);
    RUN_TEST_SAFE(decay_constant_roundtrip);
    RUN_TEST_SAFE(decay_constant_value);
    RUN_TEST_SAFE(radioactive_decay_law);
    RUN_TEST_SAFE(decay_two_halflives);
    RUN_TEST_SAFE(activity);
    RUN_TEST_SAFE(nuclear_radius);
    RUN_TEST_SAFE(u235_fission_energy);
    RUN_TEST_SAFE(load_common_nuclides);
    RUN_TEST_SAFE(coulomb_barrier);
    RUN_TEST_SAFE(step_runs);
TEST_MAIN_END()
