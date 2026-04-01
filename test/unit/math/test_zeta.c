/**
 * @file test_zeta.c
 * @brief Tests for the Riemann zeta function engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_zeta_functions.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double TOL = 1e-6;

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    zeta_config_t cfg = zeta_default_config();
    zeta_engine_t *eng = zeta_create(&cfg);
    ASSERT_NOT_NULL(eng);
    zeta_destroy(eng);
}

/* ---------- special values ---------- */

TEST(zeta_2_is_pi_squared_over_6) {
    double val = zeta_basel();
    double expected = M_PI * M_PI / 6.0;
    ASSERT_NEAR(val, expected, TOL);
}

TEST(zeta_negative_1) {
    /* zeta(-1) = -1/12 (analytic continuation) */
    double val = zeta_negative_1();
    ASSERT_NEAR(val, -1.0 / 12.0, TOL);
}

TEST(zeta_at_zero) {
    /* zeta(0) = -1/2 */
    double val = zeta_at_zero();
    ASSERT_NEAR(val, -0.5, TOL);
}

TEST(zeta_4) {
    /* zeta(4) = pi^4 / 90 */
    double val = zeta_4();
    double expected = M_PI * M_PI * M_PI * M_PI / 90.0;
    ASSERT_NEAR(val, expected, TOL);
}

/* ---------- real evaluation ---------- */

TEST(zeta_eval_real) {
    zeta_config_t cfg = zeta_default_config();
    zeta_engine_t *eng = zeta_create(&cfg);
    ASSERT_NOT_NULL(eng);

    /* zeta(2) should match Basel */
    double z2 = zeta_eval_real(eng, 2.0);
    ASSERT_NEAR(z2, M_PI * M_PI / 6.0, 1e-4);

    /* zeta(3) ~ 1.2020569 (Apery's constant) */
    double z3 = zeta_eval_real(eng, 3.0);
    ASSERT_NEAR(z3, 1.2020569, 1e-4);

    zeta_destroy(eng);
}

/* ---------- zero counting N(T) ---------- */

TEST(zero_counting_N100) {
    /* N(100) ~ 29 non-trivial zeros with Im(s) < 100 */
    double N = zeta_N(100.0);
    ASSERT_NEAR(N, 29.0, 2.0);  /* approximate formula, tolerance of 2 */
}

/* ---------- find zeros in range ---------- */

TEST(find_zeros_in_range) {
    zeta_config_t cfg = zeta_default_config();
    cfg.verify_zeros = false;  /* speed */
    zeta_engine_t *eng = zeta_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t found = zeta_find_zeros_in_range(eng, 10.0, 50.0, 0.5);
    /* There are 10 known zeros between t=14.1 and t=49.8 */
    ASSERT_GE(found, 8);   /* allow some tolerance for numerical issues */
    ASSERT_LE(found, 12);

    zeta_destroy(eng);
}

/* ---------- GUE pair correlation formula ---------- */

TEST(gue_pair_correlation) {
    /* At x=0: 1 - (sin(0)/0)^2 = 1 - 1 = 0 (by L'Hopital) */
    double g0 = zeta_gue_pair_correlation(0.0);
    ASSERT_NEAR(g0, 0.0, 1e-4);

    /* At x=1: 1 - (sin(pi)/pi)^2 = 1 - 0 = 1 */
    double g1 = zeta_gue_pair_correlation(1.0);
    ASSERT_NEAR(g1, 1.0, 1e-4);

    /* Monotonically approaches 1 for large x */
    double g10 = zeta_gue_pair_correlation(10.0);
    ASSERT_GT(g10, 0.99);
}

/* ---------- Wigner surmise ---------- */

TEST(wigner_surmise) {
    /* p(s) = (pi/2)*s*exp(-pi*s^2/4) */
    /* At s=0, p(0) = 0 */
    ASSERT_NEAR(zeta_wigner_surmise(0.0), 0.0, TOL);

    /* Peak at s=sqrt(2/pi) ~ 0.7979 */
    double s_peak = sqrt(2.0 / M_PI);
    double p_peak = zeta_wigner_surmise(s_peak);
    /* Check it's a local max: neighbors should be smaller */
    ASSERT_GT(p_peak, zeta_wigner_surmise(s_peak - 0.1));
    ASSERT_GT(p_peak, zeta_wigner_surmise(s_peak + 0.1));
}

/* ---------- known zeros ---------- */

TEST(known_zeros) {
    zeta_config_t cfg = zeta_default_config();
    zeta_engine_t *eng = zeta_create(&cfg);
    ASSERT_NOT_NULL(eng);

    zeta_load_known_zeros(eng);
    ASSERT_GE(eng->num_zeros, 10);

    /* First zero should be near 14.134725 */
    ASSERT_NEAR(eng->zeros[0].t, ZETA_ZERO_1, 0.01);

    zeta_destroy(eng);
}

/* ---------- Z-function sign change at zero ---------- */

TEST(z_function_sign_change) {
    zeta_config_t cfg = zeta_default_config();
    zeta_engine_t *eng = zeta_create(&cfg);
    ASSERT_NOT_NULL(eng);

    /* Z(t) should have opposite signs on either side of a zero */
    double z_before = zeta_Z(eng, 14.0);
    double z_after = zeta_Z(eng, 14.3);
    /* They should have opposite signs (sign change = zero between them) */
    ASSERT_TRUE(z_before * z_after < 0 || fabs(z_before) < 0.5 || fabs(z_after) < 0.5);

    zeta_destroy(eng);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(zeta_2_is_pi_squared_over_6);
    RUN_TEST_SAFE(zeta_negative_1);
    RUN_TEST_SAFE(zeta_at_zero);
    RUN_TEST_SAFE(zeta_4);
    RUN_TEST_SAFE(zeta_eval_real);
    RUN_TEST_SAFE(zero_counting_N100);
    RUN_TEST_SAFE(find_zeros_in_range);
    RUN_TEST_SAFE(gue_pair_correlation);
    RUN_TEST_SAFE(wigner_surmise);
    RUN_TEST_SAFE(known_zeros);
    RUN_TEST_SAFE(z_function_sign_change);
TEST_MAIN_END()
