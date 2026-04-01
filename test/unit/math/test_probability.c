/**
 * @file test_probability.c
 * @brief Tests for the probability and statistics engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_probability.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double TOL = 1e-6;

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    probability_engine_t *eng = probability_create(42);
    ASSERT_NOT_NULL(eng);
    probability_destroy(eng);
}

/* ---------- normal PDF at mean is maximum ---------- */

TEST(normal_pdf_at_mean) {
    dist_params_t d = {0};
    d.type = DIST_NORMAL;
    d.mu = 5.0;
    d.sigma = 2.0;

    double pdf_at_mean = prob_pdf(&d, 5.0);
    /* PDF of normal at mean = 1/(sigma*sqrt(2*pi)) */
    double expected = 1.0 / (2.0 * sqrt(2.0 * M_PI));
    ASSERT_NEAR(pdf_at_mean, expected, TOL);

    /* PDF slightly off-mean should be less */
    double pdf_off = prob_pdf(&d, 5.5);
    ASSERT_TRUE(pdf_off < pdf_at_mean);
}

/* ---------- uniform CDF ---------- */

TEST(uniform_cdf_midpoint) {
    dist_params_t d = {0};
    d.type = DIST_UNIFORM;
    d.a = 0.0;
    d.b = 1.0;

    ASSERT_NEAR(prob_cdf(&d, 0.5), 0.5, TOL);
    ASSERT_NEAR(prob_cdf(&d, 0.0), 0.0, TOL);
    ASSERT_NEAR(prob_cdf(&d, 1.0), 1.0, TOL);
    ASSERT_NEAR(prob_cdf(&d, 0.25), 0.25, TOL);
}

/* ---------- Poisson PMF ---------- */

TEST(poisson_pmf) {
    /* P(k=0 | lambda=1) = e^{-1} */
    dist_params_t d = {0};
    d.type = DIST_POISSON;
    d.lambda = 1.0;

    double p0 = prob_pdf(&d, 0.0);
    ASSERT_NEAR(p0, exp(-1.0), TOL);

    /* P(k=1 | lambda=1) = e^{-1} */
    double p1 = prob_pdf(&d, 1.0);
    ASSERT_NEAR(p1, exp(-1.0), TOL);
}

/* ---------- Pearson correlation ---------- */

TEST(pearson_identical) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double r = prob_pearson(x, x, 5);
    ASSERT_NEAR(r, 1.0, TOL);
}

TEST(pearson_negatively_correlated) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double r = prob_pearson(x, y, 5);
    ASSERT_NEAR(r, -1.0, TOL);
}

/* ---------- z-test ---------- */

TEST(z_test_rejects_large_shift) {
    /* Data from N(10, 1): test H0: mu=5 with known sigma=1 => reject */
    double data[20];
    for (int i = 0; i < 20; i++) data[i] = 10.0 + 0.1 * (i - 10);

    test_result_t res = prob_z_test(data, 20, 5.0, 1.0);
    ASSERT_TRUE(res.reject_null);
    ASSERT_TRUE(fabs(res.statistic) > 3.0);
}

TEST(z_test_fails_to_reject_correct_mean) {
    /* Data centered around 5 with sigma=1 => do not reject mu=5 */
    double data[30];
    for (int i = 0; i < 30; i++) data[i] = 5.0 + 0.05 * (i - 15);

    test_result_t res = prob_z_test(data, 30, 5.0, 1.0);
    ASSERT_FALSE(res.reject_null);
}

/* ---------- descriptive statistics ---------- */

TEST(descriptive_stats) {
    double data[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    descriptive_stats_t s = prob_descriptive(data, 8);

    ASSERT_EQ(s.n, 8);
    ASSERT_NEAR(s.mean, 5.0, TOL);
    ASSERT_NEAR(s.min, 2.0, TOL);
    ASSERT_NEAR(s.max, 9.0, TOL);
    ASSERT_TRUE(s.variance > 0.0);
    ASSERT_TRUE(s.std_dev > 0.0);
}

/* ---------- normal CDF ---------- */

TEST(normal_cdf_symmetry) {
    dist_params_t d = {0};
    d.type = DIST_NORMAL;
    d.mu = 0.0;
    d.sigma = 1.0;

    /* CDF at mean = 0.5 */
    ASSERT_NEAR(prob_cdf(&d, 0.0), 0.5, 1e-3);
    /* CDF(-x) + CDF(x) = 1 */
    double cdf_neg = prob_cdf(&d, -1.96);
    double cdf_pos = prob_cdf(&d, 1.96);
    ASSERT_NEAR(cdf_neg + cdf_pos, 1.0, 1e-3);
}

/* ---------- Spearman correlation ---------- */

TEST(spearman_monotonic) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {2.0, 4.0, 8.0, 16.0, 32.0}; /* monotonically increasing */
    double r = prob_spearman(x, y, 5);
    ASSERT_NEAR(r, 1.0, TOL);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(normal_pdf_at_mean);
    RUN_TEST_SAFE(uniform_cdf_midpoint);
    RUN_TEST_SAFE(poisson_pmf);
    RUN_TEST_SAFE(pearson_identical);
    RUN_TEST_SAFE(pearson_negatively_correlated);
    RUN_TEST_SAFE(z_test_rejects_large_shift);
    RUN_TEST_SAFE(z_test_fails_to_reject_correct_mean);
    RUN_TEST_SAFE(descriptive_stats);
    RUN_TEST_SAFE(normal_cdf_symmetry);
    RUN_TEST_SAFE(spearman_monotonic);
TEST_MAIN_END()
