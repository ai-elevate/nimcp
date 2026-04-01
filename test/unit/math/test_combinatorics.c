/**
 * @file test_combinatorics.c
 * @brief Tests for the combinatorics engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_combinatorics.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    combinatorics_t *ctx = combinatorics_create();
    ASSERT_NOT_NULL(ctx);
    combinatorics_destroy(ctx);
}

/* ---------- factorial ---------- */

TEST(factorial_known) {
    combinatorics_t *ctx = combinatorics_create();
    ASSERT_NOT_NULL(ctx);
    ASSERT_EQ(comb_factorial(ctx, 0), 1);
    ASSERT_EQ(comb_factorial(ctx, 1), 1);
    ASSERT_EQ(comb_factorial(ctx, 5), 120);
    ASSERT_EQ(comb_factorial(ctx, 10), 3628800);
    ASSERT_EQ(comb_factorial(ctx, 20), (uint64_t)2432902008176640000ULL);
    combinatorics_destroy(ctx);
}

/* ---------- binomial coefficients ---------- */

TEST(binomial_known) {
    combinatorics_t *ctx = combinatorics_create();
    ASSERT_NOT_NULL(ctx);
    ASSERT_EQ(comb_binomial(ctx, 10, 3), 120);
    ASSERT_EQ(comb_binomial(ctx, 10, 0), 1);
    ASSERT_EQ(comb_binomial(ctx, 10, 10), 1);
    ASSERT_EQ(comb_binomial(ctx, 6, 3), 20);
    ASSERT_EQ(comb_binomial(ctx, 20, 10), 184756);
    combinatorics_destroy(ctx);
}

/* ---------- Catalan numbers ---------- */

TEST(catalan_known) {
    combinatorics_t *ctx = combinatorics_create();
    ASSERT_NOT_NULL(ctx);
    ASSERT_EQ(comb_catalan(ctx, 0), 1);
    ASSERT_EQ(comb_catalan(ctx, 1), 1);
    ASSERT_EQ(comb_catalan(ctx, 2), 2);
    ASSERT_EQ(comb_catalan(ctx, 3), 5);
    ASSERT_EQ(comb_catalan(ctx, 4), 14);
    ASSERT_EQ(comb_catalan(ctx, 5), 42);
    combinatorics_destroy(ctx);
}

/* ---------- Fibonacci ---------- */

TEST(fibonacci_known) {
    ASSERT_EQ(comb_fibonacci(0), 0);
    ASSERT_EQ(comb_fibonacci(1), 1);
    ASSERT_EQ(comb_fibonacci(2), 1);
    ASSERT_EQ(comb_fibonacci(5), 5);
    ASSERT_EQ(comb_fibonacci(10), 55);
    ASSERT_EQ(comb_fibonacci(20), 6765);
}

/* ---------- derangements ---------- */

TEST(derangement_known) {
    combinatorics_t *ctx = combinatorics_create();
    ASSERT_NOT_NULL(ctx);
    /* D(0)=1, D(1)=0, D(2)=1, D(3)=2, D(4)=9, D(5)=44 */
    ASSERT_EQ(comb_derangement(ctx, 0), 1);
    ASSERT_EQ(comb_derangement(ctx, 1), 0);
    ASSERT_EQ(comb_derangement(ctx, 2), 1);
    ASSERT_EQ(comb_derangement(ctx, 3), 2);
    ASSERT_EQ(comb_derangement(ctx, 4), 9);
    ASSERT_EQ(comb_derangement(ctx, 5), 44);
    combinatorics_destroy(ctx);
}

/* ---------- Stirling numbers (second kind) ---------- */

TEST(stirling_second_known) {
    /* S(4,2)=7, S(4,1)=1, S(4,4)=1, S(5,3)=25 */
    ASSERT_EQ(comb_stirling_second(4, 2), 7);
    ASSERT_EQ(comb_stirling_second(4, 1), 1);
    ASSERT_EQ(comb_stirling_second(4, 4), 1);
    ASSERT_EQ(comb_stirling_second(5, 3), 25);
    ASSERT_EQ(comb_stirling_second(3, 2), 3);
}

/* ---------- Stirling numbers (first kind) ---------- */

TEST(stirling_first_known) {
    /* |s(3,1)|=2, |s(3,2)|=3, |s(3,3)|=1, |s(4,2)|=11 */
    ASSERT_EQ(comb_stirling_first(3, 1), 2);
    ASSERT_EQ(comb_stirling_first(3, 2), 3);
    ASSERT_EQ(comb_stirling_first(3, 3), 1);
    ASSERT_EQ(comb_stirling_first(4, 2), 11);
}

/* ---------- Bell numbers ---------- */

TEST(bell_known) {
    /* B(0)=1, B(1)=1, B(2)=2, B(3)=5, B(4)=15, B(5)=52 */
    ASSERT_EQ(comb_bell(0), 1);
    ASSERT_EQ(comb_bell(1), 1);
    ASSERT_EQ(comb_bell(2), 2);
    ASSERT_EQ(comb_bell(3), 5);
    ASSERT_EQ(comb_bell(4), 15);
    ASSERT_EQ(comb_bell(5), 52);
}

/* ---------- permutation ---------- */

TEST(permutation_known) {
    combinatorics_t *ctx = combinatorics_create();
    ASSERT_NOT_NULL(ctx);
    /* P(5,2) = 20, P(10,3) = 720 */
    ASSERT_EQ(comb_permutation(ctx, 5, 2), 20);
    ASSERT_EQ(comb_permutation(ctx, 10, 3), 720);
    ASSERT_EQ(comb_permutation(ctx, 5, 5), 120);  /* = 5! */
    combinatorics_destroy(ctx);
}

/* ---------- partition function ---------- */

TEST(partition_known) {
    /* p(0)=1, p(1)=1, p(2)=2, p(3)=3, p(4)=5, p(5)=7, p(10)=42 */
    ASSERT_EQ(comb_partition(0), 1);
    ASSERT_EQ(comb_partition(1), 1);
    ASSERT_EQ(comb_partition(4), 5);
    ASSERT_EQ(comb_partition(5), 7);
    ASSERT_EQ(comb_partition(10), 42);
}

/* ---------- generating function ---------- */

TEST(ogf_evaluate) {
    /* 1 + 2x + 3x^2 at x=2 => 1+4+12 = 17 */
    double c[] = {1.0, 2.0, 3.0};
    ogf_t gf = comb_ogf_create(c, 3);
    ASSERT_NEAR(comb_ogf_evaluate(&gf, 2.0), 17.0, 1e-9);
    ASSERT_NEAR(comb_ogf_evaluate(&gf, 0.0), 1.0, 1e-9);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(factorial_known);
    RUN_TEST_SAFE(binomial_known);
    RUN_TEST_SAFE(catalan_known);
    RUN_TEST_SAFE(fibonacci_known);
    RUN_TEST_SAFE(derangement_known);
    RUN_TEST_SAFE(stirling_second_known);
    RUN_TEST_SAFE(stirling_first_known);
    RUN_TEST_SAFE(bell_known);
    RUN_TEST_SAFE(permutation_known);
    RUN_TEST_SAFE(partition_known);
    RUN_TEST_SAFE(ogf_evaluate);
TEST_MAIN_END()
