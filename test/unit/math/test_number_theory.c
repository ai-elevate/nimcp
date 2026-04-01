/**
 * @file test_number_theory.c
 * @brief Tests for the number theory engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_number_theory.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    number_theory_t *nt = nt_create();
    ASSERT_NOT_NULL(nt);
    nt_destroy(nt);
}

/* ---------- primality ---------- */

TEST(is_prime_true) {
    number_theory_t *nt = nt_create();
    ASSERT_NOT_NULL(nt);
    ASSERT_TRUE(nt_is_prime(nt, 997));
    ASSERT_TRUE(nt_is_prime(nt, 2));
    ASSERT_TRUE(nt_is_prime(nt, 3));
    ASSERT_TRUE(nt_is_prime(nt, 7919));
    nt_destroy(nt);
}

TEST(is_prime_false) {
    number_theory_t *nt = nt_create();
    ASSERT_NOT_NULL(nt);
    ASSERT_FALSE(nt_is_prime(nt, 998));
    ASSERT_FALSE(nt_is_prime(nt, 0));
    ASSERT_FALSE(nt_is_prime(nt, 1));
    ASSERT_FALSE(nt_is_prime(nt, 100));
    nt_destroy(nt);
}

TEST(is_prime_trial_division) {
    ASSERT_TRUE(nt_is_prime_trial(997));
    ASSERT_FALSE(nt_is_prime_trial(998));
    ASSERT_TRUE(nt_is_prime_trial(104729));
    ASSERT_FALSE(nt_is_prime_trial(104730));
}

/* ---------- GCD / LCM ---------- */

TEST(gcd_known_values) {
    ASSERT_EQ(nt_gcd(12, 8), 4);
    ASSERT_EQ(nt_gcd(100, 75), 25);
    ASSERT_EQ(nt_gcd(17, 13), 1);
    ASSERT_EQ(nt_gcd(0, 5), 5);
    ASSERT_EQ(nt_gcd(48, 18), 6);
}

TEST(lcm_known_values) {
    ASSERT_EQ(nt_lcm(4, 6), 12);
    ASSERT_EQ(nt_lcm(12, 8), 24);
    ASSERT_EQ(nt_lcm(7, 13), 91);
}

/* ---------- totient ---------- */

TEST(euler_totient) {
    number_theory_t *nt = nt_create();
    ASSERT_NOT_NULL(nt);
    ASSERT_EQ(nt_euler_totient(nt, 12), 4);
    ASSERT_EQ(nt_euler_totient(nt, 1), 1);
    ASSERT_EQ(nt_euler_totient(nt, 7), 6);   /* prime: phi(p) = p-1 */
    ASSERT_EQ(nt_euler_totient(nt, 10), 4);
    nt_destroy(nt);
}

/* ---------- modular exponentiation ---------- */

TEST(mod_pow_known) {
    ASSERT_EQ(nt_mod_pow(2, 10, 1000), 24);  /* 1024 mod 1000 = 24 */
    ASSERT_EQ(nt_mod_pow(3, 5, 13), 9);      /* 243 mod 13 = 9 */
    ASSERT_EQ(nt_mod_pow(2, 0, 100), 1);     /* anything^0 = 1 */
    ASSERT_EQ(nt_mod_pow(7, 1, 5), 2);       /* 7 mod 5 = 2 */
}

/* ---------- CRT ---------- */

TEST(crt_solve) {
    /* x = 2 mod 3, x = 3 mod 5, x = 2 mod 7 => x = 23 mod 105 */
    crt_system_t crt;
    memset(&crt, 0, sizeof(crt));
    crt.remainders[0] = 2; crt.moduli[0] = 3;
    crt.remainders[1] = 3; crt.moduli[1] = 5;
    crt.remainders[2] = 2; crt.moduli[2] = 7;
    crt.count = 3;

    bool ok = nt_crt_solve(&crt);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(crt.valid);
    ASSERT_EQ(crt.solution, 23);
    ASSERT_EQ(crt.combined_modulus, 105);
}

/* ---------- factorization ---------- */

TEST(factorize_60) {
    number_theory_t *nt = nt_create();
    ASSERT_NOT_NULL(nt);
    factorization_t f = nt_factorize(nt, 60);
    /* 60 = 2^2 * 3 * 5 => 3 distinct primes */
    ASSERT_EQ(f.count, 3);
    ASSERT_EQ(f.factors[0].prime, 2);
    ASSERT_EQ(f.factors[0].exponent, 2);
    ASSERT_EQ(f.factors[1].prime, 3);
    ASSERT_EQ(f.factors[1].exponent, 1);
    ASSERT_EQ(f.factors[2].prime, 5);
    ASSERT_EQ(f.factors[2].exponent, 1);
    nt_destroy(nt);
}

/* ---------- extended GCD ---------- */

TEST(extended_gcd) {
    ext_gcd_result_t r = nt_extended_gcd(35, 15);
    ASSERT_EQ(r.gcd, 5);
    /* Verify Bezout identity: 35*x + 15*y = 5 */
    ASSERT_EQ(35 * r.x + 15 * r.y, 5);
}

/* ---------- sieve ---------- */

TEST(sieve_prime_count) {
    number_theory_t *nt = nt_create();
    ASSERT_NOT_NULL(nt);
    ASSERT_TRUE(nt_build_sieve(nt, 1000));
    /* There are 168 primes <= 1000 */
    ASSERT_EQ(nt_prime_count(nt, 1000), 168);
    ASSERT_TRUE(nt_is_prime_sieve(nt, 997));
    ASSERT_FALSE(nt_is_prime_sieve(nt, 998));
    nt_destroy(nt);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(is_prime_true);
    RUN_TEST_SAFE(is_prime_false);
    RUN_TEST_SAFE(is_prime_trial_division);
    RUN_TEST_SAFE(gcd_known_values);
    RUN_TEST_SAFE(lcm_known_values);
    RUN_TEST_SAFE(euler_totient);
    RUN_TEST_SAFE(mod_pow_known);
    RUN_TEST_SAFE(crt_solve);
    RUN_TEST_SAFE(factorize_60);
    RUN_TEST_SAFE(extended_gcd);
    RUN_TEST_SAFE(sieve_prime_count);
TEST_MAIN_END()
