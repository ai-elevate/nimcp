/**
 * @file test_numerical_methods.c
 * @brief Tests for the numerical methods engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_numerical_methods.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double TOL = 1e-6;

/* --- helper functions --- */

static double f_x2_minus_2(double x, void *p) {
    (void)p;
    return x * x - 2.0;   /* root at sqrt(2) */
}

static double f_x3_minus_27(double x, void *p) {
    (void)p;
    return x * x * x - 27.0;  /* root at 3 */
}

static double df_x3_minus_27(double x, void *p) {
    (void)p;
    return 3.0 * x * x;
}

static double f_x_squared(double x, void *p) {
    (void)p;
    return x * x;
}

static double f_sin(double x, void *p) {
    (void)p;
    return sin(x);
}

/* ---------- lifecycle (stateless, no create/destroy) ---------- */

/* ---------- bisection finds sqrt(2) ---------- */

TEST(bisection_sqrt2) {
    root_result_t r = num_bisection(f_x2_minus_2, NULL, 1.0, 2.0, 1e-12, 100);
    ASSERT_TRUE(r.converged);
    ASSERT_NEAR(r.root, sqrt(2.0), 1e-10);
}

/* ---------- Newton finds cube root of 27 = 3 ---------- */

TEST(newton_cbrt27) {
    root_result_t r = num_newton(f_x3_minus_27, df_x3_minus_27, NULL,
                                 4.0, 1e-12, 100);
    ASSERT_TRUE(r.converged);
    ASSERT_NEAR(r.root, 3.0, 1e-10);
}

/* ---------- secant method ---------- */

TEST(secant_sqrt2) {
    root_result_t r = num_secant(f_x2_minus_2, NULL, 1.0, 2.0, 1e-12, 100);
    ASSERT_TRUE(r.converged);
    ASSERT_NEAR(r.root, sqrt(2.0), 1e-10);
}

/* ---------- Brent method ---------- */

TEST(brent_sqrt2) {
    root_result_t r = num_brent(f_x2_minus_2, NULL, 1.0, 2.0, 1e-12, 100);
    ASSERT_TRUE(r.converged);
    ASSERT_NEAR(r.root, sqrt(2.0), 1e-10);
}

/* ---------- Lagrange interpolation exact for quadratic ---------- */

TEST(lagrange_quadratic) {
    /* f(x) = x^2 at points 0, 1, 2 => exact for any query */
    double xs[] = {0.0, 1.0, 2.0};
    double ys[] = {0.0, 1.0, 4.0};

    ASSERT_NEAR(num_lagrange_interp(xs, ys, 3, 0.5), 0.25, TOL);
    ASSERT_NEAR(num_lagrange_interp(xs, ys, 3, 1.5), 2.25, TOL);
    ASSERT_NEAR(num_lagrange_interp(xs, ys, 3, 3.0), 9.0, TOL);
}

/* ---------- Simpson integrates x^2 from 0 to 1 = 1/3 ---------- */

TEST(simpson_x_squared) {
    double result = num_simpson(f_x_squared, NULL, 0.0, 1.0, 100);
    ASSERT_NEAR(result, 1.0 / 3.0, TOL);
}

/* ---------- trapezoidal rule ---------- */

TEST(trapezoidal_x_squared) {
    double result = num_trapezoidal(f_x_squared, NULL, 0.0, 1.0, 1000);
    ASSERT_NEAR(result, 1.0 / 3.0, 1e-4);
}

/* ---------- FFT of sine wave ---------- */

TEST(fft_sine_peak) {
    /* Create a 64-sample sine wave at frequency k=4 */
    int N = 64;
    complex_d_t input[64];
    for (int i = 0; i < N; i++) {
        input[i].re = sin(2.0 * M_PI * 4.0 * i / N);
        input[i].im = 0.0;
    }

    fft_result_t *fft = num_fft(input, N);
    ASSERT_NOT_NULL(fft);
    ASSERT_EQ(fft->n, N);

    /* Find the bin with maximum magnitude */
    double max_mag = 0.0;
    int max_bin = 0;
    for (int i = 0; i < N / 2; i++) {
        double mag = sqrt(fft->data[i].re * fft->data[i].re +
                          fft->data[i].im * fft->data[i].im);
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = i;
        }
    }
    /* Peak should be at bin 4 (frequency index 4) */
    ASSERT_EQ(max_bin, 4);

    num_fft_free(fft);
}

/* ---------- cubic spline ---------- */

TEST(cubic_spline_interpolation) {
    double xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};  /* x^2 */

    cubic_spline_t *sp = num_cubic_spline_create(xs, ys, 5);
    ASSERT_NOT_NULL(sp);

    /* Spline of x^2 data should be close to x^2 at intermediate points */
    ASSERT_NEAR(num_cubic_spline_eval(sp, 0.5), 0.25, 0.1);
    ASSERT_NEAR(num_cubic_spline_eval(sp, 1.5), 2.25, 0.1);
    ASSERT_NEAR(num_cubic_spline_eval(sp, 2.5), 6.25, 0.1);

    num_cubic_spline_free(sp);
}

/* ---------- numerical differentiation ---------- */

TEST(numerical_derivative_sin) {
    /* d/dx sin(x) at x=0 should be cos(0)=1 */
    double d = num_deriv_5point(f_sin, NULL, 0.0, 1e-4);
    ASSERT_NEAR(d, 1.0, 1e-6);
}

/* ---------- Romberg integration ---------- */

TEST(romberg_x_squared) {
    romberg_result_t r = num_romberg(f_x_squared, NULL, 0.0, 1.0, 8, 1e-12);
    ASSERT_NEAR(r.result, 1.0 / 3.0, 1e-10);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(bisection_sqrt2);
    RUN_TEST_SAFE(newton_cbrt27);
    RUN_TEST_SAFE(secant_sqrt2);
    RUN_TEST_SAFE(brent_sqrt2);
    RUN_TEST_SAFE(lagrange_quadratic);
    RUN_TEST_SAFE(simpson_x_squared);
    RUN_TEST_SAFE(trapezoidal_x_squared);
    RUN_TEST_SAFE(fft_sine_peak);
    RUN_TEST_SAFE(cubic_spline_interpolation);
    RUN_TEST_SAFE(numerical_derivative_sin);
    RUN_TEST_SAFE(romberg_x_squared);
TEST_MAIN_END()
