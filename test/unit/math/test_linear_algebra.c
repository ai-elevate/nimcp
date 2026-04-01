/**
 * @file test_linear_algebra.c
 * @brief Tests for the dense linear algebra engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_linear_algebra.h"

static const double TOL = 1e-9;

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    matrix_t *m = matrix_create(3, 3);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->rows, 3);
    ASSERT_EQ(m->cols, 3);
    matrix_destroy(m);
}

TEST(identity_creation) {
    matrix_t *I = matrix_create_identity(4);
    ASSERT_NOT_NULL(I);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ASSERT_NEAR(matrix_get(I, i, j), (i == j) ? 1.0 : 0.0, TOL);
    matrix_destroy(I);
}

/* ---------- determinant ---------- */

TEST(identity_det_is_one) {
    matrix_t *I = matrix_create_identity(5);
    ASSERT_NOT_NULL(I);
    double det = matrix_determinant(I);
    ASSERT_NEAR(det, 1.0, TOL);
    matrix_destroy(I);
}

TEST(det_2x2) {
    matrix_t *m = matrix_create(2, 2);
    /* [[3,8],[4,6]] => det = 3*6 - 8*4 = -14 */
    matrix_set(m, 0, 0, 3.0); matrix_set(m, 0, 1, 8.0);
    matrix_set(m, 1, 0, 4.0); matrix_set(m, 1, 1, 6.0);
    ASSERT_NEAR(matrix_determinant(m), -14.0, TOL);
    matrix_destroy(m);
}

/* ---------- inverse ---------- */

TEST(inverse_2x2) {
    matrix_t *m = matrix_create(2, 2);
    /* [[4,7],[2,6]] => inv = [[0.6,-0.7],[-0.2,0.4]] */
    matrix_set(m, 0, 0, 4.0); matrix_set(m, 0, 1, 7.0);
    matrix_set(m, 1, 0, 2.0); matrix_set(m, 1, 1, 6.0);

    matrix_t *inv = matrix_inverse(m);
    ASSERT_NOT_NULL(inv);
    ASSERT_NEAR(matrix_get(inv, 0, 0),  0.6, TOL);
    ASSERT_NEAR(matrix_get(inv, 0, 1), -0.7, TOL);
    ASSERT_NEAR(matrix_get(inv, 1, 0), -0.2, TOL);
    ASSERT_NEAR(matrix_get(inv, 1, 1),  0.4, TOL);

    /* Verify M * M^-1 = I */
    matrix_t *prod = matrix_multiply(m, inv);
    ASSERT_NOT_NULL(prod);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ASSERT_NEAR(matrix_get(prod, i, j), (i == j) ? 1.0 : 0.0, TOL);

    matrix_destroy(prod);
    matrix_destroy(inv);
    matrix_destroy(m);
}

/* ---------- solve ---------- */

TEST(solve_ax_eq_b) {
    /* [[2,1],[5,3]] * x = [[4],[7]] => x = [[5],[-6]] */
    matrix_t *A = matrix_create(2, 2);
    matrix_set(A, 0, 0, 2.0); matrix_set(A, 0, 1, 1.0);
    matrix_set(A, 1, 0, 5.0); matrix_set(A, 1, 1, 3.0);

    matrix_t *b = matrix_create(2, 1);
    matrix_set(b, 0, 0, 4.0);
    matrix_set(b, 1, 0, 7.0);

    matrix_t *x = matrix_solve_lu(A, b);
    ASSERT_NOT_NULL(x);
    ASSERT_NEAR(matrix_get(x, 0, 0),  5.0, TOL);
    ASSERT_NEAR(matrix_get(x, 1, 0), -6.0, TOL);

    matrix_destroy(x);
    matrix_destroy(b);
    matrix_destroy(A);
}

/* ---------- eigenvalues ---------- */

TEST(eigen_diagonal) {
    /* Diagonal matrix: eigenvalues are the diagonal entries */
    matrix_t *D = matrix_create(3, 3);
    matrix_set(D, 0, 0, 2.0); matrix_set(D, 1, 1, 5.0); matrix_set(D, 2, 2, 8.0);

    eigen_result_t *eig = matrix_eigen_symmetric(D);
    ASSERT_NOT_NULL(eig);
    ASSERT_EQ(eig->n, 3);

    /* Sort eigenvalues for comparison (they may come in any order) */
    double evals[3];
    for (int i = 0; i < 3; i++) evals[i] = eig->eigenvalues[i];
    for (int i = 0; i < 2; i++)
        for (int j = i + 1; j < 3; j++)
            if (evals[i] > evals[j]) { double t = evals[i]; evals[i] = evals[j]; evals[j] = t; }

    ASSERT_NEAR(evals[0], 2.0, TOL);
    ASSERT_NEAR(evals[1], 5.0, TOL);
    ASSERT_NEAR(evals[2], 8.0, TOL);

    eigen_result_free(eig);
    matrix_destroy(D);
}

/* ---------- SVD ---------- */

TEST(svd_2x2) {
    matrix_t *A = matrix_create(2, 2);
    matrix_set(A, 0, 0, 3.0); matrix_set(A, 0, 1, 0.0);
    matrix_set(A, 1, 0, 0.0); matrix_set(A, 1, 1, 4.0);

    svd_result_t *svd = matrix_svd(A);
    ASSERT_NOT_NULL(svd);
    ASSERT_EQ(svd->n_sigma, 2);

    /* Singular values of diagonal matrix are the absolute diagonal values */
    double s0 = svd->sigma[0], s1 = svd->sigma[1];
    double smax = (s0 > s1) ? s0 : s1;
    double smin = (s0 < s1) ? s0 : s1;
    ASSERT_NEAR(smax, 4.0, TOL);
    ASSERT_NEAR(smin, 3.0, TOL);

    svd_result_free(svd);
    matrix_destroy(A);
}

/* ---------- QR orthogonality ---------- */

TEST(qr_orthogonality) {
    matrix_t *A = matrix_create(3, 3);
    matrix_set(A, 0, 0, 12.0); matrix_set(A, 0, 1, -51.0); matrix_set(A, 0, 2, 4.0);
    matrix_set(A, 1, 0,  6.0); matrix_set(A, 1, 1, 167.0); matrix_set(A, 1, 2, -68.0);
    matrix_set(A, 2, 0, -4.0); matrix_set(A, 2, 1,  24.0); matrix_set(A, 2, 2, -41.0);

    qr_decomp_t *qr = matrix_qr(A);
    ASSERT_NOT_NULL(qr);

    /* Q^T * Q should be identity */
    matrix_t *Qt = matrix_transpose(qr->Q);
    ASSERT_NOT_NULL(Qt);
    matrix_t *QtQ = matrix_multiply(Qt, qr->Q);
    ASSERT_NOT_NULL(QtQ);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ASSERT_NEAR(matrix_get(QtQ, i, j), (i == j) ? 1.0 : 0.0, 1e-6);

    matrix_destroy(QtQ);
    matrix_destroy(Qt);
    qr_decomp_free(qr);
    matrix_destroy(A);
}

/* ---------- condition number ---------- */

TEST(condition_number_identity) {
    matrix_t *I = matrix_create_identity(4);
    ASSERT_NOT_NULL(I);
    double cond = matrix_condition_number(I);
    ASSERT_NEAR(cond, 1.0, TOL);
    matrix_destroy(I);
}

/* ---------- rank ---------- */

TEST(rank_full) {
    matrix_t *I = matrix_create_identity(3);
    ASSERT_NOT_NULL(I);
    ASSERT_EQ(matrix_rank(I), 3);
    matrix_destroy(I);
}

/* ---------- trace ---------- */

TEST(trace_identity) {
    matrix_t *I = matrix_create_identity(5);
    ASSERT_NOT_NULL(I);
    ASSERT_NEAR(matrix_trace(I), 5.0, TOL);
    matrix_destroy(I);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(identity_creation);
    RUN_TEST_SAFE(identity_det_is_one);
    RUN_TEST_SAFE(det_2x2);
    RUN_TEST_SAFE(inverse_2x2);
    RUN_TEST_SAFE(solve_ax_eq_b);
    RUN_TEST_SAFE(eigen_diagonal);
    RUN_TEST_SAFE(svd_2x2);
    RUN_TEST_SAFE(qr_orthogonality);
    RUN_TEST_SAFE(condition_number_identity);
    RUN_TEST_SAFE(rank_full);
    RUN_TEST_SAFE(trace_identity);
TEST_MAIN_END()
