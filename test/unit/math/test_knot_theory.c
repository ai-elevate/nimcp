/**
 * @file test_knot_theory.c
 * @brief Tests for the knot theory engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_knot_theory.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    knot_config_t cfg = knot_theory_default_config();
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);
    knot_theory_destroy(eng);
}

/* ---------- load known knots ---------- */

TEST(load_unknot) {
    knot_config_t cfg = knot_theory_default_config();
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t id = knot_load_known(eng, KNOT_UNKNOT);
    ASSERT_TRUE(knot_is_unknot(eng, id));

    knot_theory_destroy(eng);
}

TEST(load_trefoil) {
    knot_config_t cfg = knot_theory_default_config();
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t id = knot_load_known(eng, KNOT_TREFOIL);
    ASSERT_FALSE(knot_is_unknot(eng, id));

    knot_theory_destroy(eng);
}

/* ---------- writhe ---------- */

TEST(unknot_writhe_zero) {
    knot_config_t cfg = knot_theory_default_config();
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t id = knot_load_known(eng, KNOT_UNKNOT);
    int32_t w = knot_writhe(&eng->diagrams[id]);
    ASSERT_EQ(w, 0);

    knot_theory_destroy(eng);
}

TEST(trefoil_writhe) {
    knot_config_t cfg = knot_theory_default_config();
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t id = knot_load_known(eng, KNOT_TREFOIL);
    int32_t w = knot_writhe(&eng->diagrams[id]);
    /* Trefoil writhe is +3 or -3 depending on orientation */
    ASSERT_TRUE(w == 3 || w == -3);

    knot_theory_destroy(eng);
}

/* ---------- Alexander polynomial ---------- */

TEST(trefoil_alexander) {
    knot_config_t cfg = knot_theory_default_config();
    cfg.compute_alexander = true;
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t id = knot_load_known(eng, KNOT_TREFOIL);
    knot_polynomial_t alex = knot_alexander_polynomial(eng, id);

    /* Trefoil Alexander: Delta(t) = t - 1 + t^{-1} (or equivalent)
       Evaluate at t=1: Delta(1) = 1 (always for knots) */
    double val_at_1 = knot_poly_eval(&alex, 1.0);
    ASSERT_NEAR(val_at_1, 1.0, 1e-6);

    /* Evaluate at t=-1: |Delta(-1)| = determinant = 3 for trefoil */
    double val_at_neg1 = knot_poly_eval(&alex, -1.0);
    ASSERT_NEAR(fabs(val_at_neg1), 3.0, 1e-6);

    knot_theory_destroy(eng);
}

/* ---------- from Gauss code ---------- */

TEST(gauss_code_creation) {
    knot_config_t cfg = knot_theory_default_config();
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    /* Trefoil Gauss code: 1 -2 3 -1 2 -3 */
    int32_t gauss[] = {1, -2, 3, -1, 2, -3};
    uint32_t id = knot_from_gauss_code(eng, gauss, 6, "trefoil_gauss");

    ASSERT_EQ(eng->diagrams[id].num_crossings, 3);

    knot_theory_destroy(eng);
}

/* ---------- braid operations ---------- */

TEST(braid_compose_length) {
    /* Braid a: sigma1, sigma2 (length 2)
       Braid b: sigma1^-1 (length 1)
       Compose: length = 2 + 1 = 3 */
    knot_braid_t a = {{0}, 0, 3};
    a.generators[0] = 1;
    a.generators[1] = 2;
    a.length = 2;

    knot_braid_t b = {{0}, 0, 3};
    b.generators[0] = -1;
    b.length = 1;

    knot_braid_t ab = knot_braid_compose(&a, &b);
    ASSERT_EQ(ab.length, 3);
    ASSERT_EQ(knot_braid_length(&ab), 3);
    ASSERT_EQ(ab.generators[0], 1);
    ASSERT_EQ(ab.generators[1], 2);
    ASSERT_EQ(ab.generators[2], -1);
}

/* ---------- braid inverse ---------- */

TEST(braid_inverse) {
    knot_braid_t b = {{0}, 0, 3};
    b.generators[0] = 1;
    b.generators[1] = -2;
    b.generators[2] = 1;
    b.length = 3;

    knot_braid_t inv = knot_braid_inverse(&b);
    ASSERT_EQ(inv.length, 3);
    /* Inverse reverses order and negates each generator */
    ASSERT_EQ(inv.generators[0], -1);
    ASSERT_EQ(inv.generators[1], 2);
    ASSERT_EQ(inv.generators[2], -1);
}

/* ---------- polynomial operations ---------- */

TEST(polynomial_eval) {
    knot_polynomial_t p;
    memset(&p, 0, sizeof(p));
    /* 1 + 2t + 3t^2, min_exp=0 */
    p.min_exp = 0;
    p.num_terms = 3;
    p.coeffs[0] = 1.0;
    p.coeffs[1] = 2.0;
    p.coeffs[2] = 3.0;

    /* At t=2: 1 + 4 + 12 = 17 */
    ASSERT_NEAR(knot_poly_eval(&p, 2.0), 17.0, 1e-9);
    ASSERT_NEAR(knot_poly_eval(&p, 0.0), 1.0, 1e-9);
}

/* ---------- invariants ---------- */

TEST(figure_eight_invariants) {
    knot_config_t cfg = knot_theory_default_config();
    cfg.compute_alexander = true;
    knot_theory_engine_t *eng = knot_theory_create(&cfg);
    ASSERT_NOT_NULL(eng);

    uint32_t id = knot_load_known(eng, KNOT_FIGURE_8);
    knot_invariants_t inv = knot_compute_invariants(eng, id);

    /* Figure-8 is alternating and amphicheiral */
    ASSERT_TRUE(inv.is_alternating);
    /* Crossing number = 4 */
    ASSERT_EQ(inv.crossing_number, 4);
    /* Writhe of standard figure-8 diagram is 0 */
    ASSERT_EQ(inv.writhe, 0);

    knot_theory_destroy(eng);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(load_unknot);
    RUN_TEST_SAFE(load_trefoil);
    RUN_TEST_SAFE(unknot_writhe_zero);
    RUN_TEST_SAFE(trefoil_writhe);
    RUN_TEST_SAFE(trefoil_alexander);
    RUN_TEST_SAFE(gauss_code_creation);
    RUN_TEST_SAFE(braid_compose_length);
    RUN_TEST_SAFE(braid_inverse);
    RUN_TEST_SAFE(polynomial_eval);
    RUN_TEST_SAFE(figure_eight_invariants);
TEST_MAIN_END()
