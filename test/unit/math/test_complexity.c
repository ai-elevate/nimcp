/**
 * @file test_complexity.c
 * @brief Tests for the computational complexity theory engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_complexity_theory.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    cplx_config_t cfg = complexity_theory_default_config();
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);
    complexity_theory_destroy(sim);
}

/* ---------- known problem classification ---------- */

TEST(sat_is_np_complete) {
    cplx_config_t cfg = complexity_theory_default_config();
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);
    complexity_theory_load_known_problems(sim);

    cplx_class_t cls = complexity_theory_get_class(sim, CPLX_PROB_SAT);
    ASSERT_EQ(cls, CPLX_CLASS_NP_COMPLETE);
    ASSERT_TRUE(complexity_theory_is_np_complete(sim, CPLX_PROB_SAT));

    complexity_theory_destroy(sim);
}

TEST(halting_is_undecidable) {
    cplx_config_t cfg = complexity_theory_default_config();
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);
    complexity_theory_load_known_problems(sim);

    cplx_class_t cls = complexity_theory_get_class(sim, CPLX_PROB_HALTING);
    ASSERT_EQ(cls, CPLX_CLASS_UNDECIDABLE);
    ASSERT_TRUE(complexity_theory_is_undecidable(sim, CPLX_PROB_HALTING));

    complexity_theory_destroy(sim);
}

TEST(sorting_is_in_P) {
    cplx_config_t cfg = complexity_theory_default_config();
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);
    complexity_theory_load_known_problems(sim);

    ASSERT_TRUE(complexity_theory_is_in_P(sim, CPLX_PROB_SORTING));
    cplx_class_t cls = complexity_theory_get_class(sim, CPLX_PROB_SORTING);
    ASSERT_EQ(cls, CPLX_CLASS_ONLOGN);

    complexity_theory_destroy(sim);
}

/* ---------- Master theorem ---------- */

TEST(master_theorem_merge_sort) {
    /* T(n) = 2T(n/2) + O(n) => O(n log n) */
    cplx_class_t cls = complexity_theory_master_theorem(2, 2, 1.0);
    ASSERT_EQ(cls, CPLX_CLASS_ONLOGN);
}

TEST(master_theorem_binary_search) {
    /* T(n) = T(n/2) + O(1) => O(log n) */
    cplx_class_t cls = complexity_theory_master_theorem(1, 2, 0.0);
    ASSERT_EQ(cls, CPLX_CLASS_OLOGN);
}

TEST(master_theorem_strassen) {
    /* T(n) = 8T(n/2) + O(n^2) => O(n^3) (case 1: a > b^c, 8 > 2^2=4) */
    cplx_class_t cls = complexity_theory_master_theorem(8, 2, 2.0);
    ASSERT_EQ(cls, CPLX_CLASS_ON3);
}

/* ---------- Turing machine ---------- */

TEST(tm_accepts_matching_string) {
    cplx_config_t cfg = complexity_theory_default_config();
    cfg.max_tm_steps = 1000;
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* Simple TM: accepts strings of 'a' (symbol 1) ending with blank (symbol 0)
       State 0: read 1 -> write 1, move right, stay in state 0
       State 0: read 0 -> accept (state 1) */
    complexity_theory_tm_init(sim, 3, 2);  /* 3 states, 2 symbols */
    sim->tm.accept_state = 1;
    sim->tm.reject_state = 2;

    complexity_theory_tm_add_transition(sim, 0, 1, 0, 1, +1); /* read 'a', go right */
    complexity_theory_tm_add_transition(sim, 0, 0, 1, 0, 0);  /* read blank, accept */

    /* Input: "aaa" = {1, 1, 1} */
    uint8_t input[] = {1, 1, 1};
    complexity_theory_tm_load_input(sim, input, 3);

    bool halted = complexity_theory_tm_run(sim, 100);
    ASSERT_TRUE(halted);
    ASSERT_TRUE(complexity_theory_tm_accepted(sim));

    complexity_theory_destroy(sim);
}

/* ---------- more known problems ---------- */

TEST(more_known_problems) {
    cplx_config_t cfg = complexity_theory_default_config();
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);
    complexity_theory_load_known_problems(sim);

    /* 3SAT is NP-complete */
    ASSERT_TRUE(complexity_theory_is_np_complete(sim, CPLX_PROB_3SAT));
    /* Clique is NP-complete */
    ASSERT_TRUE(complexity_theory_is_np_complete(sim, CPLX_PROB_CLIQUE));
    /* Shortest path is in P */
    ASSERT_TRUE(complexity_theory_is_in_P(sim, CPLX_PROB_SHORTEST_PATH));
    /* PCP is undecidable */
    ASSERT_TRUE(complexity_theory_is_undecidable(sim, CPLX_PROB_PCP));

    complexity_theory_destroy(sim);
}

/* ---------- class names ---------- */

TEST(class_names) {
    const char *name = complexity_theory_class_name(CPLX_CLASS_P);
    ASSERT_NOT_NULL(name);

    const char *np = complexity_theory_class_name(CPLX_CLASS_NP);
    ASSERT_NOT_NULL(np);

    const char *undec = complexity_theory_class_name(CPLX_CLASS_UNDECIDABLE);
    ASSERT_NOT_NULL(undec);
}

/* ---------- TM rejects ---------- */

TEST(tm_rejects) {
    cplx_config_t cfg = complexity_theory_default_config();
    cfg.max_tm_steps = 100;
    complexity_theory_sim_t *sim = complexity_theory_create(&cfg);
    ASSERT_NOT_NULL(sim);

    /* TM that only accepts symbol 1; rejects on symbol 2 */
    complexity_theory_tm_init(sim, 3, 3);
    sim->tm.accept_state = 1;
    sim->tm.reject_state = 2;

    complexity_theory_tm_add_transition(sim, 0, 1, 1, 1, 0);  /* accept on 1 */
    complexity_theory_tm_add_transition(sim, 0, 2, 2, 2, 0);  /* reject on 2 */

    uint8_t input[] = {2};
    complexity_theory_tm_load_input(sim, input, 1);

    bool halted = complexity_theory_tm_run(sim, 100);
    ASSERT_TRUE(halted);
    ASSERT_FALSE(complexity_theory_tm_accepted(sim));

    complexity_theory_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(sat_is_np_complete);
    RUN_TEST_SAFE(halting_is_undecidable);
    RUN_TEST_SAFE(sorting_is_in_P);
    RUN_TEST_SAFE(master_theorem_merge_sort);
    RUN_TEST_SAFE(master_theorem_binary_search);
    RUN_TEST_SAFE(master_theorem_strassen);
    RUN_TEST_SAFE(tm_accepts_matching_string);
    RUN_TEST_SAFE(more_known_problems);
    RUN_TEST_SAFE(class_names);
    RUN_TEST_SAFE(tm_rejects);
TEST_MAIN_END()
