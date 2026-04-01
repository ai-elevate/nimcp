/**
 * @file test_logic.c
 * @brief Tests for the propositional and predicate logic engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_logic.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    logic_engine_t *eng = logic_engine_create();
    ASSERT_NOT_NULL(eng);
    logic_engine_destroy(eng);
}

/* ---------- formula construction ---------- */

TEST(formula_creation) {
    logic_formula_t *p = logic_var(0);
    ASSERT_NOT_NULL(p);
    ASSERT_EQ(p->op, LOGIC_OP_VAR);
    ASSERT_EQ(p->var_index, 0);

    logic_formula_t *t = logic_const(true);
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(t->op, LOGIC_OP_CONST);
    ASSERT_TRUE(t->const_value);

    logic_formula_free(p);
    logic_formula_free(t);
}

/* ---------- tautology: P v ~P = true ---------- */

TEST(tautology_p_or_not_p) {
    logic_formula_t *p = logic_var(0);
    logic_formula_t *not_p = logic_not(logic_formula_clone(p));
    logic_formula_t *f = logic_binary(LOGIC_OP_OR, p, not_p);

    ASSERT_TRUE(logic_is_tautology(f, 1));
    logic_formula_free(f);
}

/* ---------- satisfiable: P ^ Q ---------- */

TEST(satisfiable_p_and_q) {
    logic_formula_t *p = logic_var(0);
    logic_formula_t *q = logic_var(1);
    logic_formula_t *f = logic_binary(LOGIC_OP_AND, p, q);

    ASSERT_TRUE(logic_is_satisfiable(f, 2));
    /* Not a tautology: false when P=false or Q=false */
    ASSERT_FALSE(logic_is_tautology(f, 2));
    logic_formula_free(f);
}

/* ---------- unsatisfiable: P ^ ~P = false ---------- */

TEST(unsatisfiable_p_and_not_p) {
    logic_formula_t *p = logic_var(0);
    logic_formula_t *not_p = logic_not(logic_formula_clone(p));
    logic_formula_t *f = logic_binary(LOGIC_OP_AND, p, not_p);

    ASSERT_FALSE(logic_is_satisfiable(f, 1));
    logic_formula_free(f);
}

/* ---------- modus ponens ---------- */

TEST(modus_ponens_valid) {
    /* Premises: P, P->Q.  Conclusion: Q */
    logic_formula_t *p = logic_var(0);
    logic_formula_t *q = logic_var(1);
    logic_formula_t *p_implies_q = logic_binary(LOGIC_OP_IMPLIES,
                                                 logic_formula_clone(p),
                                                 logic_formula_clone(q));

    const logic_formula_t *premises[2] = { p, p_implies_q };
    bool valid = logic_verify_inference(premises, 2, q, RULE_MODUS_PONENS, 2);
    ASSERT_TRUE(valid);

    logic_formula_free(p);
    logic_formula_free(q);
    logic_formula_free(p_implies_q);
}

/* ---------- evaluation ---------- */

TEST(evaluate_formula) {
    /* P AND Q: true when both true */
    logic_formula_t *p = logic_var(0);
    logic_formula_t *q = logic_var(1);
    logic_formula_t *f = logic_binary(LOGIC_OP_AND,
                                       logic_formula_clone(p),
                                       logic_formula_clone(q));

    bool assign_tt[] = {true, true};
    bool assign_tf[] = {true, false};
    bool assign_ff[] = {false, false};

    ASSERT_TRUE(logic_evaluate(f, assign_tt));
    ASSERT_FALSE(logic_evaluate(f, assign_tf));
    ASSERT_FALSE(logic_evaluate(f, assign_ff));

    logic_formula_free(f);
    logic_formula_free(p);
    logic_formula_free(q);
}

/* ---------- truth table ---------- */

TEST(truth_table) {
    logic_formula_t *p = logic_var(0);
    logic_formula_t *not_p = logic_not(logic_formula_clone(p));
    logic_formula_t *f = logic_binary(LOGIC_OP_OR, p, not_p);

    truth_table_t *tt = logic_truth_table(f, 1);
    ASSERT_NOT_NULL(tt);
    ASSERT_EQ(tt->n_variables, 1);
    ASSERT_EQ(tt->n_rows, 2);
    /* All rows should be true (tautology) */
    ASSERT_TRUE(tt->results[0]);
    ASSERT_TRUE(tt->results[1]);

    logic_truth_table_free(tt);
    logic_formula_free(f);
}

/* ---------- equivalence ---------- */

TEST(equivalence_demorgan) {
    /* De Morgan: ~(P^Q) <=> (~P v ~Q) */
    logic_formula_t *p = logic_var(0);
    logic_formula_t *q = logic_var(1);

    logic_formula_t *lhs = logic_not(
        logic_binary(LOGIC_OP_AND,
                     logic_formula_clone(p),
                     logic_formula_clone(q)));

    logic_formula_t *rhs = logic_binary(LOGIC_OP_OR,
        logic_not(logic_formula_clone(p)),
        logic_not(logic_formula_clone(q)));

    ASSERT_TRUE(logic_are_equivalent(lhs, rhs, 2));

    logic_formula_free(lhs);
    logic_formula_free(rhs);
    logic_formula_free(p);
    logic_formula_free(q);
}

/* ---------- implication tautology ---------- */

TEST(implication_tautology) {
    /* P -> P is a tautology */
    logic_formula_t *p = logic_var(0);
    logic_formula_t *f = logic_binary(LOGIC_OP_IMPLIES,
                                       logic_formula_clone(p),
                                       logic_formula_clone(p));
    ASSERT_TRUE(logic_is_tautology(f, 1));
    logic_formula_free(f);
    logic_formula_free(p);
}

/* ---------- modus tollens ---------- */

TEST(modus_tollens_valid) {
    /* Premises: ~Q, P->Q.  Conclusion: ~P */
    logic_formula_t *p = logic_var(0);
    logic_formula_t *q = logic_var(1);
    logic_formula_t *not_q = logic_not(logic_formula_clone(q));
    logic_formula_t *p_implies_q = logic_binary(LOGIC_OP_IMPLIES,
                                                 logic_formula_clone(p),
                                                 logic_formula_clone(q));
    logic_formula_t *not_p = logic_not(logic_formula_clone(p));

    const logic_formula_t *premises[2] = { not_q, p_implies_q };
    bool valid = logic_verify_inference(premises, 2, not_p, RULE_MODUS_TOLLENS, 2);
    ASSERT_TRUE(valid);

    logic_formula_free(p);
    logic_formula_free(q);
    logic_formula_free(not_q);
    logic_formula_free(p_implies_q);
    logic_formula_free(not_p);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(formula_creation);
    RUN_TEST_SAFE(tautology_p_or_not_p);
    RUN_TEST_SAFE(satisfiable_p_and_q);
    RUN_TEST_SAFE(unsatisfiable_p_and_not_p);
    RUN_TEST_SAFE(modus_ponens_valid);
    RUN_TEST_SAFE(evaluate_formula);
    RUN_TEST_SAFE(truth_table);
    RUN_TEST_SAFE(equivalence_demorgan);
    RUN_TEST_SAFE(implication_tautology);
    RUN_TEST_SAFE(modus_tollens_valid);
TEST_MAIN_END()
