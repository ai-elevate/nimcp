/**
 * @file test_biochemistry.c
 * @brief Tests for the biochemistry simulation engine
 *
 * Validates Michaelis-Menten kinetics, Hill equation cooperativity,
 * competitive inhibition, glycolysis loading, enzyme/metabolite management.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_biochemistry.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    biochem_config_t cfg = biochemistry_default_config();
    biochemistry_sim_t* sim = biochemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    biochemistry_destroy(sim);
}

/* ---- Michaelis-Menten: v = Vmax*[S]/(Km+[S]) ---------------------------- */

TEST(michaelis_menten_half_vmax) {
    /* When [S] = Km, v = Vmax/2 */
    float v = biochemistry_michaelis_menten(1.0f, 1.0f, 1.0f);
    ASSERT_NEAR(v, 0.5f, 0.001f);
}

TEST(michaelis_menten_saturation) {
    /* Very high [S] -> v ~ Vmax */
    float v = biochemistry_michaelis_menten(1.0f, 1.0f, 10000.0f);
    ASSERT_NEAR(v, 1.0f, 0.01f);
}

TEST(michaelis_menten_low_substrate) {
    /* Very low [S] -> v ~ Vmax*[S]/Km (linear) */
    float v = biochemistry_michaelis_menten(1.0f, 1.0f, 0.01f);
    ASSERT_NEAR(v, 0.01f / 1.01f, 0.001f);
}

/* ---- Hill equation: v = Vmax*[S]^n/(K^n+[S]^n) -------------------------- */

TEST(hill_cooperativity_n2) {
    /* n=2: more sigmoidal than MM */
    /* At [S]=K: v = Vmax * K^2/(K^2+K^2) = Vmax/2 */
    float v_at_k = biochemistry_hill_equation(1.0f, 1.0f, 1.0f, 2.0f);
    ASSERT_NEAR(v_at_k, 0.5f, 0.001f);
}

TEST(hill_steeper_than_mm) {
    /* n=2 Hill curve should be steeper around K than MM (n=1) */
    /* Compare slopes: at [S]=0.5, Hill(n=2) < MM; at [S]=2.0, Hill(n=2) > MM */
    float mm_low = biochemistry_michaelis_menten(1.0f, 1.0f, 0.5f);
    float hill_low = biochemistry_hill_equation(1.0f, 1.0f, 0.5f, 2.0f);
    ASSERT_LT(hill_low, mm_low);  /* Hill is lower at sub-K */

    float mm_high = biochemistry_michaelis_menten(1.0f, 1.0f, 2.0f);
    float hill_high = biochemistry_hill_equation(1.0f, 1.0f, 2.0f, 2.0f);
    ASSERT_GT(hill_high, mm_high); /* Hill is higher at supra-K */
}

/* ---- competitive inhibition reduces rate --------------------------------- */

TEST(competitive_inhibition_reduces_rate) {
    float Vmax = 1.0f, Km = 1.0f, S = 1.0f;
    float v_no_inhib = biochemistry_michaelis_menten(Vmax, Km, S);
    /* Competitive inhibitor: apparent Km increases, so rate decreases */
    float v_with_inhib = biochemistry_competitive_inhibition(Vmax, Km, S, 1.0f, 1.0f);
    ASSERT_LT(v_with_inhib, v_no_inhib);
}

TEST(competitive_inhibition_known_value) {
    /* v = Vmax*S / (Km*(1+I/Ki) + S) */
    /* Vmax=1, Km=1, S=1, I=1, Ki=1: v = 1/(1*(1+1)+1) = 1/3 */
    float v = biochemistry_competitive_inhibition(1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    ASSERT_NEAR(v, 1.0f / 3.0f, 0.01f);
}

/* ---- glycolysis loads with enzymes --------------------------------------- */

TEST(glycolysis_loads_enzymes) {
    biochem_config_t cfg = biochemistry_default_config();
    biochemistry_sim_t* sim = biochemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    biochemistry_load_glycolysis(sim);
    /* Glycolysis has 10 enzymatic steps */
    ASSERT_GE(sim->num_enzymes, 10);
    ASSERT_GE(sim->num_pathways, 1);

    biochemistry_destroy(sim);
}

/* ---- step simulation ----------------------------------------------------- */

TEST(step_basic) {
    biochem_config_t cfg = biochemistry_default_config();
    biochemistry_sim_t* sim = biochemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    biochemistry_load_glycolysis(sim);
    int rc = biochemistry_step(sim, 0.01f);
    ASSERT_EQ(rc, 0);

    biochemistry_destroy(sim);
}

/* ---- pH activity --------------------------------------------------------- */

TEST(ph_activity_optimum) {
    /* Activity at optimum pH should be near maximum */
    /* Using pK1=5.0, pK2=9.0, pH=7.0 (near optimum) */
    float act_opt = biochemistry_ph_activity(7.0f, 5.0f, 9.0f);
    float act_extreme = biochemistry_ph_activity(2.0f, 5.0f, 9.0f);
    ASSERT_GT(act_opt, act_extreme);
}

/* ---- add metabolite / enzyme --------------------------------------------- */

TEST(add_metabolite_and_enzyme) {
    biochem_config_t cfg = biochemistry_default_config();
    biochemistry_sim_t* sim = biochemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    biochem_metabolite_t met = {0};
    snprintf(met.name, BIOCHEM_MAX_NAME, "glucose");
    met.concentration = 5.0f;
    met.molecular_weight = 180.16f;
    met.active = true;
    uint32_t mid = biochemistry_add_metabolite(sim, &met);

    biochem_enzyme_t enz = {0};
    snprintf(enz.name, BIOCHEM_MAX_NAME, "hexokinase");
    enz.Vmax = 100.0f;
    enz.Km = 0.1f;
    enz.substrate_id = mid;
    enz.hill_coefficient = 1.0f;
    enz.active = true;
    biochemistry_add_enzyme(sim, &enz);

    ASSERT_EQ(sim->num_metabolites, 1);
    ASSERT_EQ(sim->num_enzymes, 1);

    biochemistry_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(michaelis_menten_half_vmax);
    RUN_TEST_SAFE(michaelis_menten_saturation);
    RUN_TEST_SAFE(michaelis_menten_low_substrate);
    RUN_TEST_SAFE(hill_cooperativity_n2);
    RUN_TEST_SAFE(hill_steeper_than_mm);
    RUN_TEST_SAFE(competitive_inhibition_reduces_rate);
    RUN_TEST_SAFE(competitive_inhibition_known_value);
    RUN_TEST_SAFE(glycolysis_loads_enzymes);
    RUN_TEST_SAFE(step_basic);
    RUN_TEST_SAFE(ph_activity_optimum);
    RUN_TEST_SAFE(add_metabolite_and_enzyme);
TEST_MAIN_END()
