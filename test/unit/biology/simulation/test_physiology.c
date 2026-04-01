/**
 * @file test_physiology.c
 * @brief Tests for the physiology simulation engine
 *
 * Validates Hill equation O2 dissociation, cardiac output, Henderson-Hasselbalch
 * blood pH, GFR, Poiseuille flow, acid-base classification.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_physiology.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    physiology_config_t cfg = physiology_default_config();
    physiology_sim_t* sim = physiology_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    physiology_destroy(sim);
}

/* ---- Hill equation: SO2 = pO2^n / (P50^n + pO2^n) ----------------------- */

TEST(hill_equation_at_p50) {
    /* At pO2 = P50, saturation = 0.5 */
    float so2 = physiology_hill_equation(PHYSIO_P50_MMHG, PHYSIO_P50_MMHG,
                                          PHYSIO_HILL_N);
    ASSERT_NEAR(so2, 0.5f, 0.01f);
}

TEST(hill_equation_high_po2) {
    /* At pO2 = 100 mmHg, saturation should be near 1.0 (~0.97) */
    float so2 = physiology_hill_equation(100.0f, PHYSIO_P50_MMHG, PHYSIO_HILL_N);
    ASSERT_GT(so2, 0.95f);
    ASSERT_LE(so2, 1.0f);
}

TEST(hill_equation_low_po2) {
    /* At pO2 = 10 mmHg, saturation should be very low */
    float so2 = physiology_hill_equation(10.0f, PHYSIO_P50_MMHG, PHYSIO_HILL_N);
    ASSERT_LT(so2, 0.2f);
    ASSERT_GT(so2, 0.0f);
}

TEST(hill_sigmoidal_shape) {
    /* Verify sigmoidal: slope at P50 > slope at extremes */
    float so2_low = physiology_hill_equation(10.0f, PHYSIO_P50_MMHG, PHYSIO_HILL_N);
    float so2_p50 = physiology_hill_equation(PHYSIO_P50_MMHG, PHYSIO_P50_MMHG,
                                              PHYSIO_HILL_N);
    float so2_high = physiology_hill_equation(100.0f, PHYSIO_P50_MMHG, PHYSIO_HILL_N);
    /* Should be monotonically increasing */
    ASSERT_LT(so2_low, so2_p50);
    ASSERT_LT(so2_p50, so2_high);
}

/* ---- cardiac output: CO = HR * SV / 1000 -------------------------------- */

TEST(cardiac_output_resting) {
    /* HR=72 bpm, SV=70 mL -> CO = 72*70/1000 = 5.04 L/min */
    float co = physiology_cardiac_output(PHYSIO_RESTING_HR, PHYSIO_RESTING_SV_ML);
    ASSERT_NEAR(co, 5.04f, 0.5f);
}

TEST(cardiac_output_exercise) {
    /* Exercise: HR=150, SV=100 -> CO = 15.0 L/min */
    float co = physiology_cardiac_output(150.0f, 100.0f);
    ASSERT_NEAR(co, 15.0f, 1.0f);
}

/* ---- Henderson-Hasselbalch: pH = 6.1 + log([HCO3-]/(0.03*pCO2)) -------- */

TEST(henderson_hasselbalch_normal) {
    /* Normal: HCO3=24 mEq/L, pCO2=40 mmHg */
    /* pH = 6.1 + log(24/(0.03*40)) = 6.1 + log(20) = 6.1 + 1.301 = 7.401 */
    float ph = physiology_henderson_hasselbalch(24.0f, 40.0f);
    ASSERT_NEAR(ph, 7.4f, 0.05f);
}

TEST(henderson_hasselbalch_acidosis) {
    /* Respiratory acidosis: pCO2=60 -> pH drops */
    float ph = physiology_henderson_hasselbalch(24.0f, 60.0f);
    ASSERT_LT(ph, 7.35f);
}

TEST(henderson_hasselbalch_alkalosis) {
    /* Metabolic alkalosis: HCO3=30 -> pH rises */
    float ph = physiology_henderson_hasselbalch(30.0f, 40.0f);
    ASSERT_GT(ph, 7.45f);
}

/* ---- GFR: Kf * (Pgc - Pbc - pi_gc) -------------------------------------- */

TEST(gfr_normal) {
    /* Normal: Kf=12.5, Pgc=55, Pbc=15, pi=30 -> GFR=12.5*(55-15-30)=125 */
    float gfr = physiology_gfr(PHYSIO_KF_NORMAL, PHYSIO_PGC_MMHG,
                                PHYSIO_PBC_MMHG, PHYSIO_PI_GC_MMHG);
    ASSERT_NEAR(gfr, 125.0f, 5.0f);
}

/* ---- MAP = CO * TPR ------------------------------------------------------ */

TEST(map_normal) {
    float map = physiology_map(PHYSIO_RESTING_CO_L_MIN, PHYSIO_TPR_NORMAL);
    ASSERT_NEAR(map, 93.0f, 5.0f);
}

/* ---- acid-base classification -------------------------------------------- */

TEST(acidbase_classification_normal) {
    physio_acidbase_status_t status = physiology_classify_acidbase(7.40f, 40.0f, 24.0f);
    ASSERT_EQ(status, PHYSIO_ACIDBASE_NORMAL);
}

TEST(acidbase_classification_resp_acidosis) {
    /* Low pH + high pCO2 */
    physio_acidbase_status_t status = physiology_classify_acidbase(7.25f, 60.0f, 24.0f);
    ASSERT_EQ(status, PHYSIO_ACIDBASE_RESP_ACIDOSIS);
}

/* ---- load healthy adult -------------------------------------------------- */

TEST(load_healthy_adult) {
    physiology_config_t cfg = physiology_default_config();
    physiology_sim_t* sim = physiology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    physiology_load_healthy_adult(sim);
    ASSERT_NEAR(sim->cardiovascular.heart_rate, PHYSIO_RESTING_HR, 10.0f);
    ASSERT_NEAR(sim->acidbase.blood_ph, 7.4f, 0.1f);

    physiology_destroy(sim);
}

/* ---- step basic ---------------------------------------------------------- */

TEST(step_basic) {
    physiology_config_t cfg = physiology_default_config();
    physiology_sim_t* sim = physiology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    physiology_load_healthy_adult(sim);
    int rc = physiology_step(sim, 1.0f);
    ASSERT_EQ(rc, 0);

    physiology_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(hill_equation_at_p50);
    RUN_TEST_SAFE(hill_equation_high_po2);
    RUN_TEST_SAFE(hill_equation_low_po2);
    RUN_TEST_SAFE(hill_sigmoidal_shape);
    RUN_TEST_SAFE(cardiac_output_resting);
    RUN_TEST_SAFE(cardiac_output_exercise);
    RUN_TEST_SAFE(henderson_hasselbalch_normal);
    RUN_TEST_SAFE(henderson_hasselbalch_acidosis);
    RUN_TEST_SAFE(henderson_hasselbalch_alkalosis);
    RUN_TEST_SAFE(gfr_normal);
    RUN_TEST_SAFE(map_normal);
    RUN_TEST_SAFE(acidbase_classification_normal);
    RUN_TEST_SAFE(acidbase_classification_resp_acidosis);
    RUN_TEST_SAFE(load_healthy_adult);
    RUN_TEST_SAFE(step_basic);
TEST_MAIN_END()
