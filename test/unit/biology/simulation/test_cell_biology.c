/**
 * @file test_cell_biology.c
 * @brief Tests for cell biology: cell cycle, membrane transport, signaling
 *
 * Validates cell cycle phase advancement, Fick's law diffusion,
 * facilitated transport saturation, osmotic pressure, and cell creation.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_cell_biology.h"
#include <math.h>

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    cell_biology_config_t cfg = cell_biology_default_config();
    cell_biology_sim_t* sim = cell_biology_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    cell_biology_destroy(sim);
}

/* ---- add cell ------------------------------------------------------------ */

TEST(add_cell) {
    cell_biology_config_t cfg = cell_biology_default_config();
    cell_biology_sim_t* sim = cell_biology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    cell_sim_cell_t cell = {0};
    cell.phase = CELL_PHASE_G1;
    cell.phase_progress = 0.0f;
    cell.atp_level = 1.0f;
    cell.dna_integrity = 1.0f;
    cell.chromosome_count = 46;
    cell.is_diploid = true;
    cell.alive = true;

    int rc = cell_biology_add_cell(sim, &cell);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(sim->num_cells, 1);

    cell_biology_destroy(sim);
}

/* ---- cell cycle phases advance ------------------------------------------- */

TEST(cell_cycle_phases_advance) {
    cell_biology_config_t cfg = cell_biology_default_config();
    cfg.dt = 1.0f;  /* 1 hour steps */
    cfg.growth_factor_conc = 1.0f;
    cfg.nutrient_level = 1.0f;
    cfg.oxygen_level = 1.0f;
    cell_biology_sim_t* sim = cell_biology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    cell_sim_cell_t cell = {0};
    cell.phase = CELL_PHASE_G1;
    cell.phase_progress = 0.0f;
    cell.atp_level = 1.0f;
    cell.dna_integrity = 1.0f;
    cell.chromosome_count = 46;
    cell.is_diploid = true;
    cell.alive = true;
    cell.regulators.cyclin_d = 0.8f;
    cell.regulators.cyclin_e = 0.8f;
    cell.regulators.cyclin_a = 0.8f;
    cell.regulators.cyclin_b = 0.9f;
    cell.regulators.rb = 0.5f;
    cell_biology_add_cell(sim, &cell);

    cell_phase_t initial_phase = sim->cells[0].phase;

    /* Advance through several hours -- phase should eventually change */
    for (int i = 0; i < 30; i++) {
        cell_biology_advance_cycle(sim, 0, cfg.dt);
    }

    /* After 30 hours of G1 (11h) + S (8h) + G2 (4h) = 23h, should
       have advanced past G1 at minimum */
    bool phase_changed = (sim->cells[0].phase != initial_phase) ||
                         (sim->cells[0].phase_progress > 0.5f);
    ASSERT_TRUE(phase_changed);

    cell_biology_destroy(sim);
}

/* ---- Fick's law: J = -D * dC/dx ----------------------------------------- */

TEST(fick_diffusion_positive_gradient) {
    /* D=1e-9 m^2/s, dC=10 mol/m^3, dx=1e-6 m (membrane thickness) */
    float J = cell_biology_fick_diffusion(1.0e-9f, 10.0f, 1.0e-6f);
    /* J = -1e-9 * 10 / 1e-6 = -0.01 mol/(m^2*s) */
    /* Magnitude check (sign depends on convention) */
    ASSERT_NEAR(fabsf(J), 0.01f, 0.001f);
}

TEST(fick_diffusion_proportional_to_gradient) {
    float J1 = cell_biology_fick_diffusion(1.0e-9f, 5.0f, 1.0e-6f);
    float J2 = cell_biology_fick_diffusion(1.0e-9f, 10.0f, 1.0e-6f);
    /* Double the gradient -> double the flux */
    ASSERT_NEAR(fabsf(J2), 2.0f * fabsf(J1), fabsf(J1) * 0.01f);
}

/* ---- facilitated transport saturates at Vmax ----------------------------- */

TEST(facilitated_transport_at_km) {
    /* At [S] = Km, rate = Vmax/2 */
    float v = cell_biology_facilitated_transport(10.0f, 5.0f, 5.0f);
    ASSERT_NEAR(v, 5.0f, 0.1f);
}

TEST(facilitated_transport_saturation) {
    float Vmax = 10.0f, Km = 5.0f;
    /* Very high substrate -> should approach Vmax */
    float v = cell_biology_facilitated_transport(Vmax, Km, 100000.0f);
    ASSERT_NEAR(v, Vmax, 0.1f);
}

TEST(facilitated_transport_low_substrate) {
    /* Very low [S] -> nearly linear: v ~ Vmax*[S]/Km */
    float v = cell_biology_facilitated_transport(10.0f, 5.0f, 0.01f);
    float expected = 10.0f * 0.01f / 5.01f;
    ASSERT_NEAR(v, expected, 0.01f);
}

/* ---- osmotic pressure: pi = i*M*R*T ------------------------------------ */

TEST(osmotic_pressure) {
    /* i=1, M=1.0 mol/L = 1000 mol/m^3, T=310.15K */
    /* pi = 1 * 1.0 * 8.314 * 310.15 = 2578 ... (units depend on impl) */
    float pi = cell_biology_osmotic_pressure(1.0f, 1.0f, 310.15f);
    ASSERT_GT(pi, 0.0f);
    /* Higher molarity -> higher osmotic pressure */
    float pi2 = cell_biology_osmotic_pressure(1.0f, 2.0f, 310.15f);
    ASSERT_GT(pi2, pi);
}

/* ---- load mammalian cell ------------------------------------------------- */

TEST(load_mammalian_cell) {
    cell_biology_config_t cfg = cell_biology_default_config();
    cell_biology_sim_t* sim = cell_biology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    cell_biology_load_mammalian_cell(sim);
    ASSERT_GE(sim->num_cells, 1);
    ASSERT_TRUE(sim->cells[0].alive);

    cell_biology_destroy(sim);
}

/* ---- step simulation ----------------------------------------------------- */

TEST(step_basic) {
    cell_biology_config_t cfg = cell_biology_default_config();
    cell_biology_sim_t* sim = cell_biology_create(&cfg);
    ASSERT_NOT_NULL(sim);

    cell_biology_load_mammalian_cell(sim);
    int rc = cell_biology_step(sim, 0.1f);
    ASSERT_EQ(rc, 0);

    cell_biology_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(add_cell);
    RUN_TEST_SAFE(cell_cycle_phases_advance);
    RUN_TEST_SAFE(fick_diffusion_positive_gradient);
    RUN_TEST_SAFE(fick_diffusion_proportional_to_gradient);
    RUN_TEST_SAFE(facilitated_transport_at_km);
    RUN_TEST_SAFE(facilitated_transport_saturation);
    RUN_TEST_SAFE(facilitated_transport_low_substrate);
    RUN_TEST_SAFE(osmotic_pressure);
    RUN_TEST_SAFE(load_mammalian_cell);
    RUN_TEST_SAFE(step_basic);
TEST_MAIN_END()
