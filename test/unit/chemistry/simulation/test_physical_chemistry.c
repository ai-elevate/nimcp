/**
 * @file test_physical_chemistry.c
 * @brief Tests for physical chemistry: thermodynamics, stat mech, spectroscopy
 *
 * Validates Gibbs free energy, equilibrium constant, ideal gas law,
 * Beer-Lambert law, Wien displacement, Boltzmann distribution, and more.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_physical_chemistry.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    pchem_config_t cfg = physical_chemistry_default_config();
    physical_chemistry_sim_t* sim = physical_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    physical_chemistry_destroy(sim);
}

/* ---- Gibbs: dG = dH - T*dS ---------------------------------------------- */

TEST(gibbs_free_energy) {
    /* dH = -100 kJ/mol, T = 298 K, dS = 0.05 kJ/(mol*K) */
    /* dG = -100 - 298*0.05 = -100 - 14.9 = -114.9 kJ/mol */
    float dG = pchem_gibbs_free_energy(-100.0f, 298.0f, 0.05f);
    ASSERT_NEAR(dG, -114.9f, 0.5f);
}

TEST(gibbs_spontaneity) {
    /* Exothermic + positive entropy change = always spontaneous */
    float dG = pchem_gibbs_free_energy(-50.0f, 298.0f, 0.1f);
    ASSERT_LT(dG, 0.0f);
}

/* ---- Equilibrium K = exp(-dG/(RT)) -------------------------------------- */

TEST(equilibrium_constant_known) {
    /* dG = -5705.8 J/mol at 298K -> K = exp(5705.8/(8.314*298)) = exp(2.303) ~ 10 */
    /* Using kJ: dG = -5.7058 kJ/mol */
    float K = pchem_equilibrium_constant(-5.7058f, 298.0f);
    ASSERT_NEAR(K, 10.0f, 1.0f);
}

TEST(equilibrium_constant_positive_dG) {
    /* Positive dG -> K < 1 (products not favored) */
    float K = pchem_equilibrium_constant(10.0f, 298.0f);
    ASSERT_LT(K, 1.0f);
    ASSERT_GT(K, 0.0f);
}

/* ---- Ideal gas: PV = nRT, P = nRT/V ------------------------------------- */

TEST(ideal_gas_law) {
    /* n=1 mol, T=273.15 K, V=22.414 L -> P = 1 atm = 101325 Pa */
    /* But this uses R = 8.314 J/(mol*K), so P in Pa */
    /* P = 1 * 8.314 * 273.15 / 0.022414 = 101325 Pa */
    float P = pchem_ideal_gas_pressure(1.0f, 273.15f, 0.022414f);
    ASSERT_NEAR(P, 101325.0f, 500.0f);
}

TEST(ideal_gas_proportionality) {
    /* Double temperature -> double pressure (at constant n, V) */
    float P1 = pchem_ideal_gas_pressure(1.0f, 300.0f, 1.0f);
    float P2 = pchem_ideal_gas_pressure(1.0f, 600.0f, 1.0f);
    ASSERT_NEAR(P2, 2.0f * P1, P1 * 0.01f);
}

/* ---- Beer-Lambert: A = epsilon * l * c ---------------------------------- */

TEST(beer_lambert_known) {
    /* epsilon=100 L/(mol*cm), l=1 cm, c=0.01 mol/L -> A = 1.0 */
    float A = pchem_beer_lambert(100.0f, 1.0f, 0.01f);
    ASSERT_NEAR(A, 1.0f, 0.001f);
}

TEST(beer_lambert_proportional) {
    /* Double concentration -> double absorbance */
    float A1 = pchem_beer_lambert(50.0f, 1.0f, 0.01f);
    float A2 = pchem_beer_lambert(50.0f, 1.0f, 0.02f);
    ASSERT_NEAR(A2, 2.0f * A1, 0.001f);
}

/* ---- Wien displacement: lambda_max = b/T -------------------------------- */

TEST(wien_sun_temperature) {
    /* T = 5778 K -> lambda_max = 2.898e-3/5778 = 5.015e-7 m = 501.5 nm */
    float lambda = pchem_wien_displacement(5778.0f);
    ASSERT_NEAR(lambda, 5.015e-7f, 5.0e-9f);  /* ~502 nm +/- 5 nm */
}

TEST(wien_room_temperature) {
    /* T = 300 K -> lambda_max = 2.898e-3/300 = 9.66e-6 m (infrared) */
    float lambda = pchem_wien_displacement(300.0f);
    ASSERT_NEAR(lambda, 9.66e-6f, 1.0e-7f);
}

/* ---- Boltzmann probability: P(E) = exp(-E/kT) / Z ----------------------- */

TEST(boltzmann_ground_state) {
    /* Ground state E=0: P = 1/Z */
    float Z = 5.0f;
    float P = pchem_boltzmann_probability(0.0f, 300.0f, Z);
    ASSERT_NEAR(P, 1.0f / Z, 0.01f);
}

TEST(boltzmann_decreasing) {
    /* Higher energy -> lower probability */
    float P_low = pchem_boltzmann_probability(1.0e-21f, 300.0f, 1.0f);
    float P_high = pchem_boltzmann_probability(1.0e-20f, 300.0f, 1.0f);
    ASSERT_GT(P_low, P_high);
}

/* ---- add species and step ------------------------------------------------ */

TEST(add_species_and_step) {
    pchem_config_t cfg = physical_chemistry_default_config();
    physical_chemistry_sim_t* sim = physical_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    pchem_species_t sp = {0};
    snprintf(sp.name, PCHEM_MAX_NAME, "water");
    sp.enthalpy = -285.8f;
    sp.entropy = 69.91f;
    sp.molar_mass = 18.015f;
    sp.active = true;
    physical_chemistry_add_species(sim, &sp);
    ASSERT_EQ(sim->num_species, 1);

    int rc = physical_chemistry_step(sim, 0.01f);
    ASSERT_EQ(rc, 0);

    physical_chemistry_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(gibbs_free_energy);
    RUN_TEST_SAFE(gibbs_spontaneity);
    RUN_TEST_SAFE(equilibrium_constant_known);
    RUN_TEST_SAFE(equilibrium_constant_positive_dG);
    RUN_TEST_SAFE(ideal_gas_law);
    RUN_TEST_SAFE(ideal_gas_proportionality);
    RUN_TEST_SAFE(beer_lambert_known);
    RUN_TEST_SAFE(beer_lambert_proportional);
    RUN_TEST_SAFE(wien_sun_temperature);
    RUN_TEST_SAFE(wien_room_temperature);
    RUN_TEST_SAFE(boltzmann_ground_state);
    RUN_TEST_SAFE(boltzmann_decreasing);
    RUN_TEST_SAFE(add_species_and_step);
TEST_MAIN_END()
