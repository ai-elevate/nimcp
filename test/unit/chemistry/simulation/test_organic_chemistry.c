/**
 * @file test_organic_chemistry.c
 * @brief Tests for organic chemistry: unsaturation, aromaticity, mechanisms
 *
 * Validates degree of unsaturation, Huckel rule, SN2 stereochemistry,
 * molecule creation, reaction prediction.
 */

#include "../../../test_framework.h"
#include "cognitive/physics/nimcp_organic_chemistry.h"

/* ---- create / destroy ---------------------------------------------------- */

TEST(create_destroy) {
    ochem_config_t cfg = organic_chemistry_default_config();
    organic_chemistry_sim_t* sim = organic_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);
    ASSERT_TRUE(sim->initialized);
    organic_chemistry_destroy(sim);
}

/* ---- degree of unsaturation: DoU = (2C+2+N-H-X)/2 ----------------------- */

TEST(dou_benzene) {
    /* Benzene C6H6: DoU = (2*6+2-6)/2 = (14-6)/2 = 4 */
    uint32_t dou = organic_chemistry_degree_unsaturation(6, 6, 0, 0);
    ASSERT_EQ(dou, 4);
}

TEST(dou_ethane) {
    /* Ethane C2H6: DoU = (2*2+2-6)/2 = 0 (saturated) */
    uint32_t dou = organic_chemistry_degree_unsaturation(2, 6, 0, 0);
    ASSERT_EQ(dou, 0);
}

TEST(dou_ethylene) {
    /* Ethylene C2H4: DoU = (2*2+2-4)/2 = 1 (one double bond) */
    uint32_t dou = organic_chemistry_degree_unsaturation(2, 4, 0, 0);
    ASSERT_EQ(dou, 1);
}

TEST(dou_with_nitrogen) {
    /* Pyridine C5H5N: DoU = (2*5+2+1-5)/2 = 8/2 = 4 */
    uint32_t dou = organic_chemistry_degree_unsaturation(5, 5, 1, 0);
    ASSERT_EQ(dou, 4);
}

/* ---- Huckel rule: 4n+2 pi electrons = aromatic -------------------------- */

TEST(huckel_6_aromatic) {
    /* 6 pi electrons (benzene): 4(1)+2=6 -> aromatic */
    bool aromatic = organic_chemistry_is_huckel_aromatic(6);
    ASSERT_TRUE(aromatic);
}

TEST(huckel_4_antiaromatic) {
    /* 4 pi electrons (cyclobutadiene): not 4n+2 -> not aromatic */
    bool aromatic = organic_chemistry_is_huckel_aromatic(4);
    ASSERT_FALSE(aromatic);
}

TEST(huckel_10_aromatic) {
    /* 10 pi electrons (naphthalene): 4(2)+2=10 -> aromatic */
    bool aromatic = organic_chemistry_is_huckel_aromatic(10);
    ASSERT_TRUE(aromatic);
}

TEST(huckel_2_aromatic) {
    /* 2 pi electrons (cyclopropenyl cation): 4(0)+2=2 -> aromatic */
    bool aromatic = organic_chemistry_is_huckel_aromatic(2);
    ASSERT_TRUE(aromatic);
}

/* ---- SN2 stereochemistry: inversion ------------------------------------- */

TEST(sn2_predicts_inversion) {
    /* SN2 on R-substrate gives S-product (Walden inversion) */
    ochem_stereochemistry_t result = organic_chemistry_predict_stereo(
        OCHEM_RXN_SN2, OCHEM_STEREO_R);
    ASSERT_EQ(result, OCHEM_STEREO_S);
}

TEST(sn2_inversion_s_to_r) {
    /* SN2 on S-substrate gives R-product */
    ochem_stereochemistry_t result = organic_chemistry_predict_stereo(
        OCHEM_RXN_SN2, OCHEM_STEREO_S);
    ASSERT_EQ(result, OCHEM_STEREO_R);
}

/* ---- create molecule ----------------------------------------------------- */

TEST(create_molecule) {
    ochem_config_t cfg = organic_chemistry_default_config();
    organic_chemistry_sim_t* sim = organic_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    ochem_molecule_t mol = {0};
    snprintf(mol.name, OCHEM_MAX_NAME, "methane");
    snprintf(mol.smiles, OCHEM_MAX_NAME, "C");
    mol.molecular_weight = 16.04f;
    mol.degree_of_unsaturation = 0;
    mol.is_aromatic = false;
    mol.active = true;

    /* Add one carbon atom */
    mol.atoms[0].type = OCHEM_ATOM_C;
    mol.atoms[0].hybridization = 3; /* sp3 */
    mol.atoms[0].num_hydrogens = 4;
    mol.num_atoms = 1;

    uint32_t id = organic_chemistry_add_molecule(sim, &mol);
    ASSERT_EQ(sim->num_molecules, 1);
    (void)id;

    organic_chemistry_destroy(sim);
}

/* ---- step ---------------------------------------------------------------- */

TEST(step_basic) {
    ochem_config_t cfg = organic_chemistry_default_config();
    organic_chemistry_sim_t* sim = organic_chemistry_create(&cfg);
    ASSERT_NOT_NULL(sim);

    int rc = organic_chemistry_step(sim, 0.01f);
    ASSERT_EQ(rc, 0);

    organic_chemistry_destroy(sim);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(dou_benzene);
    RUN_TEST_SAFE(dou_ethane);
    RUN_TEST_SAFE(dou_ethylene);
    RUN_TEST_SAFE(dou_with_nitrogen);
    RUN_TEST_SAFE(huckel_6_aromatic);
    RUN_TEST_SAFE(huckel_4_antiaromatic);
    RUN_TEST_SAFE(huckel_10_aromatic);
    RUN_TEST_SAFE(huckel_2_aromatic);
    RUN_TEST_SAFE(sn2_predicts_inversion);
    RUN_TEST_SAFE(sn2_inversion_s_to_r);
    RUN_TEST_SAFE(create_molecule);
    RUN_TEST_SAFE(step_basic);
TEST_MAIN_END()
