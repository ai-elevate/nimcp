/**
 * @file nimcp_chemistry.c
 * @brief Chemistry reasoning implementation for parietal lobe
 *
 * Implements chemical reasoning capabilities including molecular parsing,
 * equation balancing, stoichiometry, and basic thermodynamics.
 */

#include "cognitive/parietal/nimcp_chemistry.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/thread/nimcp_thread.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define EPSILON 1e-6f

/* ============================================================================
 * PERIODIC TABLE DATA
 * ============================================================================ */

/**
 * @brief Periodic table element data (first 36 elements for now)
 */
static const element_properties_t PERIODIC_TABLE[] = {
    {0, "", "", 0.0f, 0, 0, ELEMENT_CATEGORY_NONMETAL, 0.0f, {0}, 0, 0.0f, 0.0f, 0.0f},
    {1, "H", "Hydrogen", 1.008f, 1, 1, ELEMENT_CATEGORY_NONMETAL, 2.20f, {-1, 1}, 2, 72.8f, 1312.0f, 53.0f},
    {2, "He", "Helium", 4.003f, 1, 18, ELEMENT_CATEGORY_NOBLE_GAS, 0.0f, {0}, 1, 0.0f, 2372.3f, 31.0f},
    {3, "Li", "Lithium", 6.94f, 2, 1, ELEMENT_CATEGORY_ALKALI_METAL, 0.98f, {1}, 1, 59.6f, 520.2f, 167.0f},
    {4, "Be", "Beryllium", 9.012f, 2, 2, ELEMENT_CATEGORY_ALKALINE_EARTH, 1.57f, {2}, 1, 0.0f, 899.5f, 112.0f},
    {5, "B", "Boron", 10.81f, 2, 13, ELEMENT_CATEGORY_METALLOID, 2.04f, {3}, 1, 26.7f, 800.6f, 87.0f},
    {6, "C", "Carbon", 12.011f, 2, 14, ELEMENT_CATEGORY_NONMETAL, 2.55f, {-4, 4}, 2, 121.8f, 1086.5f, 77.0f},
    {7, "N", "Nitrogen", 14.007f, 2, 15, ELEMENT_CATEGORY_NONMETAL, 3.04f, {-3, 3, 5}, 3, 0.0f, 1402.3f, 75.0f},
    {8, "O", "Oxygen", 15.999f, 2, 16, ELEMENT_CATEGORY_NONMETAL, 3.44f, {-2}, 1, 141.0f, 1313.9f, 73.0f},
    {9, "F", "Fluorine", 18.998f, 2, 17, ELEMENT_CATEGORY_HALOGEN, 3.98f, {-1}, 1, 328.0f, 1681.0f, 71.0f},
    {10, "Ne", "Neon", 20.180f, 2, 18, ELEMENT_CATEGORY_NOBLE_GAS, 0.0f, {0}, 1, 0.0f, 2080.7f, 38.0f},
    {11, "Na", "Sodium", 22.990f, 3, 1, ELEMENT_CATEGORY_ALKALI_METAL, 0.93f, {1}, 1, 52.8f, 495.8f, 190.0f},
    {12, "Mg", "Magnesium", 24.305f, 3, 2, ELEMENT_CATEGORY_ALKALINE_EARTH, 1.31f, {2}, 1, 0.0f, 737.7f, 145.0f},
    {13, "Al", "Aluminum", 26.982f, 3, 13, ELEMENT_CATEGORY_POST_TRANSITION, 1.61f, {3}, 1, 42.5f, 577.5f, 118.0f},
    {14, "Si", "Silicon", 28.086f, 3, 14, ELEMENT_CATEGORY_METALLOID, 1.90f, {-4, 4}, 2, 133.6f, 786.5f, 111.0f},
    {15, "P", "Phosphorus", 30.974f, 3, 15, ELEMENT_CATEGORY_NONMETAL, 2.19f, {-3, 3, 5}, 3, 72.0f, 1011.8f, 98.0f},
    {16, "S", "Sulfur", 32.06f, 3, 16, ELEMENT_CATEGORY_NONMETAL, 2.58f, {-2, 2, 4, 6}, 4, 200.4f, 999.6f, 88.0f},
    {17, "Cl", "Chlorine", 35.45f, 3, 17, ELEMENT_CATEGORY_HALOGEN, 3.16f, {-1, 1, 3, 5, 7}, 5, 349.0f, 1251.2f, 79.0f},
    {18, "Ar", "Argon", 39.948f, 3, 18, ELEMENT_CATEGORY_NOBLE_GAS, 0.0f, {0}, 1, 0.0f, 1520.6f, 71.0f},
    {19, "K", "Potassium", 39.098f, 4, 1, ELEMENT_CATEGORY_ALKALI_METAL, 0.82f, {1}, 1, 48.4f, 418.8f, 243.0f},
    {20, "Ca", "Calcium", 40.078f, 4, 2, ELEMENT_CATEGORY_ALKALINE_EARTH, 1.00f, {2}, 1, 2.4f, 589.8f, 194.0f},
    {21, "Sc", "Scandium", 44.956f, 4, 3, ELEMENT_CATEGORY_TRANSITION, 1.36f, {3}, 1, 18.1f, 633.1f, 184.0f},
    {22, "Ti", "Titanium", 47.867f, 4, 4, ELEMENT_CATEGORY_TRANSITION, 1.54f, {2, 3, 4}, 3, 7.6f, 658.8f, 176.0f},
    {23, "V", "Vanadium", 50.942f, 4, 5, ELEMENT_CATEGORY_TRANSITION, 1.63f, {2, 3, 4, 5}, 4, 50.6f, 650.9f, 171.0f},
    {24, "Cr", "Chromium", 51.996f, 4, 6, ELEMENT_CATEGORY_TRANSITION, 1.66f, {2, 3, 6}, 3, 64.3f, 652.9f, 166.0f},
    {25, "Mn", "Manganese", 54.938f, 4, 7, ELEMENT_CATEGORY_TRANSITION, 1.55f, {2, 3, 4, 6, 7}, 5, 0.0f, 717.3f, 161.0f},
    {26, "Fe", "Iron", 55.845f, 4, 8, ELEMENT_CATEGORY_TRANSITION, 1.83f, {2, 3}, 2, 15.7f, 762.5f, 156.0f},
    {27, "Co", "Cobalt", 58.933f, 4, 9, ELEMENT_CATEGORY_TRANSITION, 1.88f, {2, 3}, 2, 63.7f, 760.4f, 152.0f},
    {28, "Ni", "Nickel", 58.693f, 4, 10, ELEMENT_CATEGORY_TRANSITION, 1.91f, {2, 3}, 2, 112.0f, 737.1f, 149.0f},
    {29, "Cu", "Copper", 63.546f, 4, 11, ELEMENT_CATEGORY_TRANSITION, 1.90f, {1, 2}, 2, 118.4f, 745.5f, 145.0f},
    {30, "Zn", "Zinc", 65.38f, 4, 12, ELEMENT_CATEGORY_TRANSITION, 1.65f, {2}, 1, 0.0f, 906.4f, 142.0f},
    {31, "Ga", "Gallium", 69.723f, 4, 13, ELEMENT_CATEGORY_POST_TRANSITION, 1.81f, {3}, 1, 28.9f, 578.8f, 136.0f},
    {32, "Ge", "Germanium", 72.63f, 4, 14, ELEMENT_CATEGORY_METALLOID, 2.01f, {-4, 2, 4}, 3, 119.0f, 762.2f, 125.0f},
    {33, "As", "Arsenic", 74.922f, 4, 15, ELEMENT_CATEGORY_METALLOID, 2.18f, {-3, 3, 5}, 3, 78.2f, 947.0f, 114.0f},
    {34, "Se", "Selenium", 78.971f, 4, 16, ELEMENT_CATEGORY_NONMETAL, 2.55f, {-2, 2, 4, 6}, 4, 195.0f, 941.0f, 103.0f},
    {35, "Br", "Bromine", 79.904f, 4, 17, ELEMENT_CATEGORY_HALOGEN, 2.96f, {-1, 1, 3, 5}, 4, 324.6f, 1139.9f, 94.0f},
    {36, "Kr", "Krypton", 83.798f, 4, 18, ELEMENT_CATEGORY_NOBLE_GAS, 3.00f, {0, 2}, 2, 0.0f, 1350.8f, 88.0f}
};

#define NUM_KNOWN_ELEMENTS (sizeof(PERIODIC_TABLE) / sizeof(PERIODIC_TABLE[0]))

/* ============================================================================
 * INTERNAL STRUCTURES
 * ============================================================================ */

struct chemistry {
    chemistry_config_t config;

    /* Modulation state */
    float inflammation_level;
    float sleep_deprivation_level;

    /* Statistics */
    uint64_t molecules_parsed;
    uint64_t reactions_balanced;
    uint64_t stoichiometry_calcs;
    uint64_t property_lookups;
    double total_processing_time_us;

    /* Thread safety */
    nimcp_mutex_t* lock;
};

/* Thread-local error message */
static _Thread_local char g_chemistry_error[256] = {0};

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void set_chemistry_error(const char* msg) {
    strncpy(g_chemistry_error, msg, sizeof(g_chemistry_error) - 1);
    g_chemistry_error[sizeof(g_chemistry_error) - 1] = '\0';
}

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

/**
 * @brief Find element by symbol in periodic table
 */
static int find_element_by_symbol(const char* symbol) {
    for (size_t i = 1; i < NUM_KNOWN_ELEMENTS; i++) {
        if (strcmp(PERIODIC_TABLE[i].symbol, symbol) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Parse element count from formula (e.g., "2" from "H2")
 */
static uint32_t parse_count(const char** ptr) {
    uint32_t count = 0;
    while (isdigit(**ptr)) {
        count = count * 10 + (**ptr - '0');
        (*ptr)++;
    }
    return count > 0 ? count : 1;
}

/**
 * @brief Parse element symbol from formula
 */
static int parse_element_symbol(const char** ptr, char* symbol, size_t max_len) {
    if (!isupper(**ptr)) return -1;

    size_t len = 0;
    symbol[len++] = **ptr;
    (*ptr)++;

    while (islower(**ptr) && len < max_len - 1) {
        symbol[len++] = **ptr;
        (*ptr)++;
    }
    symbol[len] = '\0';

    return 0;
}

/* ============================================================================
 * LIFECYCLE API
 * ============================================================================ */

chemistry_config_t chemistry_default_config(void) {
    chemistry_config_t config = {
        .enable_3d_coordinates = false,
        .enable_thermodynamics = true,
        .enable_bond_energies = true,
        .enable_bio_async = false,
        .temperature_k = 298.15f,
        .pressure_atm = 1.0f,
        .inflammation_sensitivity = 0.2f,
        .sleep_deprivation_factor = 0.15f
    };
    return config;
}

bool chemistry_validate_config(const chemistry_config_t* config) {
    if (!config) {
        set_chemistry_error("Null config");
        return false;
    }
    if (config->temperature_k <= 0.0f) {
        set_chemistry_error("Invalid temperature");
        return false;
    }
    if (config->pressure_atm <= 0.0f) {
        set_chemistry_error("Invalid pressure");
        return false;
    }
    return true;
}

chemistry_t* chemistry_create(void) {
    return chemistry_create_custom(NULL);
}

chemistry_t* chemistry_create_custom(const chemistry_config_t* config) {
    chemistry_config_t cfg = config ? *config : chemistry_default_config();

    if (!chemistry_validate_config(&cfg)) {
        return NULL;
    }

    chemistry_t* chem = calloc(1, sizeof(chemistry_t));
    if (!chem) {
        set_chemistry_error("Failed to allocate chemistry struct");
        return NULL;
    }

    chem->config = cfg;
    chem->inflammation_level = 0.0f;
    chem->sleep_deprivation_level = 0.0f;

    mutex_attr_t attr = {.type = MUTEX_TYPE_NORMAL};
    chem->lock = nimcp_mutex_create(&attr);
    if (!chem->lock) {
        set_chemistry_error("Failed to create mutex");
        free(chem);
        return NULL;
    }

    return chem;
}

void chemistry_destroy(chemistry_t* chem) {
    if (!chem) return;

    if (chem->lock) {
        nimcp_mutex_free(chem->lock);
    }
    free(chem);
}

/* ============================================================================
 * PERIODIC TABLE API
 * ============================================================================ */

int chemistry_get_element(
    const chemistry_t* chem,
    uint8_t atomic_number,
    element_properties_t* props
) {
    if (!chem || !props) return -1;
    if (atomic_number == 0 || atomic_number >= NUM_KNOWN_ELEMENTS) {
        set_chemistry_error("Invalid atomic number");
        return -1;
    }

    *props = PERIODIC_TABLE[atomic_number];
    ((chemistry_t*)chem)->property_lookups++;
    return 0;
}

int chemistry_get_element_by_symbol(
    const chemistry_t* chem,
    const char* symbol,
    element_properties_t* props
) {
    if (!chem || !symbol || !props) return -1;

    int idx = find_element_by_symbol(symbol);
    if (idx < 0) {
        set_chemistry_error("Unknown element symbol");
        return -1;
    }

    *props = PERIODIC_TABLE[idx];
    ((chemistry_t*)chem)->property_lookups++;
    return 0;
}

bool chemistry_can_form_ionic_bond(
    const chemistry_t* chem,
    uint8_t element1,
    uint8_t element2
) {
    if (!chem) return false;
    if (element1 == 0 || element1 >= NUM_KNOWN_ELEMENTS) return false;
    if (element2 == 0 || element2 >= NUM_KNOWN_ELEMENTS) return false;

    float en1 = PERIODIC_TABLE[element1].electronegativity;
    float en2 = PERIODIC_TABLE[element2].electronegativity;

    /* Ionic bond typically forms when electronegativity difference > 1.7 */
    return fabsf(en1 - en2) > 1.7f;
}

bond_type_t chemistry_predict_bond_type(
    const chemistry_t* chem,
    uint8_t element1,
    uint8_t element2
) {
    if (!chem) return BOND_SINGLE;
    if (element1 == 0 || element1 >= NUM_KNOWN_ELEMENTS) return BOND_SINGLE;
    if (element2 == 0 || element2 >= NUM_KNOWN_ELEMENTS) return BOND_SINGLE;

    float en1 = PERIODIC_TABLE[element1].electronegativity;
    float en2 = PERIODIC_TABLE[element2].electronegativity;
    float diff = fabsf(en1 - en2);

    /* Metallic bond between metals */
    element_category_t cat1 = PERIODIC_TABLE[element1].category;
    element_category_t cat2 = PERIODIC_TABLE[element2].category;
    bool metal1 = (cat1 == ELEMENT_CATEGORY_ALKALI_METAL ||
                   cat1 == ELEMENT_CATEGORY_ALKALINE_EARTH ||
                   cat1 == ELEMENT_CATEGORY_TRANSITION ||
                   cat1 == ELEMENT_CATEGORY_POST_TRANSITION);
    bool metal2 = (cat2 == ELEMENT_CATEGORY_ALKALI_METAL ||
                   cat2 == ELEMENT_CATEGORY_ALKALINE_EARTH ||
                   cat2 == ELEMENT_CATEGORY_TRANSITION ||
                   cat2 == ELEMENT_CATEGORY_POST_TRANSITION);

    if (metal1 && metal2) {
        return BOND_METALLIC;
    }

    if (diff > 1.7f) {
        return BOND_IONIC;
    }

    return BOND_SINGLE;  /* Default to covalent */
}

/* ============================================================================
 * MOLECULE API
 * ============================================================================ */

/**
 * @brief Internal molecule parsing (no locking)
 */
static int parse_molecule_unlocked(
    chemistry_t* chem,
    const char* formula,
    molecule_t* molecule
) {
    memset(molecule, 0, sizeof(molecule_t));
    strncpy(molecule->formula, formula, CHEMISTRY_MAX_FORMULA - 1);

    const char* ptr = formula;
    uint32_t atom_idx = 0;
    float total_mass = 0.0f;

    while (*ptr && atom_idx < CHEMISTRY_MAX_ATOMS) {
        /* Skip whitespace */
        while (isspace(*ptr)) ptr++;
        if (!*ptr) break;

        /* Parse element symbol */
        char symbol[CHEMISTRY_ELEMENT_SYMBOL_LEN];
        if (parse_element_symbol(&ptr, symbol, sizeof(symbol)) != 0) {
            set_chemistry_error("Invalid element symbol");
            return -1;
        }

        /* Find element in periodic table */
        int elem_idx = find_element_by_symbol(symbol);
        if (elem_idx < 0) {
            set_chemistry_error("Unknown element");
            return -1;
        }

        /* Parse count */
        uint32_t count = parse_count(&ptr);

        /* Add atoms */
        for (uint32_t i = 0; i < count && atom_idx < CHEMISTRY_MAX_ATOMS; i++) {
            molecule->atoms[atom_idx].element = (uint8_t)elem_idx;
            molecule->atoms[atom_idx].formal_charge = 0;
            total_mass += PERIODIC_TABLE[elem_idx].atomic_mass;
            atom_idx++;
        }
    }

    molecule->num_atoms = atom_idx;
    molecule->molar_mass = total_mass;
    molecule->net_charge = 0;
    molecule->geometry = GEOMETRY_LINEAR;

    chem->molecules_parsed++;
    return 0;
}

int chemistry_parse_molecule(
    chemistry_t* chem,
    const char* formula,
    molecule_t* molecule
) {
    if (!chem || !formula || !molecule) {
        set_chemistry_error("Null parameter");
        return -1;
    }

    nimcp_mutex_lock(chem->lock);
    int result = parse_molecule_unlocked(chem, formula, molecule);
    nimcp_mutex_unlock(chem->lock);

    return result;
}

float chemistry_molar_mass(
    const chemistry_t* chem,
    const molecule_t* molecule
) {
    if (!chem || !molecule) return 0.0f;
    return molecule->molar_mass;
}

molecular_geometry_t chemistry_predict_geometry(
    chemistry_t* chem,
    const molecule_t* molecule,
    uint32_t central_atom_idx
) {
    if (!chem || !molecule) return GEOMETRY_LINEAR;
    if (central_atom_idx >= molecule->num_atoms) return GEOMETRY_LINEAR;

    /* Count bonds to central atom */
    uint32_t bond_count = 0;
    for (uint32_t i = 0; i < molecule->num_bonds; i++) {
        if (molecule->bonds[i].atom1_idx == central_atom_idx ||
            molecule->bonds[i].atom2_idx == central_atom_idx) {
            bond_count++;
        }
    }

    /* VSEPR prediction based on steric number */
    switch (bond_count) {
        case 0:
        case 1:
            return GEOMETRY_LINEAR;
        case 2:
            return GEOMETRY_LINEAR;  /* Could be bent with lone pairs */
        case 3:
            return GEOMETRY_TRIGONAL_PLANAR;
        case 4:
            return GEOMETRY_TETRAHEDRAL;
        case 5:
            return GEOMETRY_TRIGONAL_BIPYRAMIDAL;
        case 6:
            return GEOMETRY_OCTAHEDRAL;
        default:
            return GEOMETRY_OCTAHEDRAL;
    }
}

uint32_t chemistry_count_element(
    const molecule_t* molecule,
    uint8_t atomic_number
) {
    if (!molecule) return 0;

    uint32_t count = 0;
    for (uint32_t i = 0; i < molecule->num_atoms; i++) {
        if (molecule->atoms[i].element == atomic_number) {
            count++;
        }
    }
    return count;
}

int chemistry_molecule_to_string(
    const molecule_t* molecule,
    char* buffer,
    size_t buffer_size
) {
    if (!molecule || !buffer || buffer_size == 0) return -1;

    strncpy(buffer, molecule->formula, buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

/* ============================================================================
 * REACTION API
 * ============================================================================ */

/**
 * @brief Parse a single species from string
 * @return Pointer to next char after parsing, or NULL on error
 */
static const char* parse_species(
    chemistry_t* chem,
    const char* str,
    species_t* species
) {
    if (!str || !species) return NULL;

    /* Skip leading whitespace */
    while (isspace(*str)) str++;
    if (!*str) return NULL;

    /* Parse coefficient if present */
    uint32_t coeff = 1;
    if (isdigit(*str)) {
        coeff = 0;
        while (isdigit(*str)) {
            coeff = coeff * 10 + (*str - '0');
            str++;
        }
        while (isspace(*str)) str++;
    }
    species->coefficient = coeff;

    /* Find end of formula (next +, -, or end of string) */
    const char* end = str;
    while (*end && *end != '+' && *end != '-') end++;

    /* Copy formula and trim trailing whitespace */
    char formula[CHEMISTRY_MAX_FORMULA];
    size_t len = (size_t)(end - str);
    if (len >= CHEMISTRY_MAX_FORMULA) len = CHEMISTRY_MAX_FORMULA - 1;
    strncpy(formula, str, len);
    formula[len] = '\0';

    /* Trim trailing whitespace */
    char* trim_end = formula + len - 1;
    while (trim_end >= formula && isspace(*trim_end)) {
        *trim_end-- = '\0';
    }

    /* Parse the molecule (unlocked - caller holds lock) */
    parse_molecule_unlocked(chem, formula, &species->molecule);

    /* Return pointer past the '+' separator (but not past '-' which may be arrow) */
    if (*end == '+') end++;
    return end;
}

int chemistry_parse_reaction(
    chemistry_t* chem,
    const char* equation,
    reaction_t* reaction
) {
    if (!chem || !equation || !reaction) {
        set_chemistry_error("Null parameter");
        return -1;
    }

    nimcp_mutex_lock(chem->lock);

    memset(reaction, 0, sizeof(reaction_t));

    /* Find arrow separator */
    const char* arrow = strstr(equation, "->");
    if (!arrow) {
        arrow = strstr(equation, "→");
        if (!arrow) {
            set_chemistry_error("No arrow in reaction equation");
            nimcp_mutex_unlock(chem->lock);
            return -1;
        }
    }

    /* Parse reactants (before arrow) */
    const char* ptr = equation;
    while (ptr < arrow && reaction->num_reactants < CHEMISTRY_MAX_SPECIES) {
        ptr = parse_species(chem, ptr,
            &reaction->reactants[reaction->num_reactants]);
        if (!ptr) break;
        if (ptr <= arrow) {
            reaction->num_reactants++;
        }
    }

    /* Parse products (after arrow) */
    const char* products_start = arrow + 2;  /* Skip "->" */
    while (isspace(*products_start)) products_start++;

    ptr = products_start;
    while (*ptr && reaction->num_products < CHEMISTRY_MAX_SPECIES) {
        const char* next = parse_species(chem, ptr,
            &reaction->products[reaction->num_products]);
        if (!next || next == ptr) break;
        reaction->num_products++;
        ptr = next;
    }

    reaction->type = REACTION_UNKNOWN;
    reaction->is_balanced = chemistry_is_balanced(reaction);

    nimcp_mutex_unlock(chem->lock);
    return 0;
}

bool chemistry_is_balanced(const reaction_t* reaction) {
    if (!reaction) return false;

    /* Count atoms on each side */
    uint32_t reactant_counts[CHEMISTRY_NUM_ELEMENTS + 1] = {0};
    uint32_t product_counts[CHEMISTRY_NUM_ELEMENTS + 1] = {0};

    for (uint32_t i = 0; i < reaction->num_reactants; i++) {
        const species_t* sp = &reaction->reactants[i];
        for (uint32_t j = 0; j < sp->molecule.num_atoms; j++) {
            uint8_t elem = sp->molecule.atoms[j].element;
            reactant_counts[elem] += sp->coefficient;
        }
    }

    for (uint32_t i = 0; i < reaction->num_products; i++) {
        const species_t* sp = &reaction->products[i];
        for (uint32_t j = 0; j < sp->molecule.num_atoms; j++) {
            uint8_t elem = sp->molecule.atoms[j].element;
            product_counts[elem] += sp->coefficient;
        }
    }

    for (uint32_t i = 1; i <= CHEMISTRY_NUM_ELEMENTS; i++) {
        if (reactant_counts[i] != product_counts[i]) {
            return false;
        }
    }

    return true;
}

bool chemistry_balance_equation(
    chemistry_t* chem,
    reaction_t* reaction
) {
    if (!chem || !reaction) return false;

    nimcp_mutex_lock(chem->lock);

    /* Simple balancing for common cases */
    /* This is a simplified algorithm - full implementation would use
     * linear algebra to solve the system of equations */

    /* Try coefficients 1-10 for each species */
    uint32_t max_tries = 10;
    uint32_t total_species = reaction->num_reactants + reaction->num_products;

    if (total_species == 0 || total_species > 6) {
        nimcp_mutex_unlock(chem->lock);
        return false;
    }

    /* Brute force for small reactions */
    for (uint32_t c1 = 1; c1 <= max_tries; c1++) {
        for (uint32_t c2 = 1; c2 <= max_tries; c2++) {
            for (uint32_t c3 = 1; c3 <= max_tries; c3++) {
                for (uint32_t c4 = 1; c4 <= max_tries; c4++) {
                    /* Set coefficients */
                    if (reaction->num_reactants > 0)
                        reaction->reactants[0].coefficient = c1;
                    if (reaction->num_reactants > 1)
                        reaction->reactants[1].coefficient = c2;
                    if (reaction->num_products > 0)
                        reaction->products[0].coefficient = c3;
                    if (reaction->num_products > 1)
                        reaction->products[1].coefficient = c4;

                    if (chemistry_is_balanced(reaction)) {
                        reaction->is_balanced = true;
                        chem->reactions_balanced++;
                        nimcp_mutex_unlock(chem->lock);
                        return true;
                    }
                }
            }
        }
    }

    nimcp_mutex_unlock(chem->lock);
    return false;
}

reaction_type_t chemistry_classify_reaction(
    chemistry_t* chem,
    const reaction_t* reaction
) {
    if (!chem || !reaction) return REACTION_UNKNOWN;

    /* Simple classification heuristics */
    if (reaction->num_reactants == 2 && reaction->num_products == 1) {
        return REACTION_SYNTHESIS;
    }
    if (reaction->num_reactants == 1 && reaction->num_products == 2) {
        return REACTION_DECOMPOSITION;
    }
    if (reaction->num_reactants == 2 && reaction->num_products == 2) {
        /* Check for O2 reactant (combustion) */
        for (uint32_t i = 0; i < reaction->num_reactants; i++) {
            if (strcmp(reaction->reactants[i].molecule.formula, "O2") == 0) {
                /* Check for CO2 and H2O products */
                bool has_co2 = false, has_h2o = false;
                for (uint32_t j = 0; j < reaction->num_products; j++) {
                    if (strcmp(reaction->products[j].molecule.formula, "CO2") == 0)
                        has_co2 = true;
                    if (strcmp(reaction->products[j].molecule.formula, "H2O") == 0)
                        has_h2o = true;
                }
                if (has_co2 || has_h2o) {
                    return REACTION_COMBUSTION;
                }
            }
        }
        return REACTION_DOUBLE_REPLACEMENT;
    }

    return REACTION_UNKNOWN;
}

int chemistry_calculate_thermodynamics(
    chemistry_t* chem,
    reaction_t* reaction
) {
    if (!chem || !reaction) return -1;

    /* Placeholder - would need enthalpy of formation data */
    reaction->enthalpy_kj = 0.0f;
    reaction->entropy_j_k = 0.0f;
    reaction->gibbs_free_energy_kj = 0.0f;

    return 0;
}

bool chemistry_is_spontaneous(
    const reaction_t* reaction,
    float temperature_k
) {
    if (!reaction) return false;

    /* G = H - TS */
    float gibbs = reaction->enthalpy_kj * 1000.0f -
                  temperature_k * reaction->entropy_j_k;

    return gibbs < 0.0f;
}

/* ============================================================================
 * STOICHIOMETRY API
 * ============================================================================ */

int chemistry_calculate_stoichiometry(
    chemistry_t* chem,
    const reaction_t* reaction,
    float limiting_reagent_moles,
    uint32_t limiting_reagent_idx,
    stoichiometry_result_t* result
) {
    if (!chem || !reaction || !result) return -1;
    if (limiting_reagent_idx >= reaction->num_reactants) return -1;
    if (!reaction->is_balanced) {
        set_chemistry_error("Reaction not balanced");
        return -1;
    }

    nimcp_mutex_lock(chem->lock);

    memset(result, 0, sizeof(stoichiometry_result_t));
    result->limiting_reagent_idx = limiting_reagent_idx;

    uint32_t limit_coeff = reaction->reactants[limiting_reagent_idx].coefficient;
    float moles_per_coeff = limiting_reagent_moles / (float)limit_coeff;

    /* Calculate moles needed for reactants */
    for (uint32_t i = 0; i < reaction->num_reactants; i++) {
        uint32_t coeff = reaction->reactants[i].coefficient;
        float moles = moles_per_coeff * (float)coeff;
        result->moles_needed[i] = moles;
        result->grams_needed[i] = chemistry_moles_to_grams(
            moles, reaction->reactants[i].molecule.molar_mass);
    }

    /* Calculate moles produced for products */
    for (uint32_t i = 0; i < reaction->num_products; i++) {
        uint32_t coeff = reaction->products[i].coefficient;
        float moles = moles_per_coeff * (float)coeff;
        result->moles_produced[i] = moles;
        result->grams_produced[i] = chemistry_moles_to_grams(
            moles, reaction->products[i].molecule.molar_mass);
    }

    /* Theoretical yield is first product */
    if (reaction->num_products > 0) {
        result->theoretical_yield_g = result->grams_produced[0];
    }

    chem->stoichiometry_calcs++;
    nimcp_mutex_unlock(chem->lock);

    return 0;
}

uint32_t chemistry_find_limiting_reagent(
    chemistry_t* chem,
    const reaction_t* reaction,
    const float* reactant_moles
) {
    if (!chem || !reaction || !reactant_moles) return 0;
    if (reaction->num_reactants == 0) return 0;

    float min_ratio = INFINITY;
    uint32_t limiting_idx = 0;

    for (uint32_t i = 0; i < reaction->num_reactants; i++) {
        float coeff = (float)reaction->reactants[i].coefficient;
        float ratio = reactant_moles[i] / coeff;
        if (ratio < min_ratio) {
            min_ratio = ratio;
            limiting_idx = i;
        }
    }

    return limiting_idx;
}

float chemistry_percent_yield(float theoretical_yield, float actual_yield) {
    if (theoretical_yield < EPSILON) return 0.0f;
    return (actual_yield / theoretical_yield) * 100.0f;
}

float chemistry_moles_to_grams(float moles, float molar_mass) {
    return moles * molar_mass;
}

float chemistry_grams_to_moles(float grams, float molar_mass) {
    if (molar_mass < EPSILON) return 0.0f;
    return grams / molar_mass;
}

/* ============================================================================
 * MODULATION API
 * ============================================================================ */

int chemistry_set_inflammation(chemistry_t* chem, float level) {
    if (!chem) return -1;

    nimcp_mutex_lock(chem->lock);
    chem->inflammation_level = clamp01(level);
    nimcp_mutex_unlock(chem->lock);

    return 0;
}

int chemistry_set_sleep_deprivation(chemistry_t* chem, float level) {
    if (!chem) return -1;

    nimcp_mutex_lock(chem->lock);
    chem->sleep_deprivation_level = clamp01(level);
    nimcp_mutex_unlock(chem->lock);

    return 0;
}

/* ============================================================================
 * STATISTICS API
 * ============================================================================ */

int chemistry_get_stats(const chemistry_t* chem, chemistry_stats_t* stats) {
    if (!chem || !stats) return -1;

    nimcp_mutex_lock(((chemistry_t*)chem)->lock);

    stats->molecules_parsed = chem->molecules_parsed;
    stats->reactions_balanced = chem->reactions_balanced;
    stats->stoichiometry_calcs = chem->stoichiometry_calcs;
    stats->property_lookups = chem->property_lookups;

    uint64_t total_ops = chem->molecules_parsed + chem->reactions_balanced +
                         chem->stoichiometry_calcs + chem->property_lookups;
    if (total_ops > 0) {
        stats->avg_processing_time_us =
            (float)(chem->total_processing_time_us / (double)total_ops);
    } else {
        stats->avg_processing_time_us = 0.0f;
    }

    nimcp_mutex_unlock(((chemistry_t*)chem)->lock);
    return 0;
}

void chemistry_reset_stats(chemistry_t* chem) {
    if (!chem) return;

    nimcp_mutex_lock(chem->lock);
    chem->molecules_parsed = 0;
    chem->reactions_balanced = 0;
    chem->stoichiometry_calcs = 0;
    chem->property_lookups = 0;
    chem->total_processing_time_us = 0.0;
    nimcp_mutex_unlock(chem->lock);
}

const char* chemistry_get_last_error(void) {
    return g_chemistry_error;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int chemistry_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Chemistry_Reasoning");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Module self-knowledge logged */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Chemistry_Reasoning");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Chemistry_Reasoning");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
