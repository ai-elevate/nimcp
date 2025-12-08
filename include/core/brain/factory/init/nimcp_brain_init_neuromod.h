//=============================================================================
// nimcp_brain_init_neuromod.h - Neuromodulation Subsystems
//=============================================================================
/**
 * @file nimcp_brain_init_neuromod.h
 * @brief Neuromodulation Subsystems
 *
 * WHAT: Initialization functions for neuromod subsystems
 * WHY:  SRP refactoring - separate neuromod initialization logic
 * HOW:  Each function initializes a specific brain subsystem
 *
 * EXTRACTED FROM: nimcp_brain_init_subsystems.c
 * DATE: 2025-12-08
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

#ifndef NIMCP_BRAIN_INIT_NEUROMOD_H
#define NIMCP_BRAIN_INIT_NEUROMOD_H

#include "core/brain/nimcp_brain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Neuromodulation Subsystems initialization functions
bool nimcp_brain_factory_init_neuromodulator_system(brain_t brain);
bool nimcp_brain_factory_init_spatial_neuromod_system(brain_t brain);
bool nimcp_brain_factory_init_attention_subsystem(brain_t brain);
bool nimcp_brain_factory_init_brain_regions_subsystem(brain_t brain);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_INIT_NEUROMOD_H
