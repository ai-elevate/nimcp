//=============================================================================
// nimcp_brain_processing.c - Brain Processing and Decision Making
//=============================================================================
/**
 * @file nimcp_brain_processing.c
 * @brief Forward pass computation and decision making logic
 *
 * This module contains approximately 1400 lines extracted from nimcp_brain.c:
 * - perform_forward_pass() - Neural network forward propagation
 * - brain_decide() - Decision making and inference
 * - Helper functions for label determination and statistics
 *
 * @version 1.0.0
 * @date 2025-12-08
 */

#include "core/brain/nimcp_brain_processing.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "plasticity/adaptive/nimcp_adaptive.h"

#define LOG_MODULE "BRAIN_PROC"

// NOTE: Implementation functions are currently in nimcp_brain.c
// This file provides modular organization and will be fully populated
// in a future migration phase.
