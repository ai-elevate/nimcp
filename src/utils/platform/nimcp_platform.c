/**
 * @file nimcp_platform.c
 * @brief Cross-platform utility implementations
 *
 * WHAT: Platform detection and utility functions
 * WHY:  Provide runtime information about platform/compiler/architecture
 * HOW:  Return compile-time constants
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "utils/platform/nimcp_platform.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

/* ========================================================================
 * PLATFORM NAME
 * ======================================================================== */

const char* nimcp_platform_name(void)
{
#if defined(NIMCP_PLATFORM_NAME)
    return NIMCP_PLATFORM_NAME;
#else
    return "Unknown";
#endif
}

/* ========================================================================
 * COMPILER NAME
 * ======================================================================== */

const char* nimcp_compiler_name(void)
{
#if defined(NIMCP_COMPILER_NAME)
    return NIMCP_COMPILER_NAME;
#else
    return "Unknown";
#endif
}

/* ========================================================================
 * ARCHITECTURE NAME
 * ======================================================================== */

const char* nimcp_architecture_name(void)
{
#if defined(NIMCP_ARCH_NAME)
    return NIMCP_ARCH_NAME;
#elif defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "ARM64";
#elif defined(__arm__) || defined(_M_ARM)
    return "ARM";
#else
    return "Unknown";
#endif
}
