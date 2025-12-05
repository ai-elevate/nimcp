/**
 * @file nimcp_platform_thread.c
 * @brief Cross-platform thread implementation (SRP: Thread lifecycle only)
 *
 * WHAT: Platform-agnostic thread creation/management
 * WHY:  pthread doesn't exist on Windows
 * HOW:  Wrap pthread (POSIX) and Windows threads
 *
 * SRP: This module has ONE responsibility - thread lifecycle management
 * COMPLEXITY: O(1) for all operations
 * THREAD-SAFE: Yes - all functions thread-safe
 *
 * @author NIMCP Development Team
 * @date 2025
 */

#include "utils/platform/nimcp_platform_thread.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include <errno.h>
#include <stdlib.h>

#if defined(NIMCP_PLATFORM_WINDOWS)
    #include <process.h>  /* _beginthreadex */
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#endif

/* ========================================================================
 * WINDOWS-SPECIFIC HELPERS
 * ======================================================================== */

#if defined(NIMCP_PLATFORM_WINDOWS)

/**
 * WHAT: Wrapper structure for Windows thread start
 * WHY:  Windows thread function signature differs from POSIX
 * HOW:  Store POSIX-style function pointer and arg, translate return value
 */
typedef struct {
    nimcp_thread_func_t func;
    void* arg;
    void* retval;
} win_thread_wrapper_t;

/**
 * WHAT: Windows thread start routine wrapper
 * WHY:  Windows expects DWORD WINAPI func(LPVOID), POSIX expects void* func(void*)
 * HOW:  Translate between signatures, store return value
 */
static DWORD WINAPI win_thread_start(LPVOID arg)
{
    win_thread_wrapper_t* wrapper = (win_thread_wrapper_t*)arg;
    wrapper->retval = wrapper->func(wrapper->arg);
    nimcp_free(wrapper);  /* Cleanup wrapper after thread completes */
    return 0;  /* Windows thread exit code (unused) */
}

#endif /* NIMCP_PLATFORM_WINDOWS */

/* ========================================================================
 * THREAD FUNCTIONS
 * ======================================================================== */

int nimcp_platform_thread_create(nimcp_platform_thread_t* thread,
                                 nimcp_thread_func_t start_routine,
                                 void* arg)
{
    if (!thread || !start_routine) {
        return EINVAL;
    }

#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Direct pthread_create */
    int result = pthread_create(thread, NULL, start_routine, arg);
    return result;

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Create thread with wrapper */
    win_thread_wrapper_t* wrapper = (win_thread_wrapper_t*)nimcp_malloc(sizeof(win_thread_wrapper_t));
    if (!wrapper) {
        return ENOMEM;
    }

    wrapper->func = start_routine;
    wrapper->arg = arg;
    wrapper->retval = NULL;

    HANDLE handle = (HANDLE)_beginthreadex(
        NULL,                 /* Security attributes */
        0,                    /* Stack size (0 = default) */
        (unsigned (__stdcall *)(void *))win_thread_start,
        wrapper,              /* Argument */
        0,                    /* Creation flags */
        NULL                  /* Thread ID (unused) */
    );

    if (handle == 0) {
        nimcp_free(wrapper);
        return EAGAIN;  /* Thread creation failed */
    }

    *thread = handle;
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_thread_join(nimcp_platform_thread_t thread, void** retval)
{
#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Direct pthread_join */
    return pthread_join(thread, retval);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Wait for thread completion */
    DWORD result = WaitForSingleObject(thread, INFINITE);
    if (result != WAIT_OBJECT_0) {
        return EINVAL;
    }

    /* Note: retval not supported on Windows with current implementation */
    if (retval) {
        *retval = NULL;  /* Windows limitation */
    }

    CloseHandle(thread);
    return 0;

#else
    #error "Unsupported platform"
#endif
}

int nimcp_platform_thread_detach(nimcp_platform_thread_t thread)
{
#if defined(NIMCP_PLATFORM_POSIX)
    /* POSIX: Direct pthread_detach */
    return pthread_detach(thread);

#elif defined(NIMCP_PLATFORM_WINDOWS)
    /* Windows: Close handle (thread runs independently) */
    if (!CloseHandle(thread)) {
        return EINVAL;
    }
    return 0;

#else
    #error "Unsupported platform"
#endif
}

nimcp_platform_thread_t nimcp_platform_thread_self(void)
{
#if defined(NIMCP_PLATFORM_POSIX)
    return pthread_self();

#elif defined(NIMCP_PLATFORM_WINDOWS)
    return GetCurrentThread();

#else
    #error "Unsupported platform"
#endif
}
