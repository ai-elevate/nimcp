/**
 * @file nimcp_integration.c
 * @brief Implementation of numerical integration methods
 *
 * WHAT: Implements Euler, RK4, and future adaptive integration methods
 * WHY:  Provides accurate ODE integration for NIMCP neuron dynamics
 * HOW:  Classic algorithms with careful memory management and error handling
 *
 * IMPLEMENTATION NOTES:
 * - Minimize allocations: Reuse temp buffers where possible
 * - Input validation: Check all parameters before computation
 * - Error propagation: NaN/inf in derivatives propagates to state
 * - No global state: All functions are reentrant
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Stack allocation for small n (< 32), heap for large n
 * - Const correctness for compiler optimizations
 * - Cache-friendly memory access patterns
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 */

#include "nimcp_integration.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define STACK_ALLOC_THRESHOLD 32  // Use stack for n <= this, heap otherwise

//=============================================================================
// Error Handling
//=============================================================================

static void integration_error(const char* message) {
    fprintf(stderr, "NIMCP Integration Error: %s\n", message);
}

//=============================================================================
// Core Integration Functions
//=============================================================================

bool integration_step(
    integration_method_t method,
    derivative_fn_t derivative_fn,
    float* state,
    float t,
    float dt,
    uint32_t n,
    void* params)
{
    // Input validation
    if (state == NULL) {
        integration_error("state is NULL");
        return false;
    }

    if (derivative_fn == NULL) {
        integration_error("derivative_fn is NULL");
        return false;
    }

    if (n == 0) {
        integration_error("dimension n is zero");
        return false;
    }

    if (dt <= 0.0f) {
        integration_error("timestep dt must be positive");
        return false;
    }

    if (!isfinite(t) || !isfinite(dt)) {
        integration_error("time or timestep is not finite");
        return false;
    }

    // Dispatch to method-specific implementation
    switch (method) {
        case INTEGRATION_EULER:
            return integration_euler_step(derivative_fn, state, t, dt, n, params);

        case INTEGRATION_RK4:
            return integration_rk4_step(derivative_fn, state, t, dt, n, params);

        case INTEGRATION_ADAPTIVE: {
            // Adaptive requires different calling convention (t and dt are modified)
            // For single-step API, we use default config and ignore timestep changes
            float t_temp = t;
            float dt_temp = dt;
            bool result = integration_adaptive_step(derivative_fn, state, &t_temp, &dt_temp, n, params, NULL);
            return result;
        }

        case INTEGRATION_IMPLICIT:
            integration_error("Implicit integration not yet implemented (future: A1.3)");
            return false;

        default:
            integration_error("Unknown integration method");
            return false;
    }
}

bool integration_integrate(
    integration_method_t method,
    derivative_fn_t derivative_fn,
    float* state,
    float t_start,
    float t_end,
    float dt,
    uint32_t n,
    void* params,
    float** trajectory,
    uint32_t* num_steps)
{
    // Input validation
    if (state == NULL || derivative_fn == NULL) {
        integration_error("NULL pointer in integration_integrate");
        return false;
    }

    if (t_end < t_start) {
        integration_error("t_end must be >= t_start");
        return false;
    }

    if (dt <= 0.0f) {
        integration_error("timestep dt must be positive");
        return false;
    }

    if (n == 0) {
        integration_error("dimension n is zero");
        return false;
    }

    // Calculate number of steps
    float duration = t_end - t_start;
    uint32_t steps = (uint32_t)(duration / dt);
    float remainder = duration - (steps * dt);

    // If there's a remainder, we need one more step with shorter dt
    bool need_final_step = (remainder > 1e-9f);
    if (need_final_step) {
        steps++;
    }

    // Allocate trajectory if requested
    if (trajectory != NULL && *trajectory == NULL) {
        *trajectory = (float*)malloc(steps * n * sizeof(float));
        if (*trajectory == NULL) {
            integration_error("Failed to allocate trajectory memory");
            return false;
        }
    }

    // Perform integration
    float t = t_start;
    for (uint32_t step = 0; step < steps; step++) {
        // Determine timestep for this iteration
        float step_dt = dt;
        if (need_final_step && step == steps - 1) {
            step_dt = remainder;  // Final shortened step
        }

        // Save state to trajectory if requested
        if (trajectory != NULL && *trajectory != NULL) {
            memcpy(*trajectory + step * n, state, n * sizeof(float));
        }

        // Perform integration step
        bool success = integration_step(method, derivative_fn, state, t, step_dt, n, params);
        if (!success) {
            if (num_steps != NULL) {
                *num_steps = step;  // Report how far we got
            }
            return false;
        }

        t += step_dt;
    }

    // Report number of steps
    if (num_steps != NULL) {
        *num_steps = steps;
    }

    return true;
}

//=============================================================================
// Euler Integration
//=============================================================================

bool integration_euler_step(
    derivative_fn_t derivative_fn,
    float* state,
    float t,
    float dt,
    uint32_t n,
    void* params)
{
    // Allocate temp storage for derivatives
    float temp_derivatives[STACK_ALLOC_THRESHOLD];
    float* derivatives = temp_derivatives;
    bool heap_allocated = false;

    if (n > STACK_ALLOC_THRESHOLD) {
        derivatives = (float*)malloc(n * sizeof(float));
        if (derivatives == NULL) {
            integration_error("Euler: Failed to allocate derivatives");
            return false;
        }
        heap_allocated = true;
    }

    // Compute derivatives: k = f(y, t)
    derivative_fn(state, t, params, derivatives);

    // Update state: y_new = y + dt * k
    for (uint32_t i = 0; i < n; i++) {
        state[i] += dt * derivatives[i];
    }

    // Cleanup
    if (heap_allocated) {
        free(derivatives);
    }

    return true;
}

//=============================================================================
// RK4 Integration
//=============================================================================

bool integration_rk4_step(
    derivative_fn_t derivative_fn,
    float* state,
    float t,
    float dt,
    uint32_t n,
    void* params)
{
    // Allocate temp storage
    // Need: k1[n], k2[n], k3[n], k4[n], temp_state[n]
    // Total: 5n floats

    float temp_stack[STACK_ALLOC_THRESHOLD * 5];
    float* temp_storage = temp_stack;
    bool heap_allocated = false;

    if (n > STACK_ALLOC_THRESHOLD) {
        temp_storage = (float*)malloc(5 * n * sizeof(float));
        if (temp_storage == NULL) {
            integration_error("RK4: Failed to allocate temporary storage");
            return false;
        }
        heap_allocated = true;
    }

    // Set up pointers into temp storage
    float* k1 = temp_storage;
    float* k2 = temp_storage + n;
    float* k3 = temp_storage + 2 * n;
    float* k4 = temp_storage + 3 * n;
    float* temp_state = temp_storage + 4 * n;

    // Stage 1: k1 = f(y, t)
    derivative_fn(state, t, params, k1);

    // Stage 2: k2 = f(y + dt*k1/2, t + dt/2)
    for (uint32_t i = 0; i < n; i++) {
        temp_state[i] = state[i] + 0.5f * dt * k1[i];
    }
    derivative_fn(temp_state, t + 0.5f * dt, params, k2);

    // Stage 3: k3 = f(y + dt*k2/2, t + dt/2)
    for (uint32_t i = 0; i < n; i++) {
        temp_state[i] = state[i] + 0.5f * dt * k2[i];
    }
    derivative_fn(temp_state, t + 0.5f * dt, params, k3);

    // Stage 4: k4 = f(y + dt*k3, t + dt)
    for (uint32_t i = 0; i < n; i++) {
        temp_state[i] = state[i] + dt * k3[i];
    }
    derivative_fn(temp_state, t + dt, params, k4);

    // Final update: y_new = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    for (uint32_t i = 0; i < n; i++) {
        state[i] += (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    // Cleanup
    if (heap_allocated) {
        free(temp_storage);
    }

    return true;
}

//=============================================================================
// Adaptive RK45 Integration (Dormand-Prince)
//=============================================================================

/**
 * @brief Dormand-Prince RK45 adaptive integration step
 *
 * WHAT: 5th-order Runge-Kutta with 4th-order error estimate and adaptive timestep
 * WHY:  Automatic accuracy control, 3-10x faster for slowly-varying dynamics
 * HOW:  Uses 6 evaluations to compute 5th-order solution + 4th-order error estimate
 *
 * ALGORITHM (Dormand-Prince 1980):
 *   k1 = f(y, t)
 *   k2 = f(y + dt*k1/5, t + dt/5)
 *   k3 = f(y + dt*(3*k1 + 9*k2)/40, t + 3*dt/10)
 *   k4 = f(y + dt*(44*k1 - 165*k2 + 55*k3)/45, t + 4*dt/5)
 *   k5 = f(y + dt*(19372*k1 - 76080*k2 + 64448*k3 - 1908*k4)/6561, t + 8*dt/9)
 *   k6 = f(y + dt*(9017*k1 - 35568*k2 + 64488*k3 - 15662*k4 + 492*k5)/2187, t + dt)
 *
 *   y_5th = y + dt * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
 *   y_4th = y + dt * (5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7)
 *
 *   error = |y_5th - y_4th|
 *   dt_new = dt * (tol/error)^(1/5)
 *
 * @param derivative_fn Derivative function
 * @param state State [n] (modified in-place with accepted step)
 * @param t Current time (modified to t + actual_dt_taken)
 * @param dt Timestep (modified to suggested dt for next step)
 * @param n Dimension
 * @param params User parameters
 * @param config Adaptive configuration
 * @return true if step accepted, false on error
 */
bool integration_adaptive_step(
    derivative_fn_t derivative_fn,
    float* state,
    float* t,
    float* dt,
    uint32_t n,
    void* params,
    const adaptive_config_t* config)
{
    // Default config if not provided
    adaptive_config_t default_config = {
        .min_timestep = 1e-6f,
        .max_timestep = 1.0f,
        .error_tolerance = 1e-6f,
        .max_steps = 10000
    };
    if (config == NULL) {
        config = &default_config;
    }

    // Allocate temp storage (7n floats: k1-k6, temp_state)
    float temp_stack[STACK_ALLOC_THRESHOLD * 7];
    float* temp_storage = temp_stack;
    bool heap_allocated = false;

    if (n > STACK_ALLOC_THRESHOLD) {
        temp_storage = (float*)malloc(7 * n * sizeof(float));
        if (temp_storage == NULL) {
            integration_error("Adaptive: Failed to allocate temporary storage");
            return false;
        }
        heap_allocated = true;
    }

    // Set up pointers
    float* k1 = temp_storage;
    float* k2 = temp_storage + n;
    float* k3 = temp_storage + 2 * n;
    float* k4 = temp_storage + 3 * n;
    float* k5 = temp_storage + 4 * n;
    float* k6 = temp_storage + 5 * n;
    float* temp_state = temp_storage + 6 * n;

    // Dormand-Prince coefficients
    const float c2 = 1.0f/5.0f, c3 = 3.0f/10.0f, c4 = 4.0f/5.0f, c5 = 8.0f/9.0f;

    // a coefficients (for intermediate stages)
    const float a21 = 1.0f/5.0f;
    const float a31 = 3.0f/40.0f, a32 = 9.0f/40.0f;
    const float a41 = 44.0f/45.0f, a42 = -56.0f/15.0f, a43 = 32.0f/9.0f;
    const float a51 = 19372.0f/6561.0f, a52 = -25360.0f/2187.0f, a53 = 64448.0f/6561.0f, a54 = -212.0f/729.0f;
    const float a61 = 9017.0f/3168.0f, a62 = -355.0f/33.0f, a63 = 46732.0f/5247.0f, a64 = 49.0f/176.0f, a65 = -5103.0f/18656.0f;

    // b coefficients (5th order solution)
    const float b1 = 35.0f/384.0f, b3 = 500.0f/1113.0f, b4 = 125.0f/192.0f;
    const float b5 = -2187.0f/6784.0f, b6 = 11.0f/84.0f;

    // b* coefficients (4th order solution for error estimate)
    const float bs1 = 5179.0f/57600.0f, bs3 = 7571.0f/16695.0f, bs4 = 393.0f/640.0f;
    const float bs5 = -92097.0f/339200.0f, bs6 = 187.0f/2100.0f, bs7 = 1.0f/40.0f;

    float current_dt = *dt;
    bool step_accepted = false;
    uint32_t attempts = 0;

    while (!step_accepted && attempts < 10) {
        attempts++;

        // Clamp timestep
        if (current_dt < config->min_timestep) current_dt = config->min_timestep;
        if (current_dt > config->max_timestep) current_dt = config->max_timestep;

        // Stage 1: k1 = f(y, t)
        derivative_fn(state, *t, params, k1);

        // Stage 2: k2 = f(y + dt*a21*k1, t + c2*dt)
        for (uint32_t i = 0; i < n; i++) {
            temp_state[i] = state[i] + current_dt * a21 * k1[i];
        }
        derivative_fn(temp_state, *t + c2 * current_dt, params, k2);

        // Stage 3: k3 = f(y + dt*(a31*k1 + a32*k2), t + c3*dt)
        for (uint32_t i = 0; i < n; i++) {
            temp_state[i] = state[i] + current_dt * (a31 * k1[i] + a32 * k2[i]);
        }
        derivative_fn(temp_state, *t + c3 * current_dt, params, k3);

        // Stage 4: k4 = f(y + dt*(a41*k1 + a42*k2 + a43*k3), t + c4*dt)
        for (uint32_t i = 0; i < n; i++) {
            temp_state[i] = state[i] + current_dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        derivative_fn(temp_state, *t + c4 * current_dt, params, k4);

        // Stage 5: k5 = f(y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4), t + c5*dt)
        for (uint32_t i = 0; i < n; i++) {
            temp_state[i] = state[i] + current_dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        derivative_fn(temp_state, *t + c5 * current_dt, params, k5);

        // Stage 6: k6 = f(y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5), t + dt)
        for (uint32_t i = 0; i < n; i++) {
            temp_state[i] = state[i] + current_dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        derivative_fn(temp_state, *t + current_dt, params, k6);

        // Compute 5th-order solution and error estimate
        float max_error = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            float y5 = state[i] + current_dt * (b1*k1[i] + b3*k3[i] + b4*k4[i] + b5*k5[i] + b6*k6[i]);
            float y4 = state[i] + current_dt * (bs1*k1[i] + bs3*k3[i] + bs4*k4[i] + bs5*k5[i] + bs6*k6[i] + bs7*k6[i]);
            float error_i = fabsf(y5 - y4);
            if (error_i > max_error) max_error = error_i;
            temp_state[i] = y5;  // Store 5th-order solution
        }

        // Check if step should be accepted
        if (max_error <= config->error_tolerance || current_dt <= config->min_timestep) {
            // Accept step
            for (uint32_t i = 0; i < n; i++) {
                state[i] = temp_state[i];
            }
            *t += current_dt;
            step_accepted = true;

            // Compute new timestep for next step
            if (max_error > 0.0f) {
                float safety = 0.9f;  // Safety factor
                float dt_new = current_dt * safety * powf(config->error_tolerance / max_error, 0.2f);  // 1/5 power
                *dt = dt_new;
            }
        } else {
            // Reject step, reduce timestep
            float safety = 0.9f;
            current_dt = current_dt * safety * powf(config->error_tolerance / max_error, 0.25f);  // More aggressive reduction
        }
    }

    // Cleanup
    if (heap_allocated) {
        free(temp_storage);
    }

    return step_accepted;
}

/**
 * @brief Integrate with adaptive timestep over interval [t_start, t_end]
 *
 * WHAT: Full adaptive integration from t_start to t_end with automatic error control
 * WHY:  More efficient than fixed-step for varying dynamics
 * HOW:  Repeatedly calls integration_adaptive_step() until t_end is reached
 */
bool integration_integrate_adaptive(
    derivative_fn_t derivative_fn,
    float* state,
    float t_start,
    float t_end,
    float dt_initial,
    uint32_t n,
    void* params,
    const adaptive_config_t* config,
    uint32_t* num_steps_taken)
{
    // Input validation
    if (state == NULL || derivative_fn == NULL) {
        integration_error("NULL pointer in integration_integrate_adaptive");
        return false;
    }

    if (t_end < t_start) {
        integration_error("t_end must be >= t_start");
        return false;
    }

    if (dt_initial <= 0.0f) {
        integration_error("initial timestep must be positive");
        return false;
    }

    if (n == 0) {
        integration_error("dimension n is zero");
        return false;
    }

    // Default config if not provided
    adaptive_config_t default_config = {
        .min_timestep = 1e-6f,
        .max_timestep = 1.0f,
        .error_tolerance = 1e-6f,
        .max_steps = 100000
    };
    if (config == NULL) {
        config = &default_config;
    }

    float t = t_start;
    float dt = dt_initial;
    uint32_t steps = 0;

    while (t < t_end && steps < config->max_steps) {
        // Don't overshoot t_end
        if (t + dt > t_end) {
            dt = t_end - t;
        }

        // Take adaptive step
        bool success = integration_adaptive_step(derivative_fn, state, &t, &dt, n, params, config);
        if (!success) {
            if (num_steps_taken != NULL) {
                *num_steps_taken = steps;
            }
            return false;
        }

        steps++;

        // Check if we've reached the end
        if (fabsf(t - t_end) < 1e-9f) {
            break;
        }
    }

    if (steps >= config->max_steps) {
        integration_error("Adaptive integration exceeded max_steps");
        if (num_steps_taken != NULL) {
            *num_steps_taken = steps;
        }
        return false;
    }

    // Report number of steps
    if (num_steps_taken != NULL) {
        *num_steps_taken = steps;
    }

    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* integration_method_name(integration_method_t method)
{
    switch (method) {
        case INTEGRATION_EULER:    return "Euler";
        case INTEGRATION_RK4:      return "RK4";
        case INTEGRATION_ADAPTIVE: return "Adaptive RK45";
        case INTEGRATION_IMPLICIT: return "Implicit";
        default:                   return "Unknown";
    }
}

float integration_method_cost(integration_method_t method)
{
    switch (method) {
        case INTEGRATION_EULER:    return 1.0f;   // Baseline
        case INTEGRATION_RK4:      return 4.0f;   // 4 derivative evaluations
        case INTEGRATION_ADAPTIVE: return 6.0f;   // ~6 evals on average
        case INTEGRATION_IMPLICIT: return 10.0f;  // Expensive (Jacobian, solve)
        default:                   return 1.0f;
    }
}

uint32_t integration_method_order(integration_method_t method)
{
    switch (method) {
        case INTEGRATION_EULER:    return 1;  // O(dt)
        case INTEGRATION_RK4:      return 4;  // O(dt^4)
        case INTEGRATION_ADAPTIVE: return 5;  // O(dt^5) local, dt^4 global
        case INTEGRATION_IMPLICIT: return 2;  // Depends on method
        default:                   return 0;
    }
}
