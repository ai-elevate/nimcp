/**
 * @file nimcp_bigram_spectrum.h
 * @brief FFT-based bigram spectral analytics — diagnostic tool that
 *        reveals whether grammar is emerging from next-token training.
 *
 * WHAT: Lightweight bigram-frequency tracker. Records (prev_id, next_id)
 *       pairs into a square count matrix and surfaces spectral metrics
 *       computed from a 2D FFT of that matrix.
 * WHY:  PA-4 wired next-token contrastive training. We need a separate,
 *       opt-in observable that says whether emerging structure is
 *       periodic / hub-dominated (grammar-like) or uniformly random
 *       (no structure). Spectral peak strength + low-frequency
 *       concentration + spectral entropy together capture this.
 * HOW:  Square uint32 count matrix [vocab_cap × vocab_cap]. compute()
 *       runs a row-then-column 1D FFT pass to approximate the 2D FFT,
 *       takes magnitude, and computes the three metrics.
 *
 * DESIGN:
 *   - Standalone — does NOT depend on grounded_language internals.
 *   - All allocations via nimcp_malloc / nimcp_free.
 *   - vocab_cap is rounded internally for FFT to next power-of-2 when
 *     building the FFT plan.
 *   - On `compute()`, the count matrix is normalized (centered + scaled)
 *     before being fed through the FFT to keep |F| bounded.
 *
 * USAGE:
 *   bigram_spectrum_t* bs = bigram_spectrum_create(256);
 *   for (...pairs...) bigram_spectrum_record(bs, p, n);
 *   bigram_spectral_metrics_t m;
 *   bigram_spectrum_compute(bs, &m);
 *   ...
 *   bigram_spectrum_destroy(bs);
 *
 * @author NIMCP Development Team
 * @date 2026-05-07
 */

#ifndef NIMCP_BIGRAM_SPECTRUM_H
#define NIMCP_BIGRAM_SPECTRUM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Spectral metrics of the bigram count matrix.
 *
 * peak_strength            (max - mean) / std    — z-score of the
 *                          dominant frequency. High when bigrams form a
 *                          tight repeating pattern (grammar-like).
 * low_freq_concentration   sum |F|^2 in inner ~1/16 area (N/8 half-width
 *                          per axis, summing all 4 corner regions of the
 *                          DFT) / total |F|^2. (Walkthrough round 4 doc
 *                          fix — was previously "inner-quarter" which
 *                          implied 1/4 area; actual computed region is
 *                          (N/4)² / N² = 1/16. Comment in .c is correct.)
 *                          High when transitions are dominated by a
 *                          few global hubs (function words, articles,
 *                          determiners — the kind of structure that
 *                          shows up first in real grammars).
 * spectral_entropy         -Σ p log p where p = |F|^2 / Σ |F|^2.
 *                          Low under structure; ≈ log(N) under noise.
 */
typedef struct {
    float peak_strength;
    float low_freq_concentration;
    float spectral_entropy;
} bigram_spectral_metrics_t;

/**
 * @brief Opaque bigram spectrum tracker.
 */
typedef struct bigram_spectrum_struct bigram_spectrum_t;

/**
 * @brief Create a bigram spectrum with a square count matrix
 *        of size vocab_cap × vocab_cap (uint32 entries).
 *
 * @param vocab_cap Maximum vocabulary id (exclusive upper bound on
 *                  prev_id / next_id). Must be > 1 and ≤ 4096.
 * @return Spectrum on success, NULL on failure.
 */
bigram_spectrum_t* bigram_spectrum_create(uint32_t vocab_cap);

/**
 * @brief Destroy and free.
 */
void bigram_spectrum_destroy(bigram_spectrum_t* spectrum);

/**
 * @brief Record a (prev_id, next_id) bigram occurrence by incrementing
 *        count[prev_id][next_id] (saturated at UINT32_MAX).
 *
 * Out-of-range ids are silently ignored (returns 0). NULL spectrum is
 * a no-op (returns 0).
 *
 * @return 1 if recorded, 0 if silently ignored, -1 on hard error.
 */
int bigram_spectrum_record(bigram_spectrum_t* spectrum,
                           uint32_t prev_id,
                           uint32_t next_id);

/**
 * @brief Compute spectral metrics from the current count matrix.
 *
 * If the matrix is empty (no records) all metrics are 0.
 *
 * @return 0 on success, -1 on failure.
 */
int bigram_spectrum_compute(bigram_spectrum_t* spectrum,
                            bigram_spectral_metrics_t* out);

/**
 * @brief Reset the count matrix to zero. Cached metrics are also cleared.
 *
 * @return 0 on success, -1 on failure.
 */
int bigram_spectrum_reset(bigram_spectrum_t* spectrum);

/**
 * @brief Return total recorded events (sum of all counts).
 *        Useful for tests and for "is this metric meaningful yet?".
 */
uint64_t bigram_spectrum_total_events(const bigram_spectrum_t* spectrum);

/**
 * @brief Return the matrix side length (== vocab_cap passed at create).
 */
uint32_t bigram_spectrum_vocab_cap(const bigram_spectrum_t* spectrum);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_BIGRAM_SPECTRUM_H */
