/**
 * @file nimcp_phonological_loop.c
 * @brief Slice 5 — phonological-loop working-memory buffer.
 *
 * Implements Baddeley's phonological store + articulatory rehearsal for
 * the recurrent cascade. The buffer lives on brain_struct (loop_*
 * fields); these helpers are the only writers + readers.
 *
 * Concurrency: all mutating helpers take brain->loop_mutex internally.
 * Caller MUST NOT hold bridge/broca mutexes when entering — see
 * docs/claude/modules/lock-ordering.md.
 *
 * Allocation policy: grow-only. realloc failures are non-fatal; the
 * loop truncates writes rather than crashing. The buffer + trace +
 * words arrays are allocated lazily on first use (capacity > 0 at
 * init time).
 */

#include "language/nimcp_phonological_loop.h"

#include "core/brain/nimcp_brain_internal.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LOG_MODULE "PHONO_LOOP"

#define LOOP_INITIAL_BUFFER     256u
#define LOOP_INITIAL_TRACES     16u
#define LOOP_DEFAULT_MAX_WORDS  16u
#define LOOP_DEFAULT_DECAY      0.15f
#define LOOP_PRUNE_THRESHOLD    0.05f

/*============================================================================
 * Helpers (internal)
 *==========================================================================*/

static inline uint64_t monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000);
}

static inline int is_word_delim(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '.' || c == ',' || c == '!' || c == '?' ||
           c == ';' || c == ':' || c == '"' || c == '\'';
}

/* Find a word in the loop_words array. Returns its index, or UINT32_MAX
 * if not present. Case-sensitive lookup — the merge path normalizes to
 * lowercase before inserting. Caller holds the loop_mutex. */
static uint32_t find_word_unlocked(const brain_t brain, const char* word) {
    if (!brain->loop_words || !word) return UINT32_MAX;
    for (uint32_t i = 0; i < brain->loop_trace_count; i++) {
        const char* w = brain->loop_words[i];
        if (w && strcmp(w, word) == 0) return i;
    }
    return UINT32_MAX;
}

/* Grow the trace + words arrays to at least new_cap slots. Returns 0 on
 * success, -1 on alloc failure (state unchanged on failure). Caller
 * holds the loop_mutex. */
static int grow_trace_capacity_unlocked(brain_t brain, uint32_t new_cap) {
    if (new_cap <= brain->loop_trace_capacity) return 0;

    float* new_trace = (float*)nimcp_calloc(new_cap, sizeof(float));
    if (!new_trace) return -1;
    char** new_words = (char**)nimcp_calloc(new_cap, sizeof(char*));
    if (!new_words) {
        nimcp_free(new_trace);
        return -1;
    }

    if (brain->loop_phonemic_trace && brain->loop_trace_count > 0) {
        memcpy(new_trace, brain->loop_phonemic_trace,
               brain->loop_trace_count * sizeof(float));
    }
    if (brain->loop_words && brain->loop_trace_count > 0) {
        memcpy(new_words, brain->loop_words,
               brain->loop_trace_count * sizeof(char*));
    }

    if (brain->loop_phonemic_trace) nimcp_free(brain->loop_phonemic_trace);
    if (brain->loop_words)          nimcp_free(brain->loop_words);

    brain->loop_phonemic_trace = new_trace;
    brain->loop_words          = new_words;
    brain->loop_trace_capacity = new_cap;
    return 0;
}

/* Grow the surface buffer to at least new_cap bytes. Returns 0/-1; on
 * failure, state unchanged. Caller holds the loop_mutex. */
static int grow_buffer_capacity_unlocked(brain_t brain, uint32_t new_cap) {
    if (new_cap <= brain->loop_buffer_capacity) return 0;
    char* nb = (char*)nimcp_realloc(brain->loop_buffer, new_cap);
    if (!nb) return -1;
    /* Zero the freshly added tail so the buffer always reads as a valid
     * C-string up to capacity. */
    if (brain->loop_buffer_capacity < new_cap) {
        memset(nb + brain->loop_buffer_capacity, 0,
               new_cap - brain->loop_buffer_capacity);
    }
    brain->loop_buffer = nb;
    brain->loop_buffer_capacity = new_cap;
    return 0;
}

/* Re-render brain->loop_buffer from the surviving words in
 * loop_words/loop_phonemic_trace. Each word with trace >= threshold is
 * concatenated, space-separated. Truncates silently if the buffer can't
 * grow. Caller holds the loop_mutex. */
static void rerender_buffer_unlocked(brain_t brain, float threshold) {
    if (!brain->loop_buffer || brain->loop_buffer_capacity == 0) return;

    brain->loop_buffer[0] = '\0';
    brain->loop_buffer_len = 0;

    if (!brain->loop_words || brain->loop_trace_count == 0) return;

    /* Compute required size (cheap upfront, avoids realloc-per-token). */
    uint32_t need = 1;  /* trailing NUL */
    for (uint32_t i = 0; i < brain->loop_trace_count; i++) {
        if (brain->loop_phonemic_trace[i] < threshold) continue;
        const char* w = brain->loop_words[i];
        if (!w || !w[0]) continue;
        size_t wl = strlen(w);
        need += (uint32_t)wl;
        if (brain->loop_buffer_len + need > 1) need += 1;  /* space sep */
    }
    if (need > brain->loop_buffer_capacity) {
        /* Best-effort grow; if it fails we still render whatever fits. */
        uint32_t target = brain->loop_buffer_capacity;
        while (target < need) target = target * 2 + 16;
        (void)grow_buffer_capacity_unlocked(brain, target);
    }

    char* dst = brain->loop_buffer;
    uint32_t cap = brain->loop_buffer_capacity;
    if (cap == 0) return;
    uint32_t pos = 0;
    for (uint32_t i = 0; i < brain->loop_trace_count; i++) {
        if (brain->loop_phonemic_trace[i] < threshold) continue;
        const char* w = brain->loop_words[i];
        if (!w || !w[0]) continue;
        size_t wl = strlen(w);
        /* Need wl + (pos > 0 ? 1 : 0) + NUL. Stop if no room. */
        uint32_t need_here = (pos > 0 ? 1u : 0u) + (uint32_t)wl + 1u;
        if (pos + need_here > cap) break;
        if (pos > 0) dst[pos++] = ' ';
        memcpy(dst + pos, w, wl);
        pos += (uint32_t)wl;
    }
    if (pos >= cap) pos = cap - 1;
    dst[pos] = '\0';
    brain->loop_buffer_len = pos;
}

/* Strip non-ASCII control bytes and lowercase. Truncates to dst_max-1
 * including NUL. Returns the length actually written. */
static size_t normalize_token(const char* src, char* dst, size_t dst_max) {
    if (!src || !dst || dst_max == 0) return 0;
    size_t out = 0;
    for (const char* p = src; *p && out + 1 < dst_max; p++) {
        unsigned char c = (unsigned char)*p;
        if (c < 32) continue;
        dst[out++] = (char)tolower(c);
    }
    dst[out] = '\0';
    return out;
}

/*============================================================================
 * Lifecycle
 *==========================================================================*/

/* Internal lazy-init used by every helper. Returns 0 if the loop is
 * ready (already initialized or just-initialized), -1 on alloc failure.
 * Caller must NOT hold loop_mutex. */
static int ensure_init_unlocked(brain_t brain);
static int ensure_init_unlocked(brain_t brain) {
    if (!brain) return -1;
    if (brain->loop_mutex) return 0;
    return phonological_loop_init(brain);
}

int phonological_loop_init(brain_t brain) {
    if (!brain) return -1;
    /* Idempotent — if already initialized, return success. */
    if (brain->loop_mutex) return 0;

    brain->loop_mutex = (void*)nimcp_mutex_create(NULL);
    if (!brain->loop_mutex) {
        LOG_WARN(LOG_MODULE, "loop_mutex create failed (non-fatal, loop disabled)");
        return -1;
    }

    brain->loop_buffer = (char*)nimcp_calloc(LOOP_INITIAL_BUFFER, 1);
    brain->loop_buffer_capacity = brain->loop_buffer ? LOOP_INITIAL_BUFFER : 0u;
    brain->loop_buffer_len = 0;

    brain->loop_phonemic_trace =
        (float*)nimcp_calloc(LOOP_INITIAL_TRACES, sizeof(float));
    brain->loop_words = (char**)nimcp_calloc(LOOP_INITIAL_TRACES, sizeof(char*));
    brain->loop_trace_capacity =
        (brain->loop_phonemic_trace && brain->loop_words) ? LOOP_INITIAL_TRACES : 0u;
    brain->loop_trace_count = 0;

    /* Only set tunables if the caller hasn't already configured them.
     * brain_struct is calloc'd so a fresh brain has zero everywhere; a
     * setter run before init may have stamped a non-zero value that we
     * must preserve. Don't touch loop_enabled — the public contract is
     * "default OFF", but the caller may have flipped it to true via the
     * setter before lazy-init fired. */
    if (brain->loop_max_words == 0u) brain->loop_max_words = LOOP_DEFAULT_MAX_WORDS;
    if (brain->loop_decay_rate <= 0.0f) brain->loop_decay_rate = LOOP_DEFAULT_DECAY;
    brain->loop_last_refresh_ms = monotonic_ms();

    return 0;
}

void phonological_loop_destroy(brain_t brain) {
    if (!brain) return;
    /* Free under lock so concurrent readers (defense in depth) don't
     * dereference stale pointers mid-destroy. */
    nimcp_mutex_t* m = (nimcp_mutex_t*)brain->loop_mutex;
    if (m) nimcp_mutex_lock(m);

    if (brain->loop_words) {
        for (uint32_t i = 0; i < brain->loop_trace_count; i++) {
            if (brain->loop_words[i]) {
                nimcp_free(brain->loop_words[i]);
                brain->loop_words[i] = NULL;
            }
        }
        nimcp_free(brain->loop_words);
        brain->loop_words = NULL;
    }
    if (brain->loop_phonemic_trace) {
        nimcp_free(brain->loop_phonemic_trace);
        brain->loop_phonemic_trace = NULL;
    }
    if (brain->loop_buffer) {
        nimcp_free(brain->loop_buffer);
        brain->loop_buffer = NULL;
    }
    brain->loop_buffer_capacity = 0;
    brain->loop_buffer_len      = 0;
    brain->loop_trace_capacity  = 0;
    brain->loop_trace_count     = 0;
    brain->loop_enabled         = false;

    if (m) {
        nimcp_mutex_unlock(m);
        nimcp_mutex_free(m);
        brain->loop_mutex = NULL;
    }
}

/*============================================================================
 * Decay
 *==========================================================================*/

void phonological_loop_decay(brain_t brain) {
    if (!brain || !brain->loop_enabled) return;
    if (ensure_init_unlocked(brain) != 0) return;
    nimcp_mutex_t* m = (nimcp_mutex_t*)brain->loop_mutex;
    if (nimcp_mutex_lock(m) != 0) return;

    if (brain->loop_trace_count == 0) {
        nimcp_mutex_unlock(m);
        return;
    }

    float decay = brain->loop_decay_rate;
    if (decay < 0.0f) decay = 0.0f;
    if (decay > 0.5f) decay = 0.5f;

    /* Two-pass compaction: pass 1 multiplies traces, pass 2 keeps only
     * traces >= LOOP_PRUNE_THRESHOLD. Frees evicted word strings. */
    uint32_t write = 0;
    for (uint32_t read = 0; read < brain->loop_trace_count; read++) {
        float t = brain->loop_phonemic_trace[read] * (1.0f - decay);
        if (t < LOOP_PRUNE_THRESHOLD) {
            if (brain->loop_words[read]) {
                nimcp_free(brain->loop_words[read]);
                brain->loop_words[read] = NULL;
            }
            continue;
        }
        if (write != read) {
            brain->loop_phonemic_trace[write] = t;
            brain->loop_words[write] = brain->loop_words[read];
            brain->loop_words[read] = NULL;
        } else {
            brain->loop_phonemic_trace[write] = t;
        }
        write++;
    }
    /* Clear the tail of the trace array we shrunk past. */
    for (uint32_t i = write; i < brain->loop_trace_count; i++) {
        brain->loop_phonemic_trace[i] = 0.0f;
        brain->loop_words[i] = NULL;
    }
    brain->loop_trace_count = write;

    /* Re-render the surface buffer reflecting the survivors. */
    rerender_buffer_unlocked(brain, LOOP_PRUNE_THRESHOLD);
    brain->loop_last_refresh_ms = monotonic_ms();

    nimcp_mutex_unlock(m);
}

/*============================================================================
 * Merge
 *==========================================================================*/

void phonological_loop_merge_words(brain_t brain, const char* utterance) {
    if (!brain || !brain->loop_enabled) return;
    if (!utterance || !utterance[0]) return;
    if (ensure_init_unlocked(brain) != 0) return;

    nimcp_mutex_t* m = (nimcp_mutex_t*)brain->loop_mutex;
    if (nimcp_mutex_lock(m) != 0) return;

    /* Tokenize a local copy (utterance may live in caller-owned state). */
    enum { LOCAL_MAX = 2048 };
    char dup[LOCAL_MAX];
    size_t ul = strlen(utterance);
    if (ul >= sizeof(dup)) ul = sizeof(dup) - 1;
    memcpy(dup, utterance, ul);
    dup[ul] = '\0';

    char* save = NULL;
    char* tok  = strtok_r(dup, " \t\n\r.,!?;:\"'", &save);
    while (tok) {
        if (!tok[0]) { tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save); continue; }

        char norm[64];
        size_t nl = normalize_token(tok, norm, sizeof(norm));
        if (nl == 0) { tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save); continue; }

        uint32_t idx = find_word_unlocked(brain, norm);
        if (idx != UINT32_MAX) {
            /* Existing word — refresh trace to 1.0 (articulatory rehearsal). */
            brain->loop_phonemic_trace[idx] = 1.0f;
        } else if (brain->loop_trace_count < brain->loop_max_words) {
            /* New word — append, growing capacity if necessary. */
            if (brain->loop_trace_count >= brain->loop_trace_capacity) {
                uint32_t target = brain->loop_trace_capacity * 2;
                if (target < 16) target = 16;
                if (target > brain->loop_max_words) target = brain->loop_max_words;
                if (grow_trace_capacity_unlocked(brain, target) != 0) {
                    /* Alloc failure — silently drop this token. */
                    tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save);
                    continue;
                }
            }
            char* wcopy = (char*)nimcp_malloc(nl + 1);
            if (!wcopy) {
                tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save);
                continue;
            }
            memcpy(wcopy, norm, nl + 1);
            brain->loop_words[brain->loop_trace_count] = wcopy;
            brain->loop_phonemic_trace[brain->loop_trace_count] = 1.0f;
            brain->loop_trace_count++;
        }
        /* else: cap reached, ignore further new tokens this round. */

        tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save);
    }

    rerender_buffer_unlocked(brain, LOOP_PRUNE_THRESHOLD);
    brain->loop_last_refresh_ms = monotonic_ms();
    nimcp_mutex_unlock(m);
}

/*============================================================================
 * Render
 *==========================================================================*/

uint32_t phonological_loop_render_active(brain_t brain,
                                          float threshold,
                                          char* out_text,
                                          uint32_t out_max) {
    if (!brain || !out_text || out_max == 0) return 0;
    out_text[0] = '\0';
    if (!brain->loop_enabled) return 0;
    if (ensure_init_unlocked(brain) != 0) return 0;

    nimcp_mutex_t* m = (nimcp_mutex_t*)brain->loop_mutex;
    if (nimcp_mutex_lock(m) != 0) return 0;

    if (!isfinite(threshold) || threshold < 0.0f) threshold = 0.3f;

    uint32_t pos = 0;
    uint32_t words = 0;
    for (uint32_t i = 0; i < brain->loop_trace_count; i++) {
        if (brain->loop_phonemic_trace[i] < threshold) continue;
        const char* w = brain->loop_words[i];
        if (!w || !w[0]) continue;
        size_t wl = strlen(w);
        uint32_t need = (pos > 0 ? 1u : 0u) + (uint32_t)wl + 1u;
        if (pos + need > out_max) break;
        if (pos > 0) out_text[pos++] = ' ';
        memcpy(out_text + pos, w, wl);
        pos += (uint32_t)wl;
        words++;
    }
    out_text[pos] = '\0';

    nimcp_mutex_unlock(m);
    return words;
}

/*============================================================================
 * Clear
 *==========================================================================*/

void phonological_loop_clear(brain_t brain) {
    if (!brain) return;
    /* Clear of an uninitialized loop is a no-op — there's nothing
     * holding state. Don't lazy-init: the caller may not yet want the
     * loop to exist (e.g. a setup script calling clear before enable). */
    if (!brain->loop_mutex) return;
    nimcp_mutex_t* m = (nimcp_mutex_t*)brain->loop_mutex;
    if (nimcp_mutex_lock(m) != 0) return;

    if (brain->loop_words) {
        for (uint32_t i = 0; i < brain->loop_trace_count; i++) {
            if (brain->loop_words[i]) {
                nimcp_free(brain->loop_words[i]);
                brain->loop_words[i] = NULL;
            }
        }
    }
    if (brain->loop_phonemic_trace) {
        memset(brain->loop_phonemic_trace, 0,
               brain->loop_trace_capacity * sizeof(float));
    }
    brain->loop_trace_count = 0;

    if (brain->loop_buffer && brain->loop_buffer_capacity > 0) {
        brain->loop_buffer[0] = '\0';
    }
    brain->loop_buffer_len = 0;
    brain->loop_last_refresh_ms = monotonic_ms();

    nimcp_mutex_unlock(m);
}
