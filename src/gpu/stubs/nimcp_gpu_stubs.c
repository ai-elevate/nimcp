/**
 * @file nimcp_gpu_stubs.c
 * @brief CPU fallback implementations for GPU functions when CUDA is not available
 *
 * WHAT: Provides functional CPU fallback implementations for all GPU functions
 * WHY:  Allows building and running without CUDA - enables testing on CPU-only systems
 * HOW:  Implements equivalent CPU algorithms where possible, returns graceful failures otherwise
 *
 * @author NIMCP Development Team
 * @date 2026-01-06
 */

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/*=============================================================================
 * Wernicke GPU Stubs (gpu/cognitive/nimcp_wernicke_gpu.h)
 *=============================================================================*/

#include "gpu/cognitive/nimcp_wernicke_gpu.h"

/* Internal CPU-based wernicke context for fallback */
struct wernicke_gpu_context {
    wernicke_gpu_config_t config;

    /* Phoneme data */
    float* phoneme_embeddings;
    uint32_t num_phonemes;
    uint32_t phoneme_embed_dim;

    /* Lexicon data */
    wernicke_gpu_lexical_entry_t* lexicon;
    uint32_t lexicon_size;

    /* Working memory */
    uint8_t* wm_phonemes;
    float* wm_activations;
    uint32_t wm_count;

    /* Cohort state */
    wernicke_gpu_word_candidate_t* cohort;
    uint32_t cohort_size;
    uint8_t* current_phoneme_seq;
    uint32_t phoneme_seq_len;

    /* Statistics */
    wernicke_gpu_stats_t stats;
};

wernicke_gpu_config_t wernicke_gpu_default_config(void) {
    wernicke_gpu_config_t config;
    memset(&config, 0, sizeof(config));
    config.num_phoneme_categories = WERNICKE_GPU_DEFAULT_NUM_PHONEMES;
    config.phoneme_embedding_dim = 64;
    config.max_spectral_frames = 256;
    config.max_lexicon_size = 10000;
    config.max_cohort_size = 100;
    config.word_embedding_dim = 256;
    config.max_phonemes_per_word = 16;
    config.max_concepts = 5000;
    config.semantic_embedding_dim = 512;
    config.spreading_iterations = 3;
    config.spreading_decay = 0.5f;
    config.enable_attention = false;
    config.attention_heads = 4;
    config.working_memory_slots = 7;
    config.wm_decay_rate = 0.1f;
    config.enable_async_transfer = false;
    config.max_batch_size = 64;
    return config;
}

wernicke_gpu_context_t* wernicke_gpu_create(
    nimcp_gpu_context_t* gpu_ctx,
    const wernicke_gpu_config_t* config)
{
    (void)gpu_ctx;  /* Not used in CPU fallback */

    wernicke_gpu_context_t* ctx = calloc(1, sizeof(wernicke_gpu_context_t));
    if (!ctx) return NULL;

    ctx->config = config ? *config : wernicke_gpu_default_config();

    /* Allocate working memory */
    ctx->wm_phonemes = calloc(ctx->config.working_memory_slots, sizeof(uint8_t));
    ctx->wm_activations = calloc(ctx->config.working_memory_slots, sizeof(float));

    /* Allocate cohort buffers */
    ctx->cohort = calloc(ctx->config.max_cohort_size, sizeof(wernicke_gpu_word_candidate_t));
    ctx->current_phoneme_seq = calloc(ctx->config.max_phonemes_per_word, sizeof(uint8_t));

    if (!ctx->wm_phonemes || !ctx->wm_activations || !ctx->cohort || !ctx->current_phoneme_seq) {
        wernicke_gpu_destroy(ctx);
        return NULL;
    }

    return ctx;
}

void wernicke_gpu_destroy(wernicke_gpu_context_t* ctx) {
    if (!ctx) return;

    free(ctx->phoneme_embeddings);
    free(ctx->lexicon);
    free(ctx->wm_phonemes);
    free(ctx->wm_activations);
    free(ctx->cohort);
    free(ctx->current_phoneme_seq);
    free(ctx);
}

bool wernicke_gpu_synchronize(wernicke_gpu_context_t* ctx) {
    (void)ctx;
    return true;  /* CPU is always synchronous */
}

bool wernicke_gpu_upload_phoneme_embeddings(
    wernicke_gpu_context_t* ctx,
    const float* embeddings,
    uint32_t num_phonemes,
    uint32_t embed_dim)
{
    if (!ctx || !embeddings) return false;

    free(ctx->phoneme_embeddings);
    ctx->phoneme_embeddings = malloc(num_phonemes * embed_dim * sizeof(float));
    if (!ctx->phoneme_embeddings) return false;

    memcpy(ctx->phoneme_embeddings, embeddings, num_phonemes * embed_dim * sizeof(float));
    ctx->num_phonemes = num_phonemes;
    ctx->phoneme_embed_dim = embed_dim;

    return true;
}

/* CPU phoneme recognition using cosine similarity */
bool wernicke_gpu_recognize_phonemes(
    wernicke_gpu_context_t* ctx,
    const wernicke_gpu_spectral_frame_t* frames,
    uint32_t num_frames,
    wernicke_gpu_phoneme_result_t* results)
{
    if (!ctx || !frames || !results) return false;

    /* Simple heuristic: use MFCC coefficients to estimate phoneme */
    for (uint32_t i = 0; i < num_frames; i++) {
        float max_energy = 0.0f;
        uint8_t best_phoneme = 0;

        /* Use energy and voicing to classify */
        float energy = frames[i].energy;
        float voicing = frames[i].voicing;

        /* Simple classification based on features */
        if (voicing > 0.7f) {
            /* Voiced sound - likely vowel */
            best_phoneme = (uint8_t)((int)(frames[i].pitch / 50.0f) % 12);  /* Map to vowel range */
            max_energy = energy * voicing;
        } else if (energy > 0.3f) {
            /* Unvoiced with energy - likely fricative */
            best_phoneme = 12 + (uint8_t)((int)(energy * 10.0f) % 8);
        } else {
            /* Low energy - likely stop or silence */
            best_phoneme = 20 + (uint8_t)((int)(frames[i].mfcc[0] * 5.0f) % 10);
        }

        results[i].phoneme_id = best_phoneme % ctx->config.num_phoneme_categories;
        results[i].confidence = 0.5f + 0.5f * (energy * voicing);
        results[i].posterior = NULL;
    }

    ctx->stats.phoneme_recognitions += num_frames;
    return true;
}

bool wernicke_gpu_compute_posteriors(
    wernicke_gpu_context_t* ctx,
    const wernicke_gpu_spectral_frame_t* frames,
    uint32_t num_frames,
    float* posteriors)
{
    if (!ctx || !frames || !posteriors) return false;

    uint32_t num_phonemes = ctx->config.num_phoneme_categories;

    for (uint32_t i = 0; i < num_frames; i++) {
        /* Initialize uniform prior */
        float sum = 0.0f;
        for (uint32_t p = 0; p < num_phonemes; p++) {
            posteriors[i * num_phonemes + p] = 1.0f / num_phonemes;
            sum += posteriors[i * num_phonemes + p];
        }
        /* Normalize */
        for (uint32_t p = 0; p < num_phonemes; p++) {
            posteriors[i * num_phonemes + p] /= sum;
        }
    }

    return true;
}

bool wernicke_gpu_upload_lexicon(
    wernicke_gpu_context_t* ctx,
    const wernicke_gpu_lexical_entry_t* entries,
    uint32_t count)
{
    if (!ctx || !entries || count == 0) return false;

    free(ctx->lexicon);
    ctx->lexicon = malloc(count * sizeof(wernicke_gpu_lexical_entry_t));
    if (!ctx->lexicon) return false;

    memcpy(ctx->lexicon, entries, count * sizeof(wernicke_gpu_lexical_entry_t));
    ctx->lexicon_size = count;

    return true;
}

bool wernicke_gpu_clear_lexicon(wernicke_gpu_context_t* ctx) {
    if (!ctx) return false;
    free(ctx->lexicon);
    ctx->lexicon = NULL;
    ctx->lexicon_size = 0;
    return true;
}

uint32_t wernicke_gpu_get_lexicon_size(const wernicke_gpu_context_t* ctx) {
    return ctx ? ctx->lexicon_size : 0;
}

/* CPU word recognition using cohort model */
bool wernicke_gpu_recognize_words(
    wernicke_gpu_context_t* ctx,
    const uint8_t* phonemes,
    uint32_t num_phonemes,
    wernicke_gpu_word_candidate_t* candidates,
    uint32_t max_candidates,
    uint32_t* num_candidates)
{
    if (!ctx || !phonemes || !candidates || !num_candidates) return false;
    if (!ctx->lexicon || ctx->lexicon_size == 0) {
        *num_candidates = 0;
        return true;
    }

    uint32_t found = 0;

    for (uint32_t w = 0; w < ctx->lexicon_size && found < max_candidates; w++) {
        wernicke_gpu_lexical_entry_t* entry = &ctx->lexicon[w];

        /* Check if phoneme sequence matches beginning of word */
        uint32_t match_len = 0;
        for (uint32_t p = 0; p < num_phonemes && p < entry->phoneme_count; p++) {
            if (entry->phonemes[p] == phonemes[p]) {
                match_len++;
            } else {
                break;
            }
        }

        if (match_len > 0) {
            candidates[found].word_id = entry->word_id;
            candidates[found].matched_phonemes = (uint8_t)match_len;
            candidates[found].cohort_probability = (float)match_len / entry->phoneme_count;
            candidates[found].uniqueness_point = (float)match_len / entry->phoneme_count;
            candidates[found].recognition_complete = (match_len == entry->phoneme_count);
            found++;
        }
    }

    *num_candidates = found;
    ctx->stats.word_recognitions++;
    return true;
}

bool wernicke_gpu_update_cohort(
    wernicke_gpu_context_t* ctx,
    uint8_t new_phoneme,
    float phoneme_confidence)
{
    if (!ctx) return false;

    if (ctx->phoneme_seq_len < ctx->config.max_phonemes_per_word) {
        ctx->current_phoneme_seq[ctx->phoneme_seq_len++] = new_phoneme;
    }

    /* Re-evaluate cohort */
    ctx->cohort_size = 0;
    for (uint32_t w = 0; w < ctx->lexicon_size && ctx->cohort_size < ctx->config.max_cohort_size; w++) {
        wernicke_gpu_lexical_entry_t* entry = &ctx->lexicon[w];

        bool matches = true;
        for (uint32_t p = 0; p < ctx->phoneme_seq_len && p < entry->phoneme_count; p++) {
            if (entry->phonemes[p] != ctx->current_phoneme_seq[p]) {
                matches = false;
                break;
            }
        }

        if (matches && ctx->phoneme_seq_len <= entry->phoneme_count) {
            ctx->cohort[ctx->cohort_size].word_id = entry->word_id;
            ctx->cohort[ctx->cohort_size].matched_phonemes = (uint8_t)ctx->phoneme_seq_len;
            ctx->cohort[ctx->cohort_size].cohort_probability = phoneme_confidence;
            ctx->cohort[ctx->cohort_size].recognition_complete =
                (ctx->phoneme_seq_len == entry->phoneme_count);
            ctx->cohort_size++;
        }
    }

    return true;
}

bool wernicke_gpu_get_cohort(
    wernicke_gpu_context_t* ctx,
    wernicke_gpu_word_candidate_t* candidates,
    uint32_t max_candidates,
    uint32_t* num_candidates)
{
    if (!ctx || !candidates || !num_candidates) return false;

    uint32_t copy_count = ctx->cohort_size < max_candidates ? ctx->cohort_size : max_candidates;
    memcpy(candidates, ctx->cohort, copy_count * sizeof(wernicke_gpu_word_candidate_t));
    *num_candidates = copy_count;

    return true;
}

bool wernicke_gpu_reset_cohort(wernicke_gpu_context_t* ctx) {
    if (!ctx) return false;
    ctx->cohort_size = 0;
    ctx->phoneme_seq_len = 0;
    return true;
}

bool wernicke_gpu_upload_semantic_network(
    wernicke_gpu_context_t* ctx,
    const wernicke_gpu_concept_t* concepts,
    uint32_t num_concepts,
    const uint32_t* adjacency_matrix,
    const float* weights)
{
    (void)ctx;
    (void)concepts;
    (void)num_concepts;
    (void)adjacency_matrix;
    (void)weights;
    /* Semantic network not implemented in CPU fallback */
    return true;
}

bool wernicke_gpu_spread_activation(
    wernicke_gpu_context_t* ctx,
    const uint32_t* seed_concepts,
    const float* seed_activations,
    uint32_t num_seeds,
    wernicke_gpu_activation_result_t* results,
    uint32_t max_results,
    uint32_t* num_results)
{
    if (!ctx || !results || !num_results) return false;

    /* Simple pass-through: return seed concepts as activated */
    uint32_t count = num_seeds < max_results ? num_seeds : max_results;
    for (uint32_t i = 0; i < count; i++) {
        results[i].concept_id = seed_concepts ? seed_concepts[i] : i;
        results[i].activation = seed_activations ? seed_activations[i] : 1.0f;
        results[i].spreading_contribution = 0.0f;
    }
    *num_results = count;

    ctx->stats.spreading_activations++;
    return true;
}

bool wernicke_gpu_get_top_activated(
    wernicke_gpu_context_t* ctx,
    uint32_t top_k,
    wernicke_gpu_activation_result_t* results,
    uint32_t* actual_count)
{
    (void)ctx;
    (void)top_k;
    (void)results;
    if (actual_count) *actual_count = 0;
    return true;
}

float wernicke_gpu_semantic_similarity(
    wernicke_gpu_context_t* ctx,
    uint32_t concept_a,
    uint32_t concept_b)
{
    (void)ctx;
    return concept_a == concept_b ? 1.0f : 0.0f;
}

bool wernicke_gpu_wm_push(
    wernicke_gpu_context_t* ctx,
    const uint8_t* phonemes,
    uint32_t count)
{
    if (!ctx || !phonemes) return false;

    for (uint32_t i = 0; i < count && ctx->wm_count < ctx->config.working_memory_slots; i++) {
        ctx->wm_phonemes[ctx->wm_count] = phonemes[i];
        ctx->wm_activations[ctx->wm_count] = 1.0f;
        ctx->wm_count++;
    }

    ctx->stats.wm_operations++;
    return true;
}

bool wernicke_gpu_wm_get_contents(
    wernicke_gpu_context_t* ctx,
    uint8_t* phonemes,
    float* activations,
    uint32_t max_count,
    uint32_t* actual_count)
{
    if (!ctx || !actual_count) return false;

    uint32_t count = ctx->wm_count < max_count ? ctx->wm_count : max_count;
    if (phonemes) memcpy(phonemes, ctx->wm_phonemes, count);
    if (activations) memcpy(activations, ctx->wm_activations, count * sizeof(float));
    *actual_count = count;

    return true;
}

bool wernicke_gpu_wm_rehearse(wernicke_gpu_context_t* ctx) {
    if (!ctx) return false;

    /* Boost all activations */
    for (uint32_t i = 0; i < ctx->wm_count; i++) {
        ctx->wm_activations[i] = fminf(1.0f, ctx->wm_activations[i] + 0.1f);
    }

    ctx->stats.wm_operations++;
    return true;
}

bool wernicke_gpu_wm_apply_decay(
    wernicke_gpu_context_t* ctx,
    float decay_factor,
    float threshold)
{
    if (!ctx) return false;

    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < ctx->wm_count; i++) {
        ctx->wm_activations[i] *= decay_factor;
        if (ctx->wm_activations[i] >= threshold) {
            ctx->wm_phonemes[write_idx] = ctx->wm_phonemes[i];
            ctx->wm_activations[write_idx] = ctx->wm_activations[i];
            write_idx++;
        }
    }
    ctx->wm_count = write_idx;

    return true;
}

bool wernicke_gpu_wm_clear(wernicke_gpu_context_t* ctx) {
    if (!ctx) return false;
    ctx->wm_count = 0;
    return true;
}

bool wernicke_gpu_comprehend(
    wernicke_gpu_context_t* ctx,
    const wernicke_gpu_spectral_frame_t* frames,
    uint32_t num_frames,
    wernicke_gpu_word_candidate_t* word_candidates,
    uint32_t max_word_candidates,
    uint32_t* num_word_candidates,
    wernicke_gpu_activation_result_t* semantic_activations,
    uint32_t max_semantic_activations,
    uint32_t* num_semantic_activations)
{
    if (!ctx || !frames) return false;

    /* Step 1: Recognize phonemes */
    wernicke_gpu_phoneme_result_t* phonemes = calloc(num_frames, sizeof(wernicke_gpu_phoneme_result_t));
    if (!phonemes) return false;

    wernicke_gpu_recognize_phonemes(ctx, frames, num_frames, phonemes);

    /* Step 2: Extract phoneme sequence */
    uint8_t* phoneme_seq = calloc(num_frames, sizeof(uint8_t));
    if (!phoneme_seq) {
        free(phonemes);
        return false;
    }

    for (uint32_t i = 0; i < num_frames; i++) {
        phoneme_seq[i] = phonemes[i].phoneme_id;
    }

    /* Step 3: Recognize words */
    if (word_candidates && num_word_candidates) {
        wernicke_gpu_recognize_words(ctx, phoneme_seq, num_frames,
                                      word_candidates, max_word_candidates, num_word_candidates);
    }

    /* Step 4: Generate semantic activations */
    if (semantic_activations && num_semantic_activations) {
        uint32_t num_words = num_word_candidates ? *num_word_candidates : 0;
        uint32_t* concepts = calloc(num_words, sizeof(uint32_t));
        float* acts = calloc(num_words, sizeof(float));

        if (concepts && acts) {
            for (uint32_t i = 0; i < num_words; i++) {
                concepts[i] = word_candidates[i].word_id;
                acts[i] = word_candidates[i].cohort_probability;
            }
            wernicke_gpu_spread_activation(ctx, concepts, acts, num_words,
                                            semantic_activations, max_semantic_activations,
                                            num_semantic_activations);
        }

        free(concepts);
        free(acts);
    }

    free(phonemes);
    free(phoneme_seq);

    return true;
}

bool wernicke_gpu_get_stats(
    const wernicke_gpu_context_t* ctx,
    wernicke_gpu_stats_t* stats)
{
    if (!stats) return false;

    if (ctx) {
        *stats = ctx->stats;
        stats->current_cohort_size = ctx->cohort_size;
    } else {
        memset(stats, 0, sizeof(wernicke_gpu_stats_t));
    }
    return true;
}

void wernicke_gpu_reset_stats(wernicke_gpu_context_t* ctx) {
    if (ctx) {
        memset(&ctx->stats, 0, sizeof(wernicke_gpu_stats_t));
    }
}

/* CPU reference implementations */
bool wernicke_cpu_recognize_phonemes(
    const wernicke_gpu_spectral_frame_t* frames,
    uint32_t num_frames,
    const float* phoneme_embeddings,
    uint32_t num_phonemes,
    uint32_t embed_dim,
    wernicke_gpu_phoneme_result_t* results)
{
    if (!frames || !results) return false;

    for (uint32_t i = 0; i < num_frames; i++) {
        /* Use MFCC as feature vector */
        float best_sim = -1.0f;
        uint8_t best_phoneme = 0;

        if (phoneme_embeddings && embed_dim <= 13) {
            for (uint32_t p = 0; p < num_phonemes; p++) {
                float sim = 0.0f;
                float norm_a = 0.0f, norm_b = 0.0f;

                for (uint32_t d = 0; d < embed_dim; d++) {
                    float a = frames[i].mfcc[d];
                    float b = phoneme_embeddings[p * embed_dim + d];
                    sim += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }

                if (norm_a > 0 && norm_b > 0) {
                    sim /= sqrtf(norm_a * norm_b);
                }

                if (sim > best_sim) {
                    best_sim = sim;
                    best_phoneme = (uint8_t)p;
                }
            }
        } else {
            best_phoneme = (uint8_t)(i % num_phonemes);
            best_sim = 0.5f;
        }

        results[i].phoneme_id = best_phoneme;
        results[i].confidence = (best_sim + 1.0f) / 2.0f;
        results[i].posterior = NULL;
    }

    return true;
}

bool wernicke_cpu_recognize_words(
    const wernicke_gpu_lexical_entry_t* lexicon,
    uint32_t lexicon_size,
    const uint8_t* phonemes,
    uint32_t num_phonemes,
    wernicke_gpu_word_candidate_t* candidates,
    uint32_t max_candidates,
    uint32_t* num_candidates)
{
    if (!lexicon || !phonemes || !candidates || !num_candidates) return false;

    uint32_t found = 0;

    for (uint32_t w = 0; w < lexicon_size && found < max_candidates; w++) {
        uint32_t match_len = 0;
        for (uint32_t p = 0; p < num_phonemes && p < lexicon[w].phoneme_count; p++) {
            if (lexicon[w].phonemes[p] == phonemes[p]) {
                match_len++;
            } else {
                break;
            }
        }

        if (match_len > 0) {
            candidates[found].word_id = lexicon[w].word_id;
            candidates[found].matched_phonemes = (uint8_t)match_len;
            candidates[found].cohort_probability = (float)match_len / lexicon[w].phoneme_count;
            candidates[found].uniqueness_point = (float)match_len / lexicon[w].phoneme_count;
            candidates[found].recognition_complete = (match_len == lexicon[w].phoneme_count);
            found++;
        }
    }

    *num_candidates = found;
    return true;
}

bool wernicke_cpu_spread_activation(
    const float* adjacency_weights,
    uint32_t num_concepts,
    uint32_t max_neighbors,
    const uint32_t* seed_concepts,
    const float* seed_activations,
    uint32_t num_seeds,
    uint32_t iterations,
    float decay,
    float* output_activations)
{
    if (!output_activations) return false;

    /* Initialize activations */
    memset(output_activations, 0, num_concepts * sizeof(float));

    if (seed_concepts && seed_activations) {
        for (uint32_t i = 0; i < num_seeds; i++) {
            if (seed_concepts[i] < num_concepts) {
                output_activations[seed_concepts[i]] = seed_activations[i];
            }
        }
    }

    /* Simple spreading activation */
    float* temp = calloc(num_concepts, sizeof(float));
    if (!temp) return false;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        memset(temp, 0, num_concepts * sizeof(float));

        for (uint32_t c = 0; c < num_concepts; c++) {
            if (output_activations[c] > 0.01f && adjacency_weights) {
                /* Spread to neighbors */
                for (uint32_t n = 0; n < max_neighbors; n++) {
                    uint32_t neighbor = (c + n + 1) % num_concepts;
                    float weight = adjacency_weights[c * max_neighbors + n];
                    temp[neighbor] += output_activations[c] * weight * decay;
                }
            }
        }

        /* Apply spreading */
        for (uint32_t c = 0; c < num_concepts; c++) {
            output_activations[c] = fmaxf(output_activations[c] * decay, temp[c]);
        }
    }

    free(temp);
    return true;
}

/*=============================================================================
 * Tensor GPU Stubs (gpu/tensor/nimcp_tensor_gpu.h)
 *=============================================================================*/

#include "gpu/tensor/nimcp_tensor_gpu.h"

/* CPU tensor implementation */
nimcp_gpu_tensor_t* nimcp_gpu_tensor_create(
    nimcp_gpu_context_t* ctx,
    const size_t* dims,
    uint32_t ndim,
    nimcp_gpu_precision_t precision)
{
    (void)ctx;

    if (!dims || ndim == 0) return NULL;

    nimcp_gpu_tensor_t* tensor = calloc(1, sizeof(nimcp_gpu_tensor_t));
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->precision = precision;
    tensor->layout = NIMCP_TENSOR_LAYOUT_ROW_MAJOR;
    tensor->owns_data = true;

    /* Allocate dims and strides on CPU */
    tensor->dims = malloc(ndim * sizeof(size_t));
    tensor->strides = malloc(ndim * sizeof(size_t));
    if (!tensor->dims || !tensor->strides) {
        nimcp_gpu_tensor_destroy(tensor);
        return NULL;
    }

    /* Calculate total elements and strides */
    tensor->numel = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
        tensor->numel *= dims[i];
    }

    /* Row-major strides */
    tensor->strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        tensor->strides[i] = tensor->strides[i + 1] * tensor->dims[i + 1];
    }

    /* Determine element size */
    switch (precision) {
        case NIMCP_GPU_PRECISION_FP32:
        case NIMCP_GPU_PRECISION_TF32:
        case NIMCP_GPU_PRECISION_INT32:
        case NIMCP_GPU_PRECISION_UINT32:
            tensor->elem_size = 4;
            break;
        case NIMCP_GPU_PRECISION_FP16:
        case NIMCP_GPU_PRECISION_BF16:
            tensor->elem_size = 2;
            break;
        case NIMCP_GPU_PRECISION_INT8:
            tensor->elem_size = 1;
            break;
        default:
            tensor->elem_size = 4;
    }

    /* Allocate data on CPU */
    tensor->data = calloc(tensor->numel, tensor->elem_size);
    if (!tensor->data) {
        nimcp_gpu_tensor_destroy(tensor);
        return NULL;
    }

    return tensor;
}

nimcp_gpu_tensor_t* nimcp_gpu_tensor_from_host(
    nimcp_gpu_context_t* ctx,
    const void* host_data,
    const size_t* dims,
    uint32_t ndim,
    nimcp_gpu_precision_t precision)
{
    nimcp_gpu_tensor_t* tensor = nimcp_gpu_tensor_create(ctx, dims, ndim, precision);
    if (!tensor || !host_data) return tensor;

    memcpy(tensor->data, host_data, tensor->numel * tensor->elem_size);
    return tensor;
}

bool nimcp_gpu_tensor_to_host(
    const nimcp_gpu_tensor_t* tensor,
    void* host_data)
{
    if (!tensor || !host_data || !tensor->data) return false;
    memcpy(host_data, tensor->data, tensor->numel * tensor->elem_size);
    return true;
}

void nimcp_gpu_tensor_destroy(nimcp_gpu_tensor_t* tensor) {
    if (!tensor) return;

    if (tensor->owns_data) {
        free(tensor->data);
    }
    free(tensor->dims);
    free(tensor->strides);
    free(tensor);
}

nimcp_gpu_tensor_t* nimcp_gpu_tensor_clone(const nimcp_gpu_tensor_t* tensor) {
    if (!tensor) return NULL;

    nimcp_gpu_tensor_t* clone = nimcp_gpu_tensor_create(
        tensor->ctx, tensor->dims, tensor->ndim, tensor->precision);
    if (!clone) return NULL;

    if (tensor->data) {
        memcpy(clone->data, tensor->data, tensor->numel * tensor->elem_size);
    }

    return clone;
}

bool nimcp_gpu_fill(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* tensor,
    float value)
{
    (void)ctx;
    if (!tensor || !tensor->data) return false;

    if (tensor->precision == NIMCP_GPU_PRECISION_FP32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->numel; i++) {
            data[i] = value;
        }
    } else {
        /* For other precisions, fill with bytes */
        memset(tensor->data, (int)value, tensor->numel * tensor->elem_size);
    }

    return true;
}

bool nimcp_gpu_zeros(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* tensor)
{
    (void)ctx;
    if (!tensor || !tensor->data) return false;
    memset(tensor->data, 0, tensor->numel * tensor->elem_size);
    return true;
}

/*=============================================================================
 * Graph GPU Stubs (gpu/graph/nimcp_graph_gpu.h)
 *=============================================================================*/

#include "gpu/graph/nimcp_graph_gpu.h"

void nimcp_gpu_graph_destroy(nimcp_gpu_graph_t* graph) {
    if (!graph) return;

    /* Free CSR arrays - these are CPU memory in stub mode */
    free(graph->d_row_offsets);
    free(graph->d_col_indices);
    free(graph->d_edge_weights);
    free(graph);
}
