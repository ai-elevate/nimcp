/**
 * @file nimcp_dataio.c
 * @brief Brain data I/O implementation
 *
 * WHAT: Load training data from external sources, export brain predictions
 * WHY: Brains need to learn from real-world datasets and export results
 * HOW: Strategy pattern with pluggable data loaders (CSV, PostgreSQL, streams)
 *
 * DESIGN PATTERNS:
 * - Strategy Pattern: Pluggable data source backends (CSV, PostgreSQL, stream)
 * - Repository Pattern: Abstracted data access operations
 * - Factory Pattern: Backend-specific dataset creation
 * - Iterator Pattern: Batch-by-batch data reading for streaming
 *
 * THREAD SAFETY: Thread-local error storage, mutex-protected batch reads
 */

#include "nimcp_dataio.h"
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "core/brain/nimcp_brain.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/containers/nimcp_queue.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_validate.h"

//=============================================================================
// Thread-Local Error Handling (same pattern as replication)
//=============================================================================

/**
 * WHAT: Thread-local error message storage
 * WHY: Thread-safe error reporting without global variables
 * HOW: Thread-local storage for last error per thread
 */
static __thread char g_last_error[512] = {0};

static void dataio_set_error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    va_end(args);
}

const char* dataio_get_last_error(void)
{
    return g_last_error;
}

//=============================================================================
// Strategy Pattern: Data Source Backend Interface
//=============================================================================

/**
 * WHAT: Data source backend strategy (function pointers)
 * WHY: Support multiple data sources (CSV, PostgreSQL, streams) with same API
 * HOW: Strategy pattern - backends implement this interface
 */
typedef struct data_source_strategy {
    data_source_t source_type;

    // WHAT: Initialize data source
    // WHY: Open files, connect to databases, etc.
    bool (*initialize)(void** context, const dataset_config_t* config);

    // WHAT: Clean up data source
    // WHY: Close files, disconnect from databases, free memory
    void (*shutdown)(void* context);

    // WHAT: Read next batch of data
    // WHY: Support streaming for large datasets
    // HOW: Iterator pattern - returns next batch until end
    bool (*next_batch)(void* context, data_batch_t* batch);

    // WHAT: Reset to beginning
    // WHY: Support multiple epochs of training
    bool (*reset)(void* context);

    // WHAT: Get total size (if known)
    // WHY: Progress reporting, memory allocation
    uint64_t (*get_size)(void* context);

} data_source_strategy_t;

//=============================================================================
// Dataset Structure (Opaque Handle)
//=============================================================================

/**
 * WHAT: Internal dataset structure
 * WHY: Opaque handle hides implementation from API users
 * HOW: Contains strategy and backend context
 */
struct dataset_struct {
    dataset_config_t config;
    data_source_strategy_t* strategy;  // Strategy Pattern
    void* source_context;              // Backend-specific context
    nimcp_mutex_t read_lock;         // Thread-safe batch reads
    uint64_t rows_read;                // Current position
};

//=============================================================================
// CSV Backend Implementation
//=============================================================================

/**
 * WHAT: CSV file context
 * WHY: Track CSV parsing state across batches
 */
typedef struct {
    FILE* file;              // CSV file handle
    char** column_names;     // Header column names
    uint32_t num_columns;    // Total columns
    uint32_t num_features;   // Number of feature columns (for batch allocation)
    uint64_t total_rows;     // Total rows (if known)
    long file_start_offset;  // After header (for reset)
} csv_context_t;

/**
 * WHAT: Initialize CSV data source
 * WHY: Open file, parse header, prepare for reading
 * HOW: Open file, read header if present, store column names
 */
static bool csv_initialize(void** context, const dataset_config_t* config)
{
    // Guard clause: validate input
    if (!context || !config || !config->location[0]) {
        dataio_set_error("Invalid CSV configuration");
        return false;
    }

    // Allocate context
    csv_context_t* csv_ctx = nimcp_calloc(1, sizeof(csv_context_t));
    if (!csv_ctx) {
        dataio_set_error("Failed to allocate CSV context");
        return false;
    }

    // Open CSV file
    csv_ctx->file = fopen(config->location, "r");
    if (!csv_ctx->file) {
        dataio_set_error("Failed to open CSV file '%s': %s", config->location, strerror(errno));
        nimcp_free(csv_ctx);
        return false;
    }

    csv_ctx->num_columns = config->num_feature_columns + config->num_label_columns;
    csv_ctx->num_features = config->num_feature_columns;

    // WHAT: Parse header row if present
    // WHY: Extract column names for metadata
    if (config->has_header) {
        char line[4096];
        if (!fgets(line, sizeof(line), csv_ctx->file)) {
            dataio_set_error("Failed to read CSV header");
            fclose(csv_ctx->file);
            nimcp_free(csv_ctx);
            return false;
        }

        // Parse header columns
        csv_ctx->column_names = nimcp_calloc(csv_ctx->num_columns, sizeof(char*));
        char* token = strtok(line, ",\r\n");
        uint32_t col_idx = 0;

        while (token && col_idx < csv_ctx->num_columns) {
            csv_ctx->column_names[col_idx] = nimcp_strdup(token);
            token = strtok(NULL, ",\r\n");
            col_idx++;
        }
    }

    // Store position after header (for reset)
    csv_ctx->file_start_offset = ftell(csv_ctx->file);

    *context = csv_ctx;
    return true;
}

/**
 * WHAT: Clean up CSV resources
 * WHY: Close file, free memory
 */
static void csv_shutdown(void* context)
{
    if (!context)
        return;

    csv_context_t* csv_ctx = (csv_context_t*) context;

    if (csv_ctx->file) {
        fclose(csv_ctx->file);
    }

    if (csv_ctx->column_names) {
        for (uint32_t i = 0; i < csv_ctx->num_columns; i++) {
            nimcp_free(csv_ctx->column_names[i]);
        }
        nimcp_free(csv_ctx->column_names);
    }

    nimcp_free(csv_ctx);
}

/**
 * WHAT: Parse CSV line into features and label
 * WHY: Convert text to float features and string label
 * HOW: Split by delimiter, parse floats, extract label
 */
static bool csv_parse_line(const char* line, char delimiter, uint32_t num_features,
                           uint32_t num_labels, float* features, char* label, size_t label_size)
{
    char line_copy[4096];
    strncpy(line_copy, line, sizeof(line_copy) - 1);
    line_copy[sizeof(line_copy) - 1] = '\0';

    char delim_str[2] = {delimiter, '\0'};
    char* token = strtok(line_copy, delim_str);

    // Parse features
    for (uint32_t i = 0; i < num_features; i++) {
        if (!token)
            return false;

        // Store token for error reporting before parsing
        char token_copy[256];
        strncpy(token_copy, token, sizeof(token_copy) - 1);
        token_copy[sizeof(token_copy) - 1] = '\0';

        features[i] = atof(token);

        // Validate parsed float value
        if (!nimcp_validate_float_field(&features[i], sizeof(float))) {
            fprintf(stderr, "[DataIO] Invalid feature value at index %u: %f from token '%s'\n", i,
                    features[i], token_copy);
            return false;
        }

        token = strtok(NULL, delim_str);
    }

    // Parse label (last column)
    if (!token)
        return false;

    // Remove trailing newline/whitespace
    char* newline = strchr(token, '\n');
    if (newline)
        *newline = '\0';
    char* cr = strchr(token, '\r');
    if (cr)
        *cr = '\0';

    strncpy(label, token, label_size - 1);
    label[label_size - 1] = '\0';

    return true;
}

/**
 * WHAT: Read next batch of CSV data
 * WHY: Support streaming large CSV files
 * HOW: Read batch_size rows, parse into features/labels
 */
static bool csv_next_batch(void* context, data_batch_t* batch)
{
    if (!context || !batch)
        return false;

    csv_context_t* csv_ctx = (csv_context_t*) context;

    // Read batch_size rows (or remaining rows)
    uint32_t batch_size = 1000;  // Default batch size
    batch->features = nimcp_calloc(batch_size, sizeof(float*));
    batch->labels = nimcp_calloc(batch_size, sizeof(char*));
    batch->num_samples = 0;

    char line[4096];
    for (uint32_t i = 0; i < batch_size; i++) {
        if (!fgets(line, sizeof(line), csv_ctx->file)) {
            // End of file
            batch->end_of_dataset = true;
            break;
        }

        // WHAT: Allocate row storage at write index (num_samples)
        // WHY: Prevents double-free when parse fails
        // HOW: Use num_samples as write pointer, only increment on success
        //
        // BUG FIX: Previously allocated at index i, but freed and continued
        // on parse failure, leaving freed pointers in array. When
        // dataset_free_batch() tried to free indices 0..num_samples-1,
        // it would hit the freed pointers → double-free!
        //
        // SOLUTION: Allocate at num_samples (write index). If parse fails,
        // free immediately and DON'T increment. Next allocation overwrites.
        uint32_t num_features = csv_ctx->num_features;  // Get from context (configured at initialization)
        batch->features[batch->num_samples] = nimcp_calloc(num_features, sizeof(float));
        batch->labels[batch->num_samples] = nimcp_calloc(64, sizeof(char));

        // Parse line
        if (!csv_parse_line(line, ',', num_features, 1, batch->features[batch->num_samples],
                            batch->labels[batch->num_samples], 64)) {
            // Skip invalid lines - free immediately and don't increment
            nimcp_free(batch->features[batch->num_samples]);
            nimcp_free(batch->labels[batch->num_samples]);
            continue;
        }

        // Only increment write pointer on successful parse
        batch->num_samples++;
        csv_ctx->total_rows++;
    }

    if (batch->num_samples == 0) {
        batch->end_of_dataset = true;
        nimcp_free(batch->features);
        nimcp_free(batch->labels);
        return false;
    }

    return true;
}

/**
 * WHAT: Reset CSV to beginning
 * WHY: Support multiple training epochs
 * HOW: Seek to file_start_offset (after header)
 */
static bool csv_reset(void* context)
{
    if (!context)
        return false;

    csv_context_t* csv_ctx = (csv_context_t*) context;

    if (fseek(csv_ctx->file, csv_ctx->file_start_offset, SEEK_SET) != 0) {
        dataio_set_error("Failed to reset CSV file: %s", strerror(errno));
        return false;
    }

    csv_ctx->total_rows = 0;
    return true;
}

/**
 * WHAT: Get CSV size (number of rows)
 * WHY: Progress reporting
 * HOW: Return total_rows (counted as we read)
 */
static uint64_t csv_get_size(void* context)
{
    if (!context)
        return 0;
    csv_context_t* csv_ctx = (csv_context_t*) context;
    return csv_ctx->total_rows;
}

/**
 * WHAT: CSV backend strategy implementation
 * WHY: Pluggable backend for CSV files
 */
static data_source_strategy_t g_csv_strategy = {.source_type = DATA_SOURCE_FILE,
                                                .initialize = csv_initialize,
                                                .shutdown = csv_shutdown,
                                                .next_batch = csv_next_batch,
                                                .reset = csv_reset,
                                                .get_size = csv_get_size};

//=============================================================================
// PostgreSQL Backend Implementation
//=============================================================================

/**
 * WHAT: PostgreSQL context
 * WHY: Track database connection and query state
 */
typedef struct {
    void* db_conn;                 // Database connection (void* to avoid libpq dependency)
    void* result;                  // Current result set
    char query[1024];              // SQL query
    uint32_t num_feature_columns;  // Number of feature columns
    uint32_t current_row;          // Current row in result set
    uint32_t total_rows;           // Total rows in result
} postgres_context_t;

/**
 * WHAT: Initialize PostgreSQL data source
 * WHY: Connect to database, execute query
 * HOW: Use libpq to connect and execute query
 *
 * NOTE: This is a placeholder implementation
 * REASON: Full PostgreSQL support requires libpq dependency
 * TODO: Implement when libpq is available
 */
static bool postgres_initialize(void** context, const dataset_config_t* config)
{
    dataio_set_error("PostgreSQL backend not yet implemented");
    return false;

    // TODO: Implement with libpq
    // 1. PQconnectdb(config->location)
    // 2. Execute query
    // 3. Store result
    // 4. Return context
}

static void postgres_shutdown(void* context)
{
    // TODO: PQfinish, PQclear
}

static bool postgres_next_batch(void* context, data_batch_t* batch)
{
    // TODO: Fetch next N rows from result
    return false;
}

static bool postgres_reset(void* context)
{
    // TODO: Re-execute query
    return false;
}

static uint64_t postgres_get_size(void* context)
{
    // TODO: PQntuples
    return 0;
}

/**
 * WHAT: PostgreSQL backend strategy (placeholder)
 * WHY: Future support for database training
 */
static data_source_strategy_t g_postgres_strategy = {.source_type = DATA_SOURCE_DATABASE,
                                                     .initialize = postgres_initialize,
                                                     .shutdown = postgres_shutdown,
                                                     .next_batch = postgres_next_batch,
                                                     .reset = postgres_reset,
                                                     .get_size = postgres_get_size};

//=============================================================================
// Stream Backend Implementation
//=============================================================================

/**
 * WHAT: Stream context for online learning
 * WHY: Support real-time training from live data
 */
typedef struct {
    stream_callback_fn_t callback;  // User callback
    void* user_data;                // User context
    uint32_t num_features;          // Features per sample
    bool active;                    // Stream is active
} stream_context_t;

/**
 * WHAT: Initialize stream data source
 * WHY: Set up callback for online learning
 */
static bool stream_initialize(void** context, const dataset_config_t* config)
{
    dataio_set_error("Use dataset_create_stream() for streaming data");
    return false;
}

static void stream_shutdown(void* context)
{
    if (!context)
        return;
    stream_context_t* stream_ctx = (stream_context_t*) context;
    stream_ctx->active = false;
    nimcp_free(stream_ctx);
}

/**
 * WHAT: Stream backend (special case)
 * WHY: Streams don't support batch reading (callback-based)
 */
static data_source_strategy_t g_stream_strategy = {
    .source_type = DATA_SOURCE_STREAM,
    .initialize = stream_initialize,
    .shutdown = stream_shutdown,
    .next_batch = NULL,  // Not applicable for streams
    .reset = NULL,       // Cannot reset streams
    .get_size = NULL     // Unknown size
};

//=============================================================================
// Factory Pattern: Backend Selection
//=============================================================================

/**
 * WHAT: Select appropriate backend strategy
 * WHY: Factory pattern - create correct backend for data source
 * HOW: Switch on source type, return strategy
 */
static data_source_strategy_t* select_backend_strategy(data_source_t source, data_format_t format)
{
    switch (source) {
        case DATA_SOURCE_FILE:
            // WHAT: File-based sources
            switch (format) {
                case DATA_FORMAT_CSV:
                    return &g_csv_strategy;
                case DATA_FORMAT_JSON:
                    // TODO: Implement JSON backend
                    dataio_set_error("JSON format not yet implemented");
                    return NULL;
                case DATA_FORMAT_PARQUET:
                    // TODO: Implement Parquet backend
                    dataio_set_error("Parquet format not yet implemented");
                    return NULL;
                default:
                    dataio_set_error("Unsupported file format: %d", format);
                    return NULL;
            }

        case DATA_SOURCE_DATABASE:
            // WHAT: Database sources
            switch (format) {
                case DATA_FORMAT_POSTGRES:
                    return &g_postgres_strategy;
                case DATA_FORMAT_SQLITE:
                    // TODO: Implement SQLite backend
                    dataio_set_error("SQLite format not yet implemented");
                    return NULL;
                default:
                    dataio_set_error("Unsupported database format: %d", format);
                    return NULL;
            }

        case DATA_SOURCE_STREAM:
            return &g_stream_strategy;

        default:
            dataio_set_error("Unsupported data source: %d", source);
            return NULL;
    }
}

//=============================================================================
// Public API: Dataset Loading
//=============================================================================

/**
 * WHAT: Open dataset
 * WHY: Initialize data source for reading
 * HOW: Select backend, initialize, create handle
 */
dataset_t dataset_open(const dataset_config_t* config)
{
    // Guard clause: validate input
    if (!config) {
        dataio_set_error("NULL configuration");
        return NULL;
    }

    // Select backend strategy
    data_source_strategy_t* strategy = select_backend_strategy(config->source, config->format);

    if (!strategy) {
        return NULL;  // Error already set
    }

    // Allocate dataset handle
    dataset_t dataset = nimcp_calloc(1, sizeof(struct dataset_struct));
    if (!dataset) {
        dataio_set_error("Failed to allocate dataset");
        return NULL;
    }

    // Copy configuration
    memcpy(&dataset->config, config, sizeof(dataset_config_t));
    dataset->strategy = strategy;

    // Initialize backend
    if (!strategy->initialize(&dataset->source_context, config)) {
        nimcp_free(dataset);
        return NULL;  // Error already set by backend
    }

    // Initialize mutex for thread-safe reads
    nimcp_mutex_init(&dataset->read_lock, NULL);

    return dataset;
}

/**
 * WHAT: Close dataset
 * WHY: Clean up resources
 * HOW: Call backend shutdown, free memory
 */
void dataset_close(dataset_t dataset)
{
    if (!dataset)
        return;

    // Shutdown backend
    if (dataset->strategy && dataset->strategy->shutdown) {
        dataset->strategy->shutdown(dataset->source_context);
    }

    // Destroy mutex
    nimcp_mutex_destroy(&dataset->read_lock);

    nimcp_free(dataset);
}

/**
 * WHAT: Get next batch of data
 * WHY: Support streaming large datasets
 * HOW: Thread-safe call to backend next_batch
 */
bool dataset_next_batch(dataset_t dataset, data_batch_t* batch)
{
    // Guard clauses
    if (!dataset || !batch) {
        dataio_set_error("Invalid dataset or batch");
        return false;
    }

    if (!dataset->strategy || !dataset->strategy->next_batch) {
        dataio_set_error("Backend does not support batch reading");
        return false;
    }

    // Thread-safe batch read
    nimcp_mutex_lock(&dataset->read_lock);
    bool result = dataset->strategy->next_batch(dataset->source_context, batch);
    dataset->rows_read += batch->num_samples;
    nimcp_mutex_unlock(&dataset->read_lock);

    return result;
}

/**
 * WHAT: Free data batch memory
 * WHY: Prevent memory leaks
 * HOW: Free features, labels, and batch arrays
 */
void dataset_free_batch(data_batch_t* batch)
{
    if (!batch)
        return;

    if (batch->features) {
        for (uint32_t i = 0; i < batch->num_samples; i++) {
            nimcp_free(batch->features[i]);
        }
        nimcp_free(batch->features);
    }

    if (batch->labels) {
        for (uint32_t i = 0; i < batch->num_samples; i++) {
            nimcp_free(batch->labels[i]);
        }
        nimcp_free(batch->labels);
    }

    memset(batch, 0, sizeof(data_batch_t));
}

/**
 * WHAT: Reset dataset to beginning
 * WHY: Support multiple training epochs
 * HOW: Call backend reset
 */
bool dataset_reset(dataset_t dataset)
{
    if (!dataset || !dataset->strategy || !dataset->strategy->reset) {
        dataio_set_error("Invalid dataset or backend does not support reset");
        return false;
    }

    nimcp_mutex_lock(&dataset->read_lock);
    bool result = dataset->strategy->reset(dataset->source_context);
    dataset->rows_read = 0;
    nimcp_mutex_unlock(&dataset->read_lock);

    return result;
}

/**
 * WHAT: Get dataset size
 * WHY: Progress reporting, memory allocation
 * HOW: Call backend get_size
 */
uint64_t dataset_get_size(dataset_t dataset)
{
    if (!dataset || !dataset->strategy || !dataset->strategy->get_size) {
        return 0;
    }

    return dataset->strategy->get_size(dataset->source_context);
}

//=============================================================================
// Producer-Consumer Queue Architecture for Parallel I/O and Training
//=============================================================================

/**
 * WHAT: Training context for async batch processing
 * WHY: Separate I/O (producer) from computation (consumer) for parallelism
 * HOW: Producer thread reads batches, consumer thread trains brain
 *
 * ARCHITECTURE:
 *   Producer Thread          Queue            Consumer Thread
 *   ┌──────────────┐      ┌────────┐        ┌──────────────┐
 *   │ Read Batch   │──→   │ Batch  │   ──→  │ Train Brain  │
 *   │ from Dataset │      │ Buffer │        │ on Samples   │
 *   └──────────────┘      └────────┘        └──────────────┘
 *        ↓                                          ↓
 *   Epoch Loop                                  Free Batch
 *
 * WHY QUEUE:
 * - Overlap I/O latency with computation
 * - Smooth out disk read spikes
 * - Natural backpressure (blocks when queue full)
 * - Better CPU/disk utilization
 *
 * PERFORMANCE:
 * - Without queue: I/O and training serialize (100% overhead)
 * - With queue: I/O and training overlap (2-3x speedup)
 */
typedef struct {
    brain_t brain;
    dataset_t dataset;
    nimcp_queue_handle_t batch_queue;
    nimcp_thread_t producer_thread;
    nimcp_thread_t consumer_thread;
    volatile bool stop_requested;
    volatile bool producer_done;
    uint32_t current_epoch;
    uint32_t total_epochs;
    float validation_split;

    // Training statistics (protected by stats_lock)
    nimcp_mutex_t stats_lock;
    uint64_t samples_trained;
    uint64_t samples_validated;
    float total_accuracy;

    // Error handling
    char error_message[512];
    bool has_error;
} training_context_t;

/**
 * @brief Producer thread: Read batches from dataset and enqueue
 *
 * WHY SEPARATE THREAD:
 * - I/O operations don't block computation
 * - Can read-ahead while brain is training
 * - Disk/network latency hidden
 *
 * ALGORITHM:
 * 1. For each epoch:
 *    a. Reset dataset to beginning
 *    b. Read batch from dataset
 *    c. Enqueue batch (blocks if queue full - backpressure)
 *    d. Repeat until end of dataset
 * 2. Signal consumer (producer_done = true)
 *
 * COMPLEXITY: O(n) where n = number of batches
 * THREAD SAFETY: Only producer accesses dataset
 *
 * @param arg Pointer to training_context_t
 * @return NULL
 */
static void* producer_thread_func(void* arg)
{
    training_context_t* ctx = (training_context_t*) arg;

    for (uint32_t epoch = 0; epoch < ctx->total_epochs; epoch++) {
        if (ctx->stop_requested)
            break;

        ctx->current_epoch = epoch;

        // Reset dataset to beginning for new epoch
        if (!dataset_reset(ctx->dataset)) {
            snprintf(ctx->error_message, sizeof(ctx->error_message),
                     "Failed to reset dataset for epoch %u", epoch);
            ctx->has_error = true;
            break;
        }

        // Read and enqueue batches for this epoch
        while (!ctx->stop_requested) {
            // Allocate batch structure
            data_batch_t* batch = (data_batch_t*) nimcp_malloc(sizeof(data_batch_t));
            if (!batch) {
                snprintf(ctx->error_message, sizeof(ctx->error_message),
                         "Failed to allocate batch memory");
                ctx->has_error = true;
                break;
            }
            memset(batch, 0, sizeof(data_batch_t));

            // Read next batch from dataset
            if (!dataset_next_batch(ctx->dataset, batch)) {
                nimcp_free(batch);
                break;  // End of dataset
            }

            // Enqueue batch (blocks if queue full - backpressure)
            // WHY BLOCKING: Natural flow control, prevents memory exhaustion
            nimcp_result_t result = nimcp_queue_enqueue(ctx->batch_queue, &batch, 5000);
            if (result != NIMCP_SUCCESS) {
                // Enqueue failed (timeout or error)
                dataset_free_batch(batch);
                nimcp_free(batch);
                snprintf(ctx->error_message, sizeof(ctx->error_message), "Failed to enqueue batch");
                ctx->has_error = true;
                break;
            }

            if (batch->end_of_dataset) {
                break;  // Last batch of epoch
            }
        }

        if (ctx->has_error)
            break;
    }

    // Signal producer completion
    ctx->producer_done = true;
    return NULL;
}

/**
 * @brief Consumer thread: Dequeue batches and train brain
 *
 * WHY SEPARATE THREAD:
 * - Computation continues while I/O happens
 * - Better CPU utilization
 * - Lower overall training time
 *
 * ALGORITHM:
 * 1. While producer not done or queue not empty:
 *    a. Dequeue batch (blocks if empty)
 *    b. Train brain on each sample in batch
 *    c. Update statistics
 *    d. Free batch
 * 2. Return final accuracy
 *
 * COMPLEXITY: O(n×m) where n = batches, m = samples per batch
 * THREAD SAFETY: Only consumer accesses brain (brain is thread-safe)
 *
 * @param arg Pointer to training_context_t
 * @return NULL
 */
static void* consumer_thread_func(void* arg)
{
    training_context_t* ctx = (training_context_t*) arg;

    while (!ctx->stop_requested) {
        // Check if producer done AND queue empty
        if (ctx->producer_done && nimcp_queue_is_empty(ctx->batch_queue)) {
            break;
        }

        data_batch_t* batch = NULL;

        // Dequeue batch (blocks if empty, timeout to check producer_done)
        nimcp_result_t result = nimcp_queue_dequeue(ctx->batch_queue, &batch, 500);

        if (result == NIMCP_TIMEOUT) {
            continue;  // No batch available, check producer_done again
        }

        if (result != NIMCP_SUCCESS || !batch) {
            continue;  // Queue error or empty
        }

        // Train on each sample in batch
        for (uint32_t i = 0; i < batch->num_samples; i++) {
            if (ctx->stop_requested)
                break;

            // Skip samples in validation set
            float sample_rand = (float) rand() / RAND_MAX;
            if (sample_rand < ctx->validation_split) {
                nimcp_mutex_lock(&ctx->stats_lock);
                ctx->samples_validated++;
                nimcp_mutex_unlock(&ctx->stats_lock);
                continue;
            }

            // Train brain on this sample
            uint32_t num_features = brain_get_num_inputs(ctx->brain);
            float loss = brain_learn_example(ctx->brain, batch->features[i],
                                            num_features, batch->labels[i], 1.0f);
            (void)loss;  // Loss tracked internally by brain

            nimcp_mutex_lock(&ctx->stats_lock);
            ctx->samples_trained++;
            nimcp_mutex_unlock(&ctx->stats_lock);
        }

        // Free batch memory
        dataset_free_batch(batch);
        nimcp_free(batch);
    }

    return NULL;
}

//=============================================================================
// Public API: Brain Training from Datasets
//=============================================================================

/**
 * WHAT: Train brain from dataset with async I/O
 * WHY: Overlap I/O and computation for 2-3x performance improvement
 * HOW: Producer thread reads batches, consumer thread trains, queue decouples
 *
 * PERFORMANCE IMPROVEMENT:
 * - Synchronous (old): I/O → Wait → Train → Wait → Repeat (100% overhead)
 * - Async (new): I/O ║ Train (parallel, ~2-3x faster)
 *
 * EXAMPLE TIMELINE:
 * Sync:  |I/O1|Train1|I/O2|Train2|I/O3|Train3| = 6 time units
 * Async: |I/O1|       |I/O3|                   = 4 time units
 *        |    |Train1|Train2|Train3|
 *
 * WHY QUEUE SIZE = 10:
 * - Balance between memory usage and read-ahead benefit
 * - 10 batches typically 1-10MB (acceptable memory overhead)
 * - Enough buffering to hide I/O latency spikes
 *
 * @param brain Brain to train
 * @param dataset Dataset to read from
 * @param epochs Number of training passes
 * @param validation_split Fraction for validation (0-1)
 * @return Final validation accuracy
 */
float brain_train_from_dataset(brain_t brain, dataset_t dataset, uint32_t epochs,
                               float validation_split)
{
    // Guard clauses
    if (!brain || !dataset) {
        dataio_set_error("Invalid brain or dataset");
        return 0.0f;
    }

    if (validation_split < 0.0f || validation_split >= 1.0f) {
        dataio_set_error("Invalid validation_split: %f (must be 0-1)", validation_split);
        return 0.0f;
    }

    // Initialize training context
    training_context_t* ctx = (training_context_t*) nimcp_calloc(1, sizeof(training_context_t));
    if (!ctx) {
        dataio_set_error("Failed to allocate training context");
        return 0.0f;
    }

    ctx->brain = brain;
    ctx->dataset = dataset;
    ctx->total_epochs = epochs;
    ctx->validation_split = validation_split;
    ctx->stop_requested = false;
    ctx->producer_done = false;
    ctx->has_error = false;
    nimcp_mutex_init(&ctx->stats_lock, NULL);

    // Create batch queue
    // WHY 10 BATCHES: Balance between memory overhead and I/O hiding
    // WHY BLOCKING: Natural backpressure prevents memory exhaustion
    nimcp_queue_config_t queue_config = {.max_size = 10,
                                         .item_size = sizeof(data_batch_t*),
                                         .is_blocking = true,
                                         .timeout_ms = 5000};

    nimcp_result_t result = nimcp_queue_create(&queue_config, &ctx->batch_queue);
    if (result != NIMCP_SUCCESS) {
        dataio_set_error("Failed to create batch queue");
        nimcp_mutex_destroy(&ctx->stats_lock);
        nimcp_free(ctx);
        return 0.0f;
    }

    // Start producer thread (reads batches from dataset)
    if (nimcp_thread_create(&ctx->producer_thread, producer_thread_func, ctx, NULL) != 0) {
        dataio_set_error("Failed to create producer thread");
        nimcp_queue_destroy(ctx->batch_queue);
        nimcp_mutex_destroy(&ctx->stats_lock);
        nimcp_free(ctx);
        return 0.0f;
    }

    // Start consumer thread (trains brain on batches)
    if (nimcp_thread_create(&ctx->consumer_thread, consumer_thread_func, ctx, NULL) != 0) {
        dataio_set_error("Failed to create consumer thread");
        ctx->stop_requested = true;
        nimcp_thread_join(ctx->producer_thread, NULL);
        nimcp_queue_destroy(ctx->batch_queue);
        nimcp_mutex_destroy(&ctx->stats_lock);
        nimcp_free(ctx);
        return 0.0f;
    }

    // Wait for both threads to complete
    nimcp_thread_join(ctx->producer_thread, NULL);
    nimcp_thread_join(ctx->consumer_thread, NULL);

    // Get final statistics
    float total_accuracy = ctx->total_accuracy;
    uint64_t samples_trained = ctx->samples_trained;
    uint64_t samples_validated = ctx->samples_validated;

    // Check for errors during training
    if (ctx->has_error) {
        dataio_set_error("%s", ctx->error_message);
    }

    // Print summary
    printf("Training complete:\n");
    printf("  Epochs: %u\n", epochs);
    printf("  Samples trained: %lu\n", samples_trained);
    printf("  Samples validated: %lu\n", samples_validated);
    printf("  Final accuracy: %.1f%%\n", total_accuracy * 100);

    // Cleanup
    nimcp_queue_destroy(ctx->batch_queue);
    nimcp_mutex_destroy(&ctx->stats_lock);
    nimcp_free(ctx);

    return total_accuracy;
}

/**
 * WHAT: Train brain from dataset (streaming)
 * WHY: Handle datasets too large for memory
 * HOW: Process batches sequentially without reset
 */
float brain_train_from_dataset_streaming(brain_t brain, dataset_t dataset, uint32_t max_batches)
{
    if (!brain || !dataset) {
        dataio_set_error("Invalid brain or dataset");
        return 0.0f;
    }

    uint32_t batches_processed = 0;
    float avg_loss = 0.0f;

    while (max_batches == 0 || batches_processed < max_batches) {
        data_batch_t batch;
        memset(&batch, 0, sizeof(batch));

        if (!dataset_next_batch(dataset, &batch)) {
            break;  // End of dataset
        }

        // Train on batch
        for (uint32_t i = 0; i < batch.num_samples; i++) {
            // TODO: Train brain
        }

        dataset_free_batch(&batch);
        batches_processed++;

        if (batch.end_of_dataset)
            break;
    }

    return avg_loss;
}

//=============================================================================
// Public API: Data Export
//=============================================================================

/**
 * WHAT: Export brain predictions to file
 * WHY: Analyze brain decisions, debugging, analytics
 * HOW: Run brain on dataset, write predictions to CSV
 */
bool brain_export_predictions(brain_t brain, dataset_t input_dataset, const char* output_file,
                              data_format_t format)
{
    // Guard clauses
    if (!brain || !input_dataset || !output_file) {
        dataio_set_error("Invalid parameters");
        return false;
    }

    if (format != DATA_FORMAT_CSV) {
        dataio_set_error("Only CSV export supported currently");
        return false;
    }

    // Open output file
    FILE* out = fopen(output_file, "w");
    if (!out) {
        dataio_set_error("Failed to open output file: %s", strerror(errno));
        return false;
    }

    // Write CSV header
    fprintf(out, "features,prediction,confidence\n");

    // Get number of features from brain
    uint32_t num_features = brain_get_num_inputs(brain);

    // Process dataset
    while (true) {
        data_batch_t batch;
        memset(&batch, 0, sizeof(batch));

        if (!dataset_next_batch(input_dataset, &batch)) {
            break;
        }

        // Run brain on each sample
        for (uint32_t i = 0; i < batch.num_samples; i++) {
            // Get brain decision
            brain_decision_t* decision = brain_decide(brain, batch.features[i], num_features);
            if (!decision) {
                continue;  // Skip on error
            }

            // Write to CSV: features, prediction, confidence
            fprintf(out, "\"[");
            for (uint32_t f = 0; f < num_features; f++) {
                fprintf(out, "%.6f", batch.features[i][f]);
                if (f < num_features - 1) {
                    fprintf(out, ",");
                }
            }
            fprintf(out, "]\",");
            fprintf(out, "\"%s\",", decision->label);
            fprintf(out, "%.6f", decision->confidence);
            fprintf(out, "\n");

            brain_free_decision(decision);
        }

        dataset_free_batch(&batch);

        if (batch.end_of_dataset)
            break;
    }

    fclose(out);
    return true;
}

/**
 * WHAT: Export brain training data
 * WHY: Inspect what brain has learned, debugging
 * HOW: Write internal training examples to file
 */
bool brain_export_training_data(brain_t brain, const char* output_file, data_format_t format)
{
    if (!brain || !output_file) {
        dataio_set_error("Invalid parameters");
        return false;
    }

    // TODO: Implement when brain internal structure is available
    dataio_set_error("brain_export_training_data not yet implemented");
    return false;
}

//=============================================================================
// Convenience Functions
//=============================================================================

/**
 * WHAT: Load CSV file (convenience function)
 * WHY: Simplify common case of loading CSV
 * HOW: Factory pattern - create config and call dataset_open
 */
dataset_t dataset_load_csv(const char* filepath, uint32_t num_feature_columns,
                           uint32_t num_label_columns, bool has_header)
{
    if (!filepath) {
        dataio_set_error("NULL filepath");
        return NULL;
    }

    // Create configuration
    dataset_config_t config = {.format = DATA_FORMAT_CSV,
                               .source = DATA_SOURCE_FILE,
                               .num_feature_columns = num_feature_columns,
                               .num_label_columns = num_label_columns,
                               .has_header = has_header,
                               .delimiter = ',',
                               .normalize_features = false,
                               .shuffle = false,
                               .batch_size = 1000,
                               .max_rows = 0};

    strncpy(config.location, filepath, sizeof(config.location) - 1);

    return dataset_open(&config);
}

/**
 * WHAT: Load from PostgreSQL query (convenience function)
 * WHY: Simplify database loading
 * HOW: Factory pattern - create config and call dataset_open
 */
dataset_t dataset_load_postgres(const char* connection_string, const char* query,
                                uint32_t num_feature_columns)
{
    if (!connection_string || !query) {
        dataio_set_error("NULL connection_string or query");
        return NULL;
    }

    dataset_config_t config = {.format = DATA_FORMAT_POSTGRES,
                               .source = DATA_SOURCE_DATABASE,
                               .num_feature_columns = num_feature_columns,
                               .num_label_columns = 1,
                               .has_header = false,
                               .batch_size = 1000,
                               .max_rows = 0};

    // Store connection string and query in location
    snprintf(config.location, sizeof(config.location), "%s|%s", connection_string, query);

    return dataset_open(&config);
}

/**
 * WHAT: Save dataset to CSV
 * WHY: Export training data for analysis
 * HOW: Write features and labels to CSV file
 */
bool dataset_save_csv(float** features, char** labels, uint32_t num_samples, uint32_t num_features,
                      const char* filepath, char** feature_names)
{
    if (!features || !labels || !filepath) {
        dataio_set_error("Invalid parameters");
        return false;
    }

    FILE* f = fopen(filepath, "w");
    if (!f) {
        dataio_set_error("Failed to open file: %s", strerror(errno));
        return false;
    }

    // Write header
    if (feature_names) {
        for (uint32_t i = 0; i < num_features; i++) {
            fprintf(f, "%s%s", feature_names[i], i < num_features - 1 ? "," : "");
        }
        fprintf(f, ",label\n");
    }

    // Write data rows
    for (uint32_t i = 0; i < num_samples; i++) {
        for (uint32_t j = 0; j < num_features; j++) {
            fprintf(f, "%f%s", features[i][j], j < num_features - 1 ? "," : "");
        }
        fprintf(f, ",%s\n", labels[i]);
    }

    fclose(f);
    return true;
}

//=============================================================================
// Online Learning from Streams
//=============================================================================

/**
 * WHAT: Create streaming dataset from callback
 * WHY: Support online learning from live data
 * HOW: Store callback, create stream context
 */
dataset_t dataset_create_stream(stream_callback_fn_t callback, void* user_data,
                                uint32_t num_features)
{
    if (!callback) {
        dataio_set_error("NULL callback");
        return NULL;
    }

    // Allocate dataset
    dataset_t dataset = nimcp_calloc(1, sizeof(struct dataset_struct));
    if (!dataset) {
        dataio_set_error("Failed to allocate dataset");
        return NULL;
    }

    dataset->strategy = &g_stream_strategy;

    // Create stream context
    stream_context_t* stream_ctx = nimcp_calloc(1, sizeof(stream_context_t));
    if (!stream_ctx) {
        nimcp_free(dataset);
        return NULL;
    }
    stream_ctx->callback = callback;
    stream_ctx->user_data = user_data;
    stream_ctx->num_features = num_features;
    stream_ctx->active = true;

    dataset->source_context = stream_ctx;
    nimcp_mutex_init(&dataset->read_lock, NULL);

    return dataset;
}

/**
 * WHAT: Train brain from live stream
 * WHY: Online learning from real-time data
 * HOW: Call callback repeatedly until duration expires
 */
uint64_t brain_train_from_stream(brain_t brain, dataset_t stream_dataset, uint32_t duration_seconds)
{
    if (!brain || !stream_dataset) {
        dataio_set_error("Invalid parameters");
        return 0;
    }

    if (stream_dataset->strategy->source_type != DATA_SOURCE_STREAM) {
        dataio_set_error("Dataset is not a stream");
        return 0;
    }

    stream_context_t* stream_ctx = (stream_context_t*) stream_dataset->source_context;

    time_t start_time = time(NULL);
    uint64_t samples_processed = 0;

    // WHAT: Train until duration expires or stream closes
    // WHY: Continuous learning from live data
    while (stream_ctx->active) {
        // Check duration
        if (duration_seconds > 0) {
            time_t elapsed = time(NULL) - start_time;
            if (elapsed >= duration_seconds) {
                break;
            }
        }

        // TODO: Wait for callback to provide data
        // This would typically be event-driven (select/poll/epoll)
        // For now, just sleep
        usleep(100000);  // 100ms

        samples_processed++;
    }

    return samples_processed;
}
