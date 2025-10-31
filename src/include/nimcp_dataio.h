/**
 * @file nimcp_dataio.h
 * @brief Brain data I/O for external datasets
 *
 * WHAT: Read/write training data from various sources
 * WHY: Brains need to learn from external datasets (CSV, JSON, databases)
 * HOW: Pluggable data loaders with streaming support
 */

#ifndef NIMCP_DATAIO_H
#define NIMCP_DATAIO_H

#include "nimcp_brain.h"
#include "nimcp_export.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/**
 * @brief Data format types
 */
typedef enum {
    DATA_FORMAT_CSV,          /**< Comma-separated values */
    DATA_FORMAT_JSON,         /**< JSON array of objects */
    DATA_FORMAT_SQLITE,       /**< SQLite database */
    DATA_FORMAT_POSTGRES,     /**< PostgreSQL database */
    DATA_FORMAT_PARQUET,      /**< Apache Parquet (columnar) */
    DATA_FORMAT_CUSTOM        /**< Custom format */
} data_format_t;

/**
 * @brief Data source types
 */
typedef enum {
    DATA_SOURCE_FILE,         /**< Local file */
    DATA_SOURCE_DATABASE,     /**< Database connection */
    DATA_SOURCE_HTTP,         /**< HTTP/HTTPS URL */
    DATA_SOURCE_S3,           /**< AWS S3 bucket */
    DATA_SOURCE_STREAM        /**< Stream (stdin, socket, etc.) */
} data_source_t;

/**
 * @brief Dataset configuration
 */
typedef struct {
    data_format_t format;
    data_source_t source;

    // Source location
    char location[512];           /**< File path, URL, connection string */

    // Schema
    uint32_t num_feature_columns; /**< Number of feature columns */
    uint32_t num_label_columns;   /**< Number of label columns */
    char** feature_names;         /**< Feature column names */
    char** label_names;           /**< Label column names */

    // Options
    bool has_header;              /**< First row is header */
    char delimiter;               /**< CSV delimiter (default ',') */
    bool normalize_features;      /**< Auto-normalize features to 0-1 */
    bool shuffle;                 /**< Shuffle rows during reading */

    // Streaming
    uint32_t batch_size;          /**< Read this many rows at a time */
    uint32_t max_rows;            /**< Max rows to read (0 = all) */
} dataset_config_t;

/**
 * @brief Opaque dataset handle
 */
typedef struct dataset_struct* dataset_t;

/**
 * @brief Data batch (for streaming)
 */
typedef struct {
    float** features;             /**< Array of feature vectors */
    char** labels;                /**< Array of labels */
    uint32_t num_samples;         /**< Number of samples in batch */
    bool end_of_dataset;          /**< No more data available */
} data_batch_t;

//=============================================================================
// Dataset Loading API
//=============================================================================

/**
 * @brief Open dataset
 *
 * @param config Dataset configuration
 * @return Dataset handle or NULL on error
 */
dataset_t dataset_open(const dataset_config_t* config);

/**
 * @brief Close dataset
 *
 * @param dataset Dataset handle
 */
void dataset_close(dataset_t dataset);

/**
 * @brief Get next batch of data
 *
 * For streaming reading of large datasets
 *
 * @param dataset Dataset handle
 * @param batch Output batch (caller must free with dataset_free_batch)
 * @return true if data was read
 */
bool dataset_next_batch(dataset_t dataset, data_batch_t* batch);

/**
 * @brief Free data batch
 *
 * @param batch Batch to free
 */
void dataset_free_batch(data_batch_t* batch);

/**
 * @brief Reset dataset to beginning
 *
 * @param dataset Dataset handle
 * @return true on success
 */
bool dataset_reset(dataset_t dataset);

/**
 * @brief Get dataset size
 *
 * @param dataset Dataset handle
 * @return Number of rows in dataset (or 0 if unknown)
 */
uint64_t dataset_get_size(dataset_t dataset);

//=============================================================================
// Brain Training from Data API
//=============================================================================

/**
 * @brief Train brain from dataset
 *
 * Reads entire dataset and trains brain
 *
 * @param brain Brain to train
 * @param dataset Dataset to train from
 * @param epochs Number of passes through data
 * @param validation_split Fraction of data to use for validation (0-1)
 * @return Final validation accuracy
 */
float brain_train_from_dataset(brain_t brain,
                               dataset_t dataset,
                               uint32_t epochs,
                               float validation_split);

/**
 * @brief Train brain from dataset (streaming)
 *
 * For datasets too large to fit in memory
 *
 * @param brain Brain to train
 * @param dataset Dataset to train from
 * @param max_batches Maximum batches to process (0 = all)
 * @return Average loss
 */
float brain_train_from_dataset_streaming(brain_t brain,
                                         dataset_t dataset,
                                         uint32_t max_batches);

//=============================================================================
// Data Export API
//=============================================================================

/**
 * @brief Export brain decisions to file
 *
 * Runs brain on dataset and writes predictions to file
 *
 * @param brain Brain to use
 * @param input_dataset Input data
 * @param output_file Output file path
 * @param format Output format
 * @return true on success
 */
bool brain_export_predictions(brain_t brain,
                              dataset_t input_dataset,
                              const char* output_file,
                              data_format_t format);

/**
 * @brief Export brain training data
 *
 * Writes brain's internal training examples to file
 *
 * @param brain Brain handle
 * @param output_file Output file path
 * @param format Output format
 * @return true on success
 */
bool brain_export_training_data(brain_t brain,
                                const char* output_file,
                                data_format_t format);

//=============================================================================
// Convenience Functions for Common Formats
//=============================================================================

/**
 * @brief Load CSV file
 *
 * Simple CSV loading with automatic schema detection
 *
 * @param filepath CSV file path
 * @param num_feature_columns Number of feature columns
 * @param num_label_columns Number of label columns
 * @param has_header True if first row is header
 * @return Dataset handle or NULL on error
 */
dataset_t dataset_load_csv(const char* filepath,
                           uint32_t num_feature_columns,
                           uint32_t num_label_columns,
                           bool has_header);

/**
 * @brief Load from PostgreSQL query
 *
 * @param connection_string PostgreSQL connection string
 * @param query SQL query returning features and labels
 * @param num_feature_columns Number of feature columns in result
 * @return Dataset handle or NULL on error
 */
dataset_t dataset_load_postgres(const char* connection_string,
                                const char* query,
                                uint32_t num_feature_columns);

/**
 * @brief Save CSV file
 *
 * @param features Feature vectors
 * @param labels Labels
 * @param num_samples Number of samples
 * @param num_features Features per sample
 * @param filepath Output file path
 * @param feature_names Column names (optional)
 * @return true on success
 */
bool dataset_save_csv(float** features,
                     char** labels,
                     uint32_t num_samples,
                     uint32_t num_features,
                     const char* filepath,
                     char** feature_names);

//=============================================================================
// Online Learning from Streams
//=============================================================================

/**
 * @brief Stream callback function
 *
 * Called for each new data point
 *
 * @param features Feature vector
 * @param num_features Feature count
 * @param label Target label
 * @param user_data User context
 */
typedef void (*stream_callback_fn_t)(
    const float* features,
    uint32_t num_features,
    const char* label,
    void* user_data
);

/**
 * @brief Create streaming dataset from callback
 *
 * For online learning from live data streams
 *
 * @param callback Function called for each data point
 * @param user_data Context passed to callback
 * @param num_features Features per sample
 * @return Dataset handle
 */
dataset_t dataset_create_stream(stream_callback_fn_t callback,
                                void* user_data,
                                uint32_t num_features);

/**
 * @brief Train brain from live stream
 *
 * Continuously trains as new data arrives
 *
 * @param brain Brain to train
 * @param stream_dataset Streaming dataset
 * @param duration_seconds How long to train (0 = forever)
 * @return Number of samples processed
 */
uint64_t brain_train_from_stream(brain_t brain,
                                 dataset_t stream_dataset,
                                 uint32_t duration_seconds);

#endif // NIMCP_DATAIO_H
