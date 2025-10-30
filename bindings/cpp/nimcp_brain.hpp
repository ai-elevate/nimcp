/**
 * @file nimcp_brain.hpp
 * @brief C++ bindings for NIMCP Brain API
 *
 * Modern C++ wrapper with RAII, exceptions, and STL containers.
 *
 * Example usage:
 * @code
 * #include "nimcp_brain.hpp"
 *
 * using namespace nimcp;
 *
 * // Create brain
 * Brain brain("ethics", BrainSize::SMALL, BrainTask::CLASSIFICATION, 4, 3);
 *
 * // Learn
 * std::vector<float> features = {0.8f, 0.3f, 0.5f, 0.6f};
 * brain.learn_example(features, "block", 0.9f);
 *
 * // Decide
 * auto decision = brain.decide(features);
 * std::cout << "Decision: " << decision.label
 *           << " (confidence: " << decision.confidence << ")\n";
 *
 * // Save
 * brain.save("brain.nimcp");
 * @endcode
 */

#ifndef NIMCP_BRAIN_HPP
#define NIMCP_BRAIN_HPP

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <functional>

extern "C" {
    #include "../../src/include/nimcp_brain.h"
}

namespace nimcp {

//=============================================================================
// Enumerations
//=============================================================================

enum class BrainSize {
    TINY = BRAIN_SIZE_TINY,
    SMALL = BRAIN_SIZE_SMALL,
    MEDIUM = BRAIN_SIZE_MEDIUM,
    LARGE = BRAIN_SIZE_LARGE,
    CUSTOM = BRAIN_SIZE_CUSTOM
};

enum class BrainTask {
    CLASSIFICATION = BRAIN_TASK_CLASSIFICATION,
    REGRESSION = BRAIN_TASK_REGRESSION,
    PATTERN_MATCHING = BRAIN_TASK_PATTERN_MATCHING,
    SEQUENCE = BRAIN_TASK_SEQUENCE,
    ASSOCIATION = BRAIN_TASK_ASSOCIATION,
    CUSTOM = BRAIN_TASK_CUSTOM
};

//=============================================================================
// Result Types
//=============================================================================

/**
 * @brief Decision result from brain inference
 */
struct Decision {
    std::string label;
    float confidence;
    std::vector<float> output_vector;
    uint32_t num_active_neurons;
    std::vector<uint32_t> active_neuron_ids;
    float sparsity;
    std::string explanation;
    uint64_t inference_time_us;

    /** Get inference time in milliseconds */
    double inference_time_ms() const {
        return inference_time_us / 1000.0;
    }
};

/**
 * @brief Brain statistics
 */
struct Stats {
    std::string task_name;
    BrainSize size;
    uint32_t num_neurons;
    uint32_t num_synapses;
    uint32_t num_active_synapses;
    uint64_t total_inferences;
    uint64_t total_learning_steps;
    float avg_sparsity;
    float avg_inference_time_us;
    float current_learning_rate;
    float accuracy;
    size_t memory_bytes;

    /** Get average inference time in milliseconds */
    double avg_inference_time_ms() const {
        return avg_inference_time_us / 1000.0;
    }

    /** Get memory usage in megabytes */
    double memory_mb() const {
        return memory_bytes / (1024.0 * 1024.0);
    }
};

/**
 * @brief Neuron importance ranking
 */
struct NeuronImportance {
    uint32_t neuron_id;
    float importance;
};

//=============================================================================
// Exception Types
//=============================================================================

class BrainException : public std::runtime_error {
public:
    explicit BrainException(const std::string& msg)
        : std::runtime_error(msg) {}
};

//=============================================================================
// Brain Class
//=============================================================================

/**
 * @brief High-level C++ interface to NIMCP Brain
 *
 * RAII wrapper around brain_t with modern C++ conveniences.
 */
class Brain {
public:
    /**
     * @brief Create a new brain
     *
     * @param task_name Human-readable name
     * @param size Brain size preset
     * @param task Task type
     * @param num_inputs Number of input features
     * @param num_outputs Number of output classes/values
     *
     * @throws BrainException if creation fails
     */
    Brain(const std::string& task_name,
          BrainSize size,
          BrainTask task,
          uint32_t num_inputs,
          uint32_t num_outputs)
        : handle_(brain_create(task_name.c_str(),
                              static_cast<brain_size_t>(size),
                              static_cast<brain_task_t>(task),
                              num_inputs,
                              num_outputs))
        , task_name_(task_name)
        , size_(size)
        , task_(task)
        , num_inputs_(num_inputs)
        , num_outputs_(num_outputs)
    {
        if (!handle_) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Failed to create brain");
        }
    }

    /**
     * @brief Load brain from file
     *
     * @param filepath Path to brain file
     * @return Loaded Brain instance
     * @throws BrainException if loading fails
     */
    static Brain load(const std::string& filepath) {
        brain_t handle = brain_load(filepath.c_str());
        if (!handle) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Failed to load brain");
        }

        Brain brain;
        brain.handle_ = handle;

        // Populate metadata from stats
        auto stats = brain.get_stats();
        brain.task_name_ = stats.task_name;
        brain.size_ = stats.size;

        return brain;
    }

    // Disable copy, enable move
    Brain(const Brain&) = delete;
    Brain& operator=(const Brain&) = delete;

    Brain(Brain&& other) noexcept
        : handle_(other.handle_)
        , task_name_(std::move(other.task_name_))
        , size_(other.size_)
        , task_(other.task_)
        , num_inputs_(other.num_inputs_)
        , num_outputs_(other.num_outputs_)
    {
        other.handle_ = nullptr;
    }

    Brain& operator=(Brain&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                brain_destroy(handle_);
            }

            handle_ = other.handle_;
            task_name_ = std::move(other.task_name_);
            size_ = other.size_;
            task_ = other.task_;
            num_inputs_ = other.num_inputs_;
            num_outputs_ = other.num_outputs_;

            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Brain() {
        if (handle_) {
            brain_destroy(handle_);
        }
    }

    /**
     * @brief Learn from a single example
     *
     * @param features Input feature vector
     * @param label Target label
     * @param confidence Training weight (0.0-1.0)
     * @return Loss value
     * @throws BrainException if learning fails
     */
    float learn_example(const std::vector<float>& features,
                       const std::string& label,
                       float confidence = 1.0f) {
        if (features.size() != num_inputs_) {
            throw BrainException("Feature count mismatch");
        }

        float loss = brain_learn_example(handle_,
                                         features.data(),
                                         features.size(),
                                         label.c_str(),
                                         confidence);

        if (loss < 0) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Learning failed");
        }

        return loss;
    }

    /**
     * @brief Learn from batch of examples
     *
     * @param examples Vector of (features, label, confidence) tuples
     * @return Average loss
     */
    float learn_batch(const std::vector<std::tuple<std::vector<float>, std::string, float>>& examples) {
        float total_loss = 0.0f;

        for (const auto& [features, label, confidence] : examples) {
            total_loss += learn_example(features, label, confidence);
        }

        return total_loss / examples.size();
    }

    /**
     * @brief Learn from LLM teacher
     *
     * @param features Input features
     * @param teacher_fn Teacher function callback
     * @return Loss value
     */
    using TeacherFunction = std::function<std::pair<std::string, float>(const std::vector<float>&)>;

    float learn_from_llm(const std::vector<float>& features,
                        TeacherFunction teacher_fn) {
        // Wrapper to convert C++ function to C callback
        static TeacherFunction* current_teacher = nullptr;
        current_teacher = &teacher_fn;

        auto c_wrapper = [](const float* input, uint32_t num_features,
                           void* context, char* output_label, uint32_t max_label_len) -> float {
            (void)context;

            std::vector<float> features(input, input + num_features);
            auto [label, confidence] = (*current_teacher)(features);

            strncpy(output_label, label.c_str(), max_label_len - 1);
            output_label[max_label_len - 1] = '\0';

            return confidence;
        };

        float loss = brain_learn_from_llm(handle_,
                                         features.data(),
                                         features.size(),
                                         c_wrapper,
                                         nullptr);

        if (loss < 0) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "LLM learning failed");
        }

        return loss;
    }

    /**
     * @brief Make decision for input
     *
     * @param features Input feature vector
     * @return Decision result
     * @throws BrainException if inference fails
     */
    Decision decide(const std::vector<float>& features) {
        if (features.size() != num_inputs_) {
            throw BrainException("Feature count mismatch");
        }

        brain_decision_t* c_decision = brain_decide(handle_,
                                                    features.data(),
                                                    features.size());

        if (!c_decision) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Decision failed");
        }

        // Convert to C++ Decision
        Decision decision;
        decision.label = c_decision->label;
        decision.confidence = c_decision->confidence;
        decision.output_vector.assign(c_decision->output_vector,
                                     c_decision->output_vector + c_decision->output_size);
        decision.num_active_neurons = c_decision->num_active_neurons;
        decision.active_neuron_ids.assign(c_decision->active_neuron_ids,
                                         c_decision->active_neuron_ids + c_decision->num_active_neurons);
        decision.sparsity = c_decision->sparsity;
        decision.explanation = c_decision->explanation;
        decision.inference_time_us = c_decision->inference_time_us;

        brain_free_decision(c_decision);

        return decision;
    }

    /**
     * @brief Save brain to file
     *
     * @param filepath Path to save to
     * @throws BrainException if save fails
     */
    void save(const std::string& filepath) {
        if (!brain_save(handle_, filepath.c_str())) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Save failed");
        }
    }

    /**
     * @brief Get brain statistics
     *
     * @return Stats object
     */
    Stats get_stats() const {
        brain_stats_t c_stats;
        if (!brain_get_stats(handle_, &c_stats)) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Get stats failed");
        }

        Stats stats;
        stats.task_name = c_stats.task_name;
        stats.size = static_cast<BrainSize>(c_stats.size);
        stats.num_neurons = c_stats.num_neurons;
        stats.num_synapses = c_stats.num_synapses;
        stats.num_active_synapses = c_stats.num_active_synapses;
        stats.total_inferences = c_stats.total_inferences;
        stats.total_learning_steps = c_stats.total_learning_steps;
        stats.avg_sparsity = c_stats.avg_sparsity;
        stats.avg_inference_time_us = c_stats.avg_inference_time_us;
        stats.current_learning_rate = c_stats.current_learning_rate;
        stats.accuracy = c_stats.accuracy;
        stats.memory_bytes = c_stats.memory_bytes;

        return stats;
    }

    /**
     * @brief Get top contributing neurons
     *
     * @param top_n Number of top neurons to return
     * @return Vector of (neuron_id, importance) pairs
     */
    std::vector<NeuronImportance> get_top_neurons(uint32_t top_n = 10) {
        std::vector<uint32_t> neuron_ids(top_n);
        std::vector<float> importances(top_n);

        uint32_t count = brain_get_top_neurons(handle_, top_n,
                                               neuron_ids.data(),
                                               importances.data());

        std::vector<NeuronImportance> result;
        result.reserve(count);

        for (uint32_t i = 0; i < count; i++) {
            result.push_back({neuron_ids[i], importances[i]});
        }

        return result;
    }

    /**
     * @brief Prune weak connections
     *
     * @param threshold Prune synapses with weight < threshold
     * @return Number of synapses pruned
     */
    uint32_t prune(float threshold = 0.01f) {
        return brain_prune(handle_, threshold);
    }

    /**
     * @brief Optimize brain for inference
     */
    void optimize_for_inference() {
        if (!brain_optimize_for_inference(handle_)) {
            const char* error = brain_get_last_error();
            throw BrainException(error ? error : "Optimization failed");
        }
    }

    // Getters
    const std::string& task_name() const { return task_name_; }
    BrainSize size() const { return size_; }
    BrainTask task() const { return task_; }
    uint32_t num_inputs() const { return num_inputs_; }
    uint32_t num_outputs() const { return num_outputs_; }

private:
    // Private constructor for load()
    Brain() : handle_(nullptr) {}

    brain_t handle_;
    std::string task_name_;
    BrainSize size_;
    BrainTask task_;
    uint32_t num_inputs_;
    uint32_t num_outputs_;
};

//=============================================================================
// Convenience Functions
//=============================================================================

/**
 * @brief Create a classification brain with sensible defaults
 */
inline Brain create_classifier(const std::string& name,
                              uint32_t num_inputs,
                              uint32_t num_outputs,
                              BrainSize size = BrainSize::SMALL) {
    return Brain(name, size, BrainTask::CLASSIFICATION, num_inputs, num_outputs);
}

/**
 * @brief Create a pattern matching brain
 */
inline Brain create_pattern_matcher(const std::string& name,
                                   uint32_t num_inputs,
                                   BrainSize size = BrainSize::SMALL) {
    return Brain(name, size, BrainTask::PATTERN_MATCHING, num_inputs, 1);
}

} // namespace nimcp

#endif // NIMCP_BRAIN_HPP
