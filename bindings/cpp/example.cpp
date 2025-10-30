/**
 * @file example.cpp
 * @brief Example usage of NIMCP Brain C++ API
 *
 * Compile:
 *   g++ -std=c++17 example.cpp -I../../src/include -L../../build/src/lib -lnimcp_core -o brain_example
 *
 * Run:
 *   LD_LIBRARY_PATH=../../build/src/lib ./brain_example
 */

#include "nimcp_brain.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace nimcp;

void ethics_demo() {
    std::cout << "==================================================\n";
    std::cout << " NIMCP Brain C++ API Demo\n";
    std::cout << " Use Case: Ethics Decision Caching\n";
    std::cout << "==================================================\n\n";

    // Create brain for ethics decisions
    Brain brain("artemis_ethics", BrainSize::SMALL, BrainTask::CLASSIFICATION,
                4,  // inputs: harm, fairness, transparency, autonomy
                3); // outputs: allow, warn, block

    std::cout << "Created brain: " << brain.task_name() << "\n";
    auto stats = brain.get_stats();
    std::cout << "  Neurons: " << stats.num_neurons << "\n";
    std::cout << "  Memory: " << std::fixed << std::setprecision(2)
              << stats.memory_mb() << " MB\n\n";

    // Training data (simulated LLM decisions)
    std::vector<std::tuple<std::vector<float>, std::string, float>> training_data = {
        {{0.9f, 0.5f, 0.5f, 0.5f}, "block", 0.95f},  // High harm
        {{0.2f, 0.8f, 0.8f, 0.8f}, "allow", 0.90f},  // Safe
        {{0.5f, 0.2f, 0.5f, 0.5f}, "block", 0.85f},  // Unfair
        {{0.3f, 0.6f, 0.1f, 0.7f}, "warn", 0.75f},   // Low transparency
        {{0.1f, 0.7f, 0.8f, 0.9f}, "allow", 0.90f},  // Good
        {{0.8f, 0.3f, 0.3f, 0.4f}, "block", 0.90f},  // High harm + unfair
        {{0.4f, 0.5f, 0.2f, 0.5f}, "warn", 0.70f},   // Some concerns
    };

    // Train
    std::cout << "Training from LLM decisions...\n";
    float avg_loss = brain.learn_batch(training_data);
    std::cout << "  Average loss: " << std::fixed << std::setprecision(4)
              << avg_loss << "\n\n";

    // Test inference
    std::cout << "Testing fast inference (no LLM needed):\n\n";

    std::vector<std::pair<std::vector<float>, std::string>> test_cases = {
        {{0.9f, 0.5f, 0.5f, 0.5f}, "High harm scenario"},
        {{0.2f, 0.8f, 0.8f, 0.8f}, "Safe, fair scenario"},
        {{0.5f, 0.2f, 0.5f, 0.5f}, "Moderate harm, unfair"},
        {{0.3f, 0.6f, 0.1f, 0.7f}, "Low transparency"},
    };

    uint64_t total_time = 0;

    for (const auto& [features, description] : test_cases) {
        auto decision = brain.decide(features);

        std::cout << "Test Case: " << description << "\n";
        std::cout << "  Features: harm=" << features[0]
                  << ", fair=" << features[1]
                  << ", trans=" << features[2]
                  << ", auto=" << features[3] << "\n";
        std::cout << "  Decision: " << decision.label
                  << " (confidence: " << std::fixed << std::setprecision(2)
                  << decision.confidence << ")\n";
        std::cout << "  Active neurons: " << decision.num_active_neurons
                  << " (sparsity: " << std::fixed << std::setprecision(1)
                  << decision.sparsity * 100.0f << "%)\n";
        std::cout << "  Inference time: " << std::fixed << std::setprecision(3)
                  << decision.inference_time_ms() << " ms\n";
        std::cout << "  Explanation: " << decision.explanation << "\n\n";

        total_time += decision.inference_time_us;
    }

    double avg_time_ms = (total_time / test_cases.size()) / 1000.0;

    std::cout << "Performance Comparison:\n";
    std::cout << "  LLM API call: ~500-2000 ms\n";
    std::cout << "  NIMCP Brain: ~" << std::fixed << std::setprecision(3)
              << avg_time_ms << " ms\n";
    std::cout << "  Speedup: ~" << std::fixed << std::setprecision(0)
              << (1000.0 / avg_time_ms) << "x faster\n\n";

    // Interpretability
    std::cout << "Interpretability - Top Contributing Neurons:\n";
    auto top_neurons = brain.get_top_neurons(5);
    for (const auto& neuron : top_neurons) {
        std::cout << "  Neuron " << neuron.neuron_id
                  << ": importance = " << std::fixed << std::setprecision(4)
                  << neuron.importance << "\n";
    }
    std::cout << "\n";

    // Statistics
    stats = brain.get_stats();
    std::cout << "Brain Statistics:\n";
    std::cout << "  Name: " << stats.task_name << "\n";
    std::cout << "  Neurons: " << stats.num_neurons << "\n";
    std::cout << "  Training steps: " << stats.total_learning_steps << "\n";
    std::cout << "  Inferences: " << stats.total_inferences << "\n";
    std::cout << "  Avg inference time: " << std::fixed << std::setprecision(3)
              << stats.avg_inference_time_ms() << " ms\n";
    std::cout << "  Avg sparsity: " << std::fixed << std::setprecision(1)
              << stats.avg_sparsity * 100.0f << "%\n";
    std::cout << "  Memory: " << std::fixed << std::setprecision(2)
              << stats.memory_mb() << " MB\n\n";

    // Save
    std::cout << "Saving trained brain...\n";
    brain.save("artemis_ethics_brain.nimcp");
    std::cout << "  Saved to: artemis_ethics_brain.nimcp\n\n";

    // Test loading
    std::cout << "Testing load...\n";
    auto loaded_brain = Brain::load("artemis_ethics_brain.nimcp");
    std::cout << "  Loaded: " << loaded_brain.task_name() << "\n";

    // Verify loaded brain works
    auto test_decision = loaded_brain.decide({0.8f, 0.3f, 0.5f, 0.6f});
    std::cout << "  Test decision: " << test_decision.label
              << " (confidence: " << std::fixed << std::setprecision(2)
              << test_decision.confidence << ")\n\n";

    std::cout << "==================================================\n";
    std::cout << " Summary\n";
    std::cout << "==================================================\n\n";
    std::cout << "Benefits:\n";
    std::cout << "  ✓ 100-1000x faster than LLM calls\n";
    std::cout << "  ✓ Works offline (no API dependency)\n";
    std::cout << "  ✓ Zero cost per inference\n";
    std::cout << "  ✓ Privacy preserved (local inference)\n";
    std::cout << "  ✓ Interpretable (can see active neurons)\n";
    std::cout << "  ✓ Lightweight (~" << std::fixed << std::setprecision(1)
              << stats.memory_mb() << "MB vs 7GB+ for LLMs)\n\n";
}

void llm_teacher_demo() {
    std::cout << "\n==================================================\n";
    std::cout << " LLM Teacher Example\n";
    std::cout << "==================================================\n\n";

    // Create brain
    auto brain = create_classifier("student", 2, 2);

    // Simulated LLM teacher
    auto llm_teacher = [](const std::vector<float>& features) -> std::pair<std::string, float> {
        float sum = features[0] + features[1];
        if (sum > 1.0f) {
            return {"class_A", 0.9f};
        } else {
            return {"class_B", 0.85f};
        }
    };

    // Learn from teacher
    std::cout << "Learning from LLM teacher...\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < 100; i++) {
        std::vector<float> features = {dis(gen), dis(gen)};
        float loss = brain.learn_from_llm(features, llm_teacher);

        if ((i + 1) % 20 == 0) {
            std::cout << "  Step " << (i + 1) << ", loss: "
                      << std::fixed << std::setprecision(4) << loss << "\n";
        }
    }

    std::cout << "\nTesting learned knowledge:\n";

    std::vector<std::vector<float>> test_inputs = {
        {0.8f, 0.7f},  // Sum > 1.0, expect class_A
        {0.3f, 0.2f},  // Sum < 1.0, expect class_B
    };

    for (const auto& input : test_inputs) {
        auto decision = brain.decide(input);
        std::cout << "  Input: [" << input[0] << ", " << input[1] << "] "
                  << "-> " << decision.label
                  << " (confidence: " << std::fixed << std::setprecision(2)
                  << decision.confidence << ")\n";
    }

    std::cout << "\n";
}

int main() {
    try {
        // Run ethics demo
        ethics_demo();

        // Run LLM teacher demo
        llm_teacher_demo();

        std::cout << "Demo complete!\n";
        return 0;

    } catch (const BrainException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
