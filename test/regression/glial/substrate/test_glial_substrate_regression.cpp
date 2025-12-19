/**
 * @file test_glial_substrate_regression.cpp
 * @brief Regression tests for glial-substrate bridge
 *
 * WHAT: Test for regression in bridge behavior, performance, and numerical stability
 * WHY:  Ensure consistent behavior and detect performance/correctness regressions
 * HOW:  Test performance, memory, numerical stability, thread safety
 *
 * REGRESSION TEST CATEGORIES:
 * - Performance overhead
 * - Memory usage and leaks
 * - Numerical stability
 * - Thread safety
 * - Historical bug reproductions
 *
 * @author NIMCP Development Team
 * @date 2025-12-19
 */

#include <gtest/gtest.h>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

#include "glial/nimcp_glial_substrate_bridge.h"
#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "glial/astrocytes/nimcp_astrocytes.h"
#include "glial/oligodendrocytes/nimcp_oligodendrocytes.h"
#include "glial/microglia/nimcp_microglia.h"
#include "glial/myelin_sheath/nimcp_myelin_sheath.h"
#include "utils/memory/nimcp_memory.h"

//=============================================================================
// Test Fixture
//=============================================================================

class GlialSubstrateRegressionTest : public ::testing::Test {
protected:
    neural_substrate_t* substrate = nullptr;
    astrocyte_network_t* astro_network = nullptr;
    oligodendrocyte_network_t* oligo_network = nullptr;
    microglia_network_t* micro_network = nullptr;
    myelin_sheath_network_t* myelin_network = nullptr;
    glial_substrate_bridge_t* bridge = nullptr;

    void SetUp() override {
        substrate_config_t sub_config = {};
        substrate_config_init(&sub_config);
        substrate = substrate_create(&sub_config);
        ASSERT_NE(substrate, nullptr);

        astrocyte_network_config_t astro_config;
        astrocyte_network_default_config(&astro_config);
        astro_network = astrocyte_network_create(&astro_config, 100);
        ASSERT_NE(astro_network, nullptr);

        oligodendrocyte_network_config_t oligo_config;
        oligodendrocyte_network_default_config(&oligo_config);
        oligo_network = oligodendrocyte_network_create(&oligo_config, 50);
        ASSERT_NE(oligo_network, nullptr);

        microglia_network_config_t micro_config;
        microglia_network_default_config(&micro_config);
        micro_network = microglia_network_create(&micro_config, 20);
        ASSERT_NE(micro_network, nullptr);

        myelin_sheath_network_config_t myelin_config;
        myelin_sheath_network_default_config(&myelin_config);
        myelin_network = myelin_sheath_network_create(&myelin_config, 100);
        ASSERT_NE(myelin_network, nullptr);

        glial_substrate_config_t config;
        glial_substrate_default_config(&config);
        bridge = glial_substrate_bridge_create(&config, substrate,
            astro_network, oligo_network, micro_network, myelin_network);
        ASSERT_NE(bridge, nullptr);
    }

    void TearDown() override {
        glial_substrate_bridge_destroy(bridge);
        myelin_sheath_network_destroy(myelin_network);
        microglia_network_destroy(micro_network);
        oligodendrocyte_network_destroy(oligo_network);
        astrocyte_network_destroy(astro_network);
        substrate_destroy(substrate);
    }
};

//=============================================================================
// Performance Regression Tests
//=============================================================================

TEST_F(GlialSubstrateRegressionTest, BridgeUpdatePerformance) {
    // WHAT: Measure bridge update performance
    // WHY:  Detect performance regressions

    const int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        glial_substrate_bridge_update(bridge, 1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double us_per_update = static_cast<double>(duration.count()) / iterations;

    // Should be < 20 microseconds per update (all glial types)
    EXPECT_LT(us_per_update, 20.0);

    std::cout << "Bridge update: " << us_per_update << " us/update" << std::endl;
}

TEST_F(GlialSubstrateRegressionTest, AstrocyteEffectsUpdatePerformance) {
    // WHAT: Measure astrocyte effects update performance
    // WHY:  Baseline for individual glial type

    const int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        glial_substrate_update_astrocyte_effects(bridge);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double us_per_update = static_cast<double>(duration.count()) / iterations;

    EXPECT_LT(us_per_update, 5.0);

    std::cout << "Astrocyte update: " << us_per_update << " us/update" << std::endl;
}

TEST_F(GlialSubstrateRegressionTest, GlialSupportComputationPerformance) {
    // WHAT: Measure glial support computation performance
    // WHY:  Verify lactate shuttle overhead

    const int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        glial_substrate_compute_all_support(bridge);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double us_per_compute = static_cast<double>(duration.count()) / iterations;

    EXPECT_LT(us_per_compute, 15.0);

    std::cout << "Support computation: " << us_per_compute << " us/compute" << std::endl;
}

//=============================================================================
// Memory Regression Tests
//=============================================================================

TEST_F(GlialSubstrateRegressionTest, BridgeMemoryFootprint) {
    // WHAT: Verify bridge memory footprint
    // WHY:  Detect memory bloat

    size_t initial_allocated = nimcp_get_allocated_memory();

    glial_substrate_config_t config;
    glial_substrate_default_config(&config);
    glial_substrate_bridge_t* test_bridge = glial_substrate_bridge_create(&config, substrate,
        astro_network, oligo_network, micro_network, myelin_network);
    ASSERT_NE(test_bridge, nullptr);

    size_t after_create = nimcp_get_allocated_memory();
    size_t footprint = after_create - initial_allocated;

    glial_substrate_bridge_destroy(test_bridge);

    // Should be < 20 KB (4 glial types + stats)
    EXPECT_LT(footprint, 20480);

    std::cout << "Bridge footprint: " << footprint << " bytes" << std::endl;
}

TEST_F(GlialSubstrateRegressionTest, NoMemoryLeaks) {
    // WHAT: Test for memory leaks
    // WHY:  Verify proper cleanup

    size_t initial = nimcp_get_allocated_memory();

    for (int i = 0; i < 100; i++) {
        glial_substrate_config_t config;
        glial_substrate_default_config(&config);
        glial_substrate_bridge_t* test_bridge = glial_substrate_bridge_create(&config, substrate,
            astro_network, oligo_network, micro_network, myelin_network);

        glial_substrate_bridge_update(test_bridge, 1);

        glial_substrate_bridge_destroy(test_bridge);
    }

    size_t final = nimcp_get_allocated_memory();

    EXPECT_NEAR(final, initial, 2048); // Allow 2KB variance
}

TEST_F(GlialSubstrateRegressionTest, MemoryStabilityOverTime) {
    // WHAT: Test memory doesn't grow over time
    // WHY:  Detect accumulation bugs

    size_t initial = nimcp_get_allocated_memory();

    for (int i = 0; i < 1000; i++) {
        glial_substrate_bridge_update(bridge, 1);
    }

    size_t final = nimcp_get_allocated_memory();

    // Memory should be stable (no accumulation)
    EXPECT_NEAR(final, initial, 1024);
}

//=============================================================================
// Numerical Stability Tests
//=============================================================================

TEST_F(GlialSubstrateRegressionTest, ATPModulationStability) {
    // WHAT: Test ATP modulation is numerically stable
    // WHY:  Verify no NaN or overflow

    for (int i = 0; i < 1000; i++) {
        float atp = 0.1f + 0.8f * (i / 1000.0f);
        substrate_set_atp_level(substrate, atp);

        glial_substrate_update_all_effects(bridge);

        substrate_astrocyte_effects_t astro_effects;
        glial_substrate_get_astrocyte_effects(bridge, &astro_effects);

        EXPECT_FALSE(std::isnan(astro_effects.atp_modulation));
        EXPECT_FALSE(std::isinf(astro_effects.atp_modulation));
        EXPECT_GE(astro_effects.atp_modulation, 0.0f);
        EXPECT_LE(astro_effects.atp_modulation, 1.5f);
    }
}

TEST_F(GlialSubstrateRegressionTest, Q10ScalingStability) {
    // WHAT: Test Q10 temperature scaling is stable
    // WHY:  Verify no overflow at extremes

    float test_temps[] = {30.0f, 35.0f, 37.0f, 39.0f, 42.0f};

    for (float temp : test_temps) {
        substrate_set_temperature(substrate, temp);

        glial_substrate_update_all_effects(bridge);

        substrate_astrocyte_effects_t astro_effects;
        glial_substrate_get_astrocyte_effects(bridge, &astro_effects);

        EXPECT_FALSE(std::isnan(astro_effects.temp_q10_factor));
        EXPECT_FALSE(std::isinf(astro_effects.temp_q10_factor));
        EXPECT_GT(astro_effects.temp_q10_factor, 0.0f);
        EXPECT_LT(astro_effects.temp_q10_factor, 5.0f);
    }
}

TEST_F(GlialSubstrateRegressionTest, LactateShuttleStability) {
    // WHAT: Test lactate computation doesn't overflow
    // WHY:  Verify safe accumulation

    for (int i = 0; i < 1000; i++) {
        glial_substrate_compute_all_support(bridge);

        float total_support = glial_substrate_get_total_atp_support(bridge);

        EXPECT_FALSE(std::isnan(total_support));
        EXPECT_FALSE(std::isinf(total_support));
        EXPECT_GE(total_support, 0.0f);
    }
}

TEST_F(GlialSubstrateRegressionTest, OxygenEffectsStability) {
    // WHAT: Test oxygen modulation is stable
    // WHY:  Verify hypoxia handling

    float test_o2[] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    for (float o2 : test_o2) {
        substrate_set_oxygen_level(substrate, o2);

        glial_substrate_update_all_effects(bridge);

        substrate_astrocyte_effects_t astro_effects;
        glial_substrate_get_astrocyte_effects(bridge, &astro_effects);

        EXPECT_FALSE(std::isnan(astro_effects.o2_modulation));
        EXPECT_GE(astro_effects.o2_modulation, 0.0f);
        EXPECT_LE(astro_effects.o2_modulation, 1.0f);
    }
}

//=============================================================================
// Thread Safety Tests
//=============================================================================

TEST_F(GlialSubstrateRegressionTest, ConcurrentUpdates) {
    // WHAT: Test concurrent updates are thread-safe
    // WHY:  Verify mutex protection

    std::atomic<int> success_count(0);
    const int threads = 4;
    const int iterations_per_thread = 100;

    std::vector<std::thread> workers;

    for (int t = 0; t < threads; t++) {
        workers.emplace_back([this, &success_count, iterations_per_thread]() {
            for (int i = 0; i < iterations_per_thread; i++) {
                if (glial_substrate_bridge_update(bridge, 1) == 0) {
                    success_count++;
                }
                std::this_thread::yield();
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(success_count, threads * iterations_per_thread);
}

TEST_F(GlialSubstrateRegressionTest, ConcurrentReadWrite) {
    // WHAT: Test concurrent reads and writes
    // WHY:  Verify no data races

    std::atomic<bool> stop(false);
    std::atomic<int> read_count(0);
    std::atomic<int> write_count(0);

    // Writer thread
    std::thread writer([this, &stop, &write_count]() {
        while (!stop) {
            substrate_set_atp_level(substrate, 0.7f);
            glial_substrate_update_all_effects(bridge);
            glial_substrate_compute_all_support(bridge);
            write_count++;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int i = 0; i < 3; i++) {
        readers.emplace_back([this, &stop, &read_count]() {
            while (!stop) {
                glial_substrate_get_total_atp_support(bridge);
                substrate_astrocyte_effects_t effects;
                glial_substrate_get_astrocyte_effects(bridge, &effects);
                read_count++;
                std::this_thread::yield();
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;

    writer.join();
    for (auto& reader : readers) {
        reader.join();
    }

    EXPECT_GT(read_count, 0);
    EXPECT_GT(write_count, 0);
}

//=============================================================================
// Historical Bug Reproductions
//=============================================================================

TEST_F(GlialSubstrateRegressionTest, ZeroATPNoNaN) {
    // WHAT: Test zero ATP doesn't produce NaN
    // WHY:  Historical edge case

    substrate_set_atp_level(substrate, 0.0f);

    ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);

    substrate_astrocyte_effects_t astro_effects;
    ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &astro_effects), 0);

    EXPECT_FALSE(std::isnan(astro_effects.atp_modulation));
}

TEST_F(GlialSubstrateRegressionTest, ZeroOxygenNoNaN) {
    // WHAT: Test zero oxygen doesn't produce NaN
    // WHY:  Extreme hypoxia handling

    substrate_set_oxygen_level(substrate, 0.0f);

    ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);

    substrate_microglia_effects_t micro_effects;
    ASSERT_EQ(glial_substrate_get_microglia_effects(bridge, &micro_effects), 0);

    EXPECT_FALSE(std::isnan(micro_effects.o2_modulation));
}

TEST_F(GlialSubstrateRegressionTest, ExtremeTemperatureHandling) {
    // WHAT: Test extreme temperatures are handled
    // WHY:  Prevent Q10 overflow

    float extreme_temps[] = {15.0f, 20.0f, 45.0f, 50.0f};

    for (float temp : extreme_temps) {
        substrate_set_temperature(substrate, temp);

        ASSERT_EQ(glial_substrate_update_all_effects(bridge), 0);

        substrate_astrocyte_effects_t effects;
        ASSERT_EQ(glial_substrate_get_astrocyte_effects(bridge, &effects), 0);

        EXPECT_FALSE(std::isnan(effects.temp_q10_factor));
        EXPECT_FALSE(std::isinf(effects.temp_q10_factor));
    }
}

TEST_F(GlialSubstrateRegressionTest, NullSubstrateHandling) {
    // WHAT: Test NULL substrate is rejected
    // WHY:  Defensive programming

    glial_substrate_config_t config;
    glial_substrate_default_config(&config);
    glial_substrate_bridge_t* test_bridge = glial_substrate_bridge_create(&config, nullptr,
        astro_network, oligo_network, micro_network, myelin_network);

    EXPECT_EQ(test_bridge, nullptr);
}

TEST_F(GlialSubstrateRegressionTest, AllNullGlialNetworks) {
    // WHAT: Test bridge can be created with no glial networks
    // WHY:  Verify graceful degradation

    glial_substrate_config_t config;
    glial_substrate_default_config(&config);
    glial_substrate_bridge_t* test_bridge = glial_substrate_bridge_create(&config, substrate,
        nullptr, nullptr, nullptr, nullptr);

    ASSERT_NE(test_bridge, nullptr);

    // Should not crash when updating
    ASSERT_EQ(glial_substrate_update_all_effects(test_bridge), 0);

    glial_substrate_bridge_destroy(test_bridge);
}

TEST_F(GlialSubstrateRegressionTest, RepeatedCreateDestroy) {
    // WHAT: Test repeated create/destroy cycles
    // WHY:  Detect initialization bugs

    for (int i = 0; i < 50; i++) {
        glial_substrate_config_t config;
        glial_substrate_default_config(&config);
        glial_substrate_bridge_t* test_bridge = glial_substrate_bridge_create(&config, substrate,
            astro_network, oligo_network, micro_network, myelin_network);
        ASSERT_NE(test_bridge, nullptr);

        glial_substrate_bridge_update(test_bridge, 1);

        glial_substrate_bridge_destroy(test_bridge);
    }
}

//=============================================================================
// Consistency Tests
//=============================================================================

TEST_F(GlialSubstrateRegressionTest, DeterministicBehavior) {
    // WHAT: Test deterministic output
    // WHY:  Verify reproducibility

    substrate_set_atp_level(substrate, 0.6f);
    substrate_set_temperature(substrate, 37.5f);

    glial_substrate_update_all_effects(bridge);
    float support1 = glial_substrate_get_total_atp_support(bridge);

    glial_substrate_update_all_effects(bridge);
    float support2 = glial_substrate_get_total_atp_support(bridge);

    EXPECT_FLOAT_EQ(support1, support2);
}

TEST_F(GlialSubstrateRegressionTest, StatisticsMonotonicity) {
    // WHAT: Test statistics are monotonic
    // WHY:  Verify no counter rollback

    glial_substrate_stats_t stats1, stats2;
    glial_substrate_get_stats(bridge, &stats1);

    glial_substrate_bridge_update(bridge, 1);

    glial_substrate_get_stats(bridge, &stats2);

    EXPECT_GE(stats2.total_updates, stats1.total_updates);
}

TEST_F(GlialSubstrateRegressionTest, SupportNeverNegative) {
    // WHAT: Test glial support is never negative
    // WHY:  Verify physical constraint

    for (int i = 0; i < 100; i++) {
        float atp = 0.1f + 0.8f * (i / 100.0f);
        substrate_set_atp_level(substrate, atp);

        glial_substrate_compute_all_support(bridge);

        glial_substrate_support_t support;
        glial_substrate_get_support(bridge, &support);

        EXPECT_GE(support.astro_atp_contribution, 0.0f);
        EXPECT_GE(support.oligo_atp_contribution, 0.0f);
        EXPECT_GE(support.myelin_atp_savings, 0.0f);
        EXPECT_GE(support.pruning_atp_savings, 0.0f);
        EXPECT_GE(support.total_atp_support, 0.0f);
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
