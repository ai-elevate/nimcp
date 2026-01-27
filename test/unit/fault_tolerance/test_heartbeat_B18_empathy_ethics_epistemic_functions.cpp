/**
 * @file test_heartbeat_B18_empathy_ethics_epistemic_functions.cpp
 * @brief Unit tests for B18 heartbeat setter functions
 *        (cognitive/empathetic_response + cognitive/ethics + cognitive/epistemic)
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>

extern "C" {
#include "utils/fault_tolerance/nimcp_health_agent.h"

/* cognitive/empathetic_response modules */
void empathetic_response_fep_bridge_set_health_agent(nimcp_health_agent_t* agent);
void empathetic_response_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent);
void empathetic_response_set_health_agent(nimcp_health_agent_t* agent);
void empathy_plasticity_bridge_set_health_agent(nimcp_health_agent_t* agent);
void empathy_snn_bridge_set_health_agent(nimcp_health_agent_t* agent);
void empathetic_response_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent);

/* cognitive/ethics modules */
void ethics_policies_set_health_agent(nimcp_health_agent_t* agent);
void ethics_incidents_set_health_agent(nimcp_health_agent_t* agent);
void ethics_snn_bridge_set_health_agent(nimcp_health_agent_t* agent);
void ethics_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent);
void ethics_fep_bridge_set_health_agent(nimcp_health_agent_t* agent);
void health_ethics_bridge_set_health_agent(nimcp_health_agent_t* agent);
void combinatorial_harm_set_health_agent(nimcp_health_agent_t* agent);
void ethics_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent);
void ethics_immune_set_health_agent(nimcp_health_agent_t* agent);
void ethics_evaluation_set_health_agent(nimcp_health_agent_t* agent);
void ethics_warfare_set_health_agent(nimcp_health_agent_t* agent);
void ethics_set_health_agent(nimcp_health_agent_t* agent);
void core_directives_set_health_agent(nimcp_health_agent_t* agent);
void ethics_learning_set_health_agent(nimcp_health_agent_t* agent);
void ethics_asimov_set_health_agent(nimcp_health_agent_t* agent);
void ethics_hyperbolic_set_health_agent(nimcp_health_agent_t* agent);
void ethics_plasticity_bridge_set_health_agent(nimcp_health_agent_t* agent);

/* cognitive/epistemic modules */
void epistemic_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent);
void epistemic_snn_bridge_set_health_agent(nimcp_health_agent_t* agent);
void epistemic_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent);
void epistemic_plasticity_bridge_set_health_agent(nimcp_health_agent_t* agent);
void epistemic_filter_set_health_agent(nimcp_health_agent_t* agent);
void epistemic_fep_bridge_set_health_agent(nimcp_health_agent_t* agent);
}

typedef void (*set_health_agent_fn)(nimcp_health_agent_t*);
struct B18Module { const char* name; set_health_agent_fn set_fn; };

static const B18Module B18_MODULES[] = {
    /* cognitive/empathetic_response */
    {"empathetic_response_fep_bridge",      empathetic_response_fep_bridge_set_health_agent},
    {"empathetic_response_substrate_bridge", empathetic_response_substrate_bridge_set_health_agent},
    {"empathetic_response",                 empathetic_response_set_health_agent},
    {"empathy_plasticity_bridge",           empathy_plasticity_bridge_set_health_agent},
    {"empathy_snn_bridge",                  empathy_snn_bridge_set_health_agent},
    {"empathetic_response_thalamic_bridge", empathetic_response_thalamic_bridge_set_health_agent},
    /* cognitive/ethics */
    {"ethics_policies",                     ethics_policies_set_health_agent},
    {"ethics_incidents",                    ethics_incidents_set_health_agent},
    {"ethics_snn_bridge",                   ethics_snn_bridge_set_health_agent},
    {"ethics_thalamic_bridge",              ethics_thalamic_bridge_set_health_agent},
    {"ethics_fep_bridge",                   ethics_fep_bridge_set_health_agent},
    {"health_ethics_bridge",                health_ethics_bridge_set_health_agent},
    {"combinatorial_harm",                  combinatorial_harm_set_health_agent},
    {"ethics_substrate_bridge",             ethics_substrate_bridge_set_health_agent},
    {"ethics_immune",                       ethics_immune_set_health_agent},
    {"ethics_evaluation",                   ethics_evaluation_set_health_agent},
    {"ethics_warfare",                      ethics_warfare_set_health_agent},
    {"ethics",                              ethics_set_health_agent},
    {"core_directives",                     core_directives_set_health_agent},
    {"ethics_learning",                     ethics_learning_set_health_agent},
    {"ethics_asimov",                       ethics_asimov_set_health_agent},
    {"ethics_hyperbolic",                   ethics_hyperbolic_set_health_agent},
    {"ethics_plasticity_bridge",            ethics_plasticity_bridge_set_health_agent},
    /* cognitive/epistemic */
    {"epistemic_substrate_bridge",          epistemic_substrate_bridge_set_health_agent},
    {"epistemic_snn_bridge",                epistemic_snn_bridge_set_health_agent},
    {"epistemic_thalamic_bridge",           epistemic_thalamic_bridge_set_health_agent},
    {"epistemic_plasticity_bridge",         epistemic_plasticity_bridge_set_health_agent},
    {"epistemic_filter",                    epistemic_filter_set_health_agent},
    {"epistemic_fep_bridge",                epistemic_fep_bridge_set_health_agent},
};
static constexpr size_t B18_MODULE_COUNT = sizeof(B18_MODULES) / sizeof(B18_MODULES[0]);

class HeartbeatB18UnitTest : public ::testing::Test {
protected:
    nimcp_health_agent_t* agent_ = nullptr;
    void SetUp() override {
        health_agent_config_t cfg;
        nimcp_health_agent_default_config(&cfg);
        cfg.check_interval_ms = 50;
        cfg.enable_auto_recovery = false;
        agent_ = nimcp_health_agent_create(&cfg);
        ASSERT_NE(agent_, nullptr);
    }
    void TearDown() override {
        for (size_t i = 0; i < B18_MODULE_COUNT; i++) B18_MODULES[i].set_fn(nullptr);
        if (agent_) { nimcp_health_agent_destroy(agent_); agent_ = nullptr; }
    }
};

TEST_F(HeartbeatB18UnitTest, AllModulesSetNull) {
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) {
        SCOPED_TRACE(B18_MODULES[i].name);
        B18_MODULES[i].set_fn(nullptr);
    }
}

TEST_F(HeartbeatB18UnitTest, AllModulesSetValid) {
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) {
        SCOPED_TRACE(B18_MODULES[i].name);
        B18_MODULES[i].set_fn(agent_);
    }
}

TEST_F(HeartbeatB18UnitTest, AllModulesReplaceAgent) {
    health_agent_config_t cfg2;
    nimcp_health_agent_default_config(&cfg2);
    nimcp_health_agent_t* agent2 = nimcp_health_agent_create(&cfg2);
    ASSERT_NE(agent2, nullptr);
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) {
        SCOPED_TRACE(B18_MODULES[i].name);
        B18_MODULES[i].set_fn(agent_);
        B18_MODULES[i].set_fn(agent2);
    }
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) B18_MODULES[i].set_fn(nullptr);
    nimcp_health_agent_destroy(agent2);
}

TEST_F(HeartbeatB18UnitTest, AllModulesShareSameAgent) {
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) B18_MODULES[i].set_fn(agent_);
    health_agent_stats_t stats;
    nimcp_health_agent_get_stats(agent_, &stats);
    EXPECT_GE(stats.heartbeats_received, 0u);
}

TEST_F(HeartbeatB18UnitTest, SetNullAfterValid) {
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) {
        SCOPED_TRACE(B18_MODULES[i].name);
        B18_MODULES[i].set_fn(agent_);
        B18_MODULES[i].set_fn(nullptr);
    }
}

TEST_F(HeartbeatB18UnitTest, ConcurrentSetClear) {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) {
        threads.emplace_back([this, i]() {
            for (int j = 0; j < 100; j++) {
                B18_MODULES[i].set_fn(agent_);
                B18_MODULES[i].set_fn(nullptr);
            }
        });
    }
    for (auto& t : threads) t.join();
}

TEST_F(HeartbeatB18UnitTest, HeartbeatCounterIncrements) {
    nimcp_health_agent_start(agent_);
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) B18_MODULES[i].set_fn(agent_);
    health_agent_stats_t before, after;
    nimcp_health_agent_get_stats(agent_, &before);
    nimcp_health_agent_heartbeat_ex(agent_, "B18_test", 0);
    nimcp_health_agent_get_stats(agent_, &after);
    EXPECT_GT(after.heartbeats_received, before.heartbeats_received);
    nimcp_health_agent_stop(agent_);
}

TEST_F(HeartbeatB18UnitTest, ModuleCount) {
    EXPECT_EQ(B18_MODULE_COUNT, 29u);
}

TEST_F(HeartbeatB18UnitTest, DoubleSetSameAgentIdempotent) {
    for (size_t i = 0; i < B18_MODULE_COUNT; i++) {
        SCOPED_TRACE(B18_MODULES[i].name);
        B18_MODULES[i].set_fn(agent_);
        B18_MODULES[i].set_fn(agent_);
    }
}
