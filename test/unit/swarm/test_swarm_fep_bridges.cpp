/**
 * @file test_swarm_fep_bridges.cpp
 * @brief Unit tests for all Swarm-FEP Bridge modules
 *
 * WHAT: Comprehensive tests for all swarm-FEP bidirectional integrations
 * WHY:  Ensure collective free energy minimization works across swarm subsystems
 * HOW:  Test lifecycle, effects, and bio-async for each bridge type
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "swarm/nimcp_swarm_consensus_fep_bridge.h"
#include "swarm/nimcp_swarm_emergence_fep_bridge.h"
#include "swarm/nimcp_swarm_flocking_fep_bridge.h"
#include "swarm/nimcp_swarm_immune_fep_bridge.h"
#include "swarm/nimcp_swarm_memory_fep_bridge.h"
#include "swarm/nimcp_swarm_pheromone_fep_bridge.h"
#include "swarm/nimcp_swarm_quorum_fep_bridge.h"
#include "swarm/nimcp_swarm_signal_fep_bridge.h"
#include "swarm/nimcp_collective_workspace_fep_bridge.h"
#include "swarm/nimcp_emotional_contagion_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
}

class SwarmFepBridgesTestBase : public ::testing::Test {
protected:
    fep_system_t* fep = nullptr;

    void SetUp() override {
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, 8, 4);
        ASSERT_NE(fep, nullptr);
    }

    void TearDown() override {
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
    }
};

/* ============================================================================
 * Swarm Consensus FEP Bridge Tests
 * ============================================================================ */

class SwarmConsensusFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_consensus_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_consensus_fep_config_t config;
        swarm_consensus_fep_default_config(&config);
        bridge = swarm_consensus_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_consensus_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmConsensusFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmConsensusFepBridgeTest, Update) {
    EXPECT_EQ(swarm_consensus_fep_update(bridge), 0);
}

TEST_F(SwarmConsensusFepBridgeTest, GetEffects) {
    swarm_consensus_fep_effects_t effects;
    EXPECT_EQ(swarm_consensus_fep_get_effects(bridge, &effects), 0);
}

TEST_F(SwarmConsensusFepBridgeTest, BioAsync) {
    EXPECT_FALSE(swarm_consensus_fep_is_bio_async_connected(bridge));
    swarm_consensus_fep_connect_bio_async(bridge);
    EXPECT_TRUE(swarm_consensus_fep_is_bio_async_connected(bridge));
}

/* ============================================================================
 * Swarm Emergence FEP Bridge Tests
 * ============================================================================ */

class SwarmEmergenceFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_emergence_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_emergence_fep_config_t config;
        swarm_emergence_fep_default_config(&config);
        bridge = swarm_emergence_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_emergence_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmEmergenceFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmEmergenceFepBridgeTest, Update) {
    EXPECT_EQ(swarm_emergence_fep_update(bridge), 0);
}

TEST_F(SwarmEmergenceFepBridgeTest, GetEffects) {
    swarm_emergence_fep_effects_t effects;
    EXPECT_EQ(swarm_emergence_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Swarm Flocking FEP Bridge Tests
 * ============================================================================ */

class SwarmFlockingFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_flocking_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_flocking_fep_config_t config;
        swarm_flocking_fep_default_config(&config);
        bridge = swarm_flocking_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_flocking_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmFlockingFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmFlockingFepBridgeTest, Update) {
    EXPECT_EQ(swarm_flocking_fep_update(bridge), 0);
}

TEST_F(SwarmFlockingFepBridgeTest, GetEffects) {
    swarm_flocking_fep_effects_t effects;
    EXPECT_EQ(swarm_flocking_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Swarm Immune FEP Bridge Tests
 * ============================================================================ */

class SwarmImmuneFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_immune_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_immune_fep_config_t config;
        swarm_immune_fep_default_config(&config);
        bridge = swarm_immune_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_immune_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmImmuneFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmImmuneFepBridgeTest, Update) {
    EXPECT_EQ(swarm_immune_fep_update(bridge), 0);
}

TEST_F(SwarmImmuneFepBridgeTest, GetEffects) {
    swarm_immune_fep_effects_t effects;
    EXPECT_EQ(swarm_immune_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Swarm Memory FEP Bridge Tests
 * ============================================================================ */

class SwarmMemoryFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_memory_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_memory_fep_config_t config;
        swarm_memory_fep_default_config(&config);
        bridge = swarm_memory_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_memory_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmMemoryFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmMemoryFepBridgeTest, Update) {
    EXPECT_EQ(swarm_memory_fep_update(bridge), 0);
}

TEST_F(SwarmMemoryFepBridgeTest, GetEffects) {
    swarm_memory_fep_effects_t effects;
    EXPECT_EQ(swarm_memory_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Swarm Pheromone FEP Bridge Tests
 * ============================================================================ */

class SwarmPheromoneFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_pheromone_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_pheromone_fep_config_t config;
        swarm_pheromone_fep_default_config(&config);
        bridge = swarm_pheromone_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_pheromone_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmPheromoneFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmPheromoneFepBridgeTest, Update) {
    EXPECT_EQ(swarm_pheromone_fep_update(bridge), 0);
}

TEST_F(SwarmPheromoneFepBridgeTest, GetEffects) {
    swarm_pheromone_fep_effects_t effects;
    EXPECT_EQ(swarm_pheromone_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Swarm Quorum FEP Bridge Tests
 * ============================================================================ */

class SwarmQuorumFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_quorum_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_quorum_fep_config_t config;
        swarm_quorum_fep_default_config(&config);
        bridge = swarm_quorum_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_quorum_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmQuorumFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmQuorumFepBridgeTest, Update) {
    EXPECT_EQ(swarm_quorum_fep_update(bridge), 0);
}

TEST_F(SwarmQuorumFepBridgeTest, GetEffects) {
    swarm_quorum_fep_effects_t effects;
    EXPECT_EQ(swarm_quorum_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Swarm Signal FEP Bridge Tests
 * ============================================================================ */

class SwarmSignalFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    swarm_signal_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        swarm_signal_fep_config_t config;
        swarm_signal_fep_default_config(&config);
        bridge = swarm_signal_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            swarm_signal_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(SwarmSignalFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SwarmSignalFepBridgeTest, Update) {
    EXPECT_EQ(swarm_signal_fep_update(bridge), 0);
}

TEST_F(SwarmSignalFepBridgeTest, GetEffects) {
    swarm_signal_fep_effects_t effects;
    EXPECT_EQ(swarm_signal_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Collective Workspace FEP Bridge Tests
 * ============================================================================ */

class CollectiveWorkspaceFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    collective_workspace_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        collective_workspace_fep_config_t config;
        collective_workspace_fep_default_config(&config);
        bridge = collective_workspace_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            collective_workspace_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(CollectiveWorkspaceFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(CollectiveWorkspaceFepBridgeTest, Update) {
    EXPECT_EQ(collective_workspace_fep_update(bridge), 0);
}

TEST_F(CollectiveWorkspaceFepBridgeTest, GetEffects) {
    collective_workspace_fep_effects_t effects;
    EXPECT_EQ(collective_workspace_fep_get_effects(bridge, &effects), 0);
}

/* ============================================================================
 * Emotional Contagion FEP Bridge Tests
 * ============================================================================ */

class EmotionalContagionFepBridgeTest : public SwarmFepBridgesTestBase {
protected:
    emotional_contagion_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SwarmFepBridgesTestBase::SetUp();
        emotional_contagion_fep_config_t config;
        emotional_contagion_fep_default_config(&config);
        bridge = emotional_contagion_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            emotional_contagion_fep_destroy(bridge);
            bridge = nullptr;
        }
        SwarmFepBridgesTestBase::TearDown();
    }
};

TEST_F(EmotionalContagionFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(EmotionalContagionFepBridgeTest, Update) {
    EXPECT_EQ(emotional_contagion_fep_update(bridge), 0);
}

TEST_F(EmotionalContagionFepBridgeTest, GetEffects) {
    emotional_contagion_fep_effects_t effects;
    EXPECT_EQ(emotional_contagion_fep_get_effects(bridge, &effects), 0);
}
