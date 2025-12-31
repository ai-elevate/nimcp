/**
 * @file test_training_enhancements.cpp
 * @brief Comprehensive tests for Training Layer Enhancements (Phase TL-1)
 *
 * Tests coverage for:
 * 1. Mixed Precision Training (AMP)
 * 2. Distributed/Federated Training
 * 3. Knowledge Distillation
 * 4. Quantization-Aware Training (QAT)
 * 5. Meta-Learning (MAML)
 * 6. Hyperparameter Optimization (HPO)
 * 7. Continual/Lifelong Learning (EWC)
 * 8. Multi-Task Learning Coordination
 * 9. Adversarial Training
 * 10. Activation Gradient Scaling
 *
 * @author NIMCP Development Team
 * @date 2025-12-31
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <limits>
#include <cstring>

extern "C" {
#include "training/nimcp_mixed_precision.h"
#include "training/nimcp_distributed_training.h"
#include "training/nimcp_knowledge_distillation.h"
#include "training/nimcp_quantization_aware.h"
#include "training/nimcp_meta_learning.h"
#include "training/nimcp_hyperparam_opt.h"
#include "training/nimcp_continual_learning.h"
#include "training/nimcp_multi_task.h"
#include "training/nimcp_adversarial_training.h"
#include "training/nimcp_gradient_scaling.h"
}

/* ============================================================================
 * SECTION 1: Mixed Precision Training Tests
 * ============================================================================ */

class MixedPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            mp_destroy(ctx);
            ctx = nullptr;
        }
    }

    mp_ctx_t* ctx;
};

TEST_F(MixedPrecisionTest, DefaultConfig_ReasonableValues) {
    mp_config_t config;
    ASSERT_EQ(mp_default_config(&config), 0);

    EXPECT_EQ(config.precision, MP_FP16);
    EXPECT_EQ(config.loss_scaling.type, MP_SCALE_DYNAMIC);
    EXPECT_GT(config.loss_scaling.initial_scale, 0.0f);
    EXPECT_TRUE(config.autocast_enabled);
}

TEST_F(MixedPrecisionTest, Create_ValidConfig) {
    mp_config_t config;
    ASSERT_EQ(mp_default_config(&config), 0);

    ctx = mp_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MixedPrecisionTest, Create_BFloat16) {
    mp_config_t config;
    ASSERT_EQ(mp_default_config(&config), 0);
    config.precision = MP_BF16;

    ctx = mp_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MixedPrecisionTest, StaticScaling_Config) {
    mp_config_t config;
    ASSERT_EQ(mp_default_config(&config), 0);
    config.loss_scaling.type = MP_SCALE_STATIC;
    config.loss_scaling.initial_scale = 1024.0f;

    ctx = mp_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MixedPrecisionTest, ValidateConfig_InvalidPrecision) {
    mp_config_t config;
    mp_default_config(&config);
    config.precision = (mp_precision_t)999;

    EXPECT_NE(mp_validate_config(&config), 0);
}

TEST_F(MixedPrecisionTest, GetStats) {
    mp_config_t config;
    ASSERT_EQ(mp_default_config(&config), 0);
    ctx = mp_create(&config);
    ASSERT_NE(ctx, nullptr);

    mp_stats_t stats;
    EXPECT_EQ(mp_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 2: Distributed Training Tests
 * ============================================================================ */

class DistributedTrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            dist_destroy(ctx);
            ctx = nullptr;
        }
    }

    dist_ctx_t* ctx;
};

TEST_F(DistributedTrainingTest, DefaultConfig_ReasonableValues) {
    dist_config_t config;
    ASSERT_EQ(dist_default_config(&config), 0);

    EXPECT_EQ(config.strategy, DIST_DATA_PARALLEL);
    EXPECT_EQ(config.comm.world_size, 1u);
    EXPECT_EQ(config.comm.rank, 0u);
    EXPECT_FALSE(config.federated.enabled);
}

TEST_F(DistributedTrainingTest, Create_SingleNode) {
    dist_config_t config;
    ASSERT_EQ(dist_default_config(&config), 0);

    ctx = dist_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(DistributedTrainingTest, FederatedConfig) {
    dist_config_t config;
    ASSERT_EQ(dist_federated_config(&config), 0);

    EXPECT_TRUE(config.federated.enabled);
    EXPECT_EQ(config.federated.algorithm, DIST_FEDAVG);
}

TEST_F(DistributedTrainingTest, FSPDConfig) {
    dist_config_t config;
    ASSERT_EQ(dist_fsdp_config(&config), 0);

    EXPECT_EQ(config.strategy, DIST_FSDP);
    EXPECT_TRUE(config.fsdp.cpu_offload);
}

TEST_F(DistributedTrainingTest, ValidateConfig_InvalidStrategy) {
    dist_config_t config;
    dist_default_config(&config);
    config.strategy = (dist_strategy_t)999;

    EXPECT_NE(dist_validate_config(&config), 0);
}

TEST_F(DistributedTrainingTest, GetStats) {
    dist_config_t config;
    ASSERT_EQ(dist_default_config(&config), 0);
    ctx = dist_create(&config);
    ASSERT_NE(ctx, nullptr);

    dist_stats_t stats;
    EXPECT_EQ(dist_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 3: Knowledge Distillation Tests
 * ============================================================================ */

class KnowledgeDistillationTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            kd_destroy(ctx);
            ctx = nullptr;
        }
    }

    kd_ctx_t* ctx;
};

TEST_F(KnowledgeDistillationTest, DefaultConfig_ReasonableValues) {
    kd_config_t config;
    ASSERT_EQ(kd_default_config(&config), 0);

    EXPECT_EQ(config.method, KD_SOFT_LABELS);
    EXPECT_GT(config.temperature, 0.0f);
    EXPECT_GE(config.alpha, 0.0f);
    EXPECT_LE(config.alpha, 1.0f);
}

TEST_F(KnowledgeDistillationTest, Create_ValidConfig) {
    kd_config_t config;
    ASSERT_EQ(kd_default_config(&config), 0);

    ctx = kd_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(KnowledgeDistillationTest, FitNetConfig) {
    kd_config_t config;
    ASSERT_EQ(kd_fitnet_config(&config), 0);

    EXPECT_EQ(config.method, KD_FITNET);
    EXPECT_TRUE(config.hint.enabled);
}

TEST_F(KnowledgeDistillationTest, AttentionTransferConfig) {
    kd_config_t config;
    ASSERT_EQ(kd_attention_transfer_config(&config), 0);

    EXPECT_EQ(config.method, KD_ATTENTION_TRANSFER);
    EXPECT_TRUE(config.attention.enabled);
}

TEST_F(KnowledgeDistillationTest, ValidateConfig_InvalidTemperature) {
    kd_config_t config;
    kd_default_config(&config);
    config.temperature = -1.0f;

    EXPECT_NE(kd_validate_config(&config), 0);
}

TEST_F(KnowledgeDistillationTest, GetStats) {
    kd_config_t config;
    ASSERT_EQ(kd_default_config(&config), 0);
    ctx = kd_create(&config);
    ASSERT_NE(ctx, nullptr);

    kd_stats_t stats;
    EXPECT_EQ(kd_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 4: Quantization-Aware Training Tests
 * ============================================================================ */

class QuantizationAwareTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            qat_destroy(ctx);
            ctx = nullptr;
        }
    }

    qat_ctx_t* ctx;
};

TEST_F(QuantizationAwareTest, DefaultConfig_ReasonableValues) {
    qat_config_t config;
    ASSERT_EQ(qat_default_config(&config), 0);

    EXPECT_EQ(config.quant.weight_bits, 8u);
    EXPECT_EQ(config.quant.activation_bits, 8u);
    EXPECT_TRUE(config.quant.symmetric);
}

TEST_F(QuantizationAwareTest, Create_ValidConfig) {
    qat_config_t config;
    ASSERT_EQ(qat_default_config(&config), 0);

    ctx = qat_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(QuantizationAwareTest, INT4Config) {
    qat_config_t config;
    ASSERT_EQ(qat_int4_config(&config), 0);

    EXPECT_EQ(config.quant.weight_bits, 4u);
    EXPECT_EQ(config.quant.activation_bits, 8u);
}

TEST_F(QuantizationAwareTest, LSQConfig) {
    qat_config_t config;
    ASSERT_EQ(qat_lsq_config(&config), 0);

    EXPECT_EQ(config.method, QAT_LSQ);
    EXPECT_TRUE(config.lsq.learn_scale);
}

TEST_F(QuantizationAwareTest, ValidateConfig_InvalidBits) {
    qat_config_t config;
    qat_default_config(&config);
    config.quant.weight_bits = 0;

    EXPECT_NE(qat_validate_config(&config), 0);
}

TEST_F(QuantizationAwareTest, GetStats) {
    qat_config_t config;
    ASSERT_EQ(qat_default_config(&config), 0);
    ctx = qat_create(&config);
    ASSERT_NE(ctx, nullptr);

    qat_stats_t stats;
    EXPECT_EQ(qat_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 5: Meta-Learning Tests
 * ============================================================================ */

class MetaLearningTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            meta_destroy(ctx);
            ctx = nullptr;
        }
    }

    meta_ctx_t* ctx;
};

TEST_F(MetaLearningTest, DefaultConfig_ReasonableValues) {
    meta_config_t config;
    ASSERT_EQ(meta_default_config(&config), 0);

    EXPECT_EQ(config.algorithm, META_ALG_MAML);
    EXPECT_GT(config.maml.inner_lr, 0.0f);
    EXPECT_GT(config.maml.outer_lr, 0.0f);
    EXPECT_GT(config.maml.inner_steps, 0u);
}

TEST_F(MetaLearningTest, Create_MAML) {
    meta_config_t config;
    ASSERT_EQ(meta_default_config(&config), 0);

    ctx = meta_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MetaLearningTest, Create_FOMAML) {
    meta_config_t config;
    ASSERT_EQ(meta_default_config(&config), 0);
    config.algorithm = META_ALG_FOMAML;
    config.maml.first_order = true;

    ctx = meta_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MetaLearningTest, Create_Reptile) {
    meta_config_t config;
    ASSERT_EQ(meta_default_config(&config), 0);
    config.algorithm = META_ALG_REPTILE;

    ctx = meta_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MetaLearningTest, ValidateConfig_InvalidAlgorithm) {
    meta_config_t config;
    meta_default_config(&config);
    config.algorithm = (meta_algorithm_t)999;

    EXPECT_NE(meta_validate_config(&config), 0);
}

TEST_F(MetaLearningTest, AlgorithmName) {
    EXPECT_STREQ(meta_algorithm_name(META_ALG_MAML), "MAML");
    EXPECT_STREQ(meta_algorithm_name(META_ALG_REPTILE), "Reptile");
}

/* ============================================================================
 * SECTION 6: Hyperparameter Optimization Tests
 * ============================================================================ */

class HPOTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            hpo_destroy(ctx);
            ctx = nullptr;
        }
    }

    hpo_ctx_t* ctx;
};

TEST_F(HPOTest, DefaultConfig_ReasonableValues) {
    hpo_config_t config;
    ASSERT_EQ(hpo_default_config(&config), 0);

    EXPECT_EQ(config.algorithm, HPO_TPE);
    EXPECT_GT(config.max_trials, 0u);
}

TEST_F(HPOTest, Create_TPE) {
    hpo_config_t config;
    ASSERT_EQ(hpo_default_config(&config), 0);

    ctx = hpo_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(HPOTest, Create_Hyperband) {
    hpo_config_t config;
    ASSERT_EQ(hpo_hyperband_config(&config), 0);

    EXPECT_EQ(config.algorithm, HPO_HYPERBAND);
    ctx = hpo_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(HPOTest, Create_PBT) {
    hpo_config_t config;
    ASSERT_EQ(hpo_pbt_config(&config), 0);

    EXPECT_EQ(config.algorithm, HPO_PBT);
    ctx = hpo_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(HPOTest, ValidateConfig_InvalidAlgorithm) {
    hpo_config_t config;
    hpo_default_config(&config);
    config.algorithm = (hpo_algorithm_t)999;

    EXPECT_NE(hpo_validate_config(&config), 0);
}

TEST_F(HPOTest, GetStats) {
    hpo_config_t config;
    ASSERT_EQ(hpo_default_config(&config), 0);
    ctx = hpo_create(&config);
    ASSERT_NE(ctx, nullptr);

    hpo_stats_t stats;
    EXPECT_EQ(hpo_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 7: Continual Learning Tests
 * ============================================================================ */

class ContinualLearningTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            cl_destroy(ctx);
            ctx = nullptr;
        }
    }

    cl_ctx_t* ctx;
};

TEST_F(ContinualLearningTest, DefaultConfig_ReasonableValues) {
    cl_config_t config;
    ASSERT_EQ(cl_default_config(&config), 0);

    EXPECT_EQ(config.method, CL_EWC);
    EXPECT_GT(config.ewc.lambda, 0.0f);
}

TEST_F(ContinualLearningTest, Create_EWC) {
    cl_config_t config;
    ASSERT_EQ(cl_default_config(&config), 0);

    ctx = cl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(ContinualLearningTest, Create_MAS) {
    cl_config_t config;
    ASSERT_EQ(cl_mas_config(&config), 0);

    EXPECT_EQ(config.method, CL_MAS);
    ctx = cl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(ContinualLearningTest, Create_SI) {
    cl_config_t config;
    ASSERT_EQ(cl_si_config(&config), 0);

    EXPECT_EQ(config.method, CL_SI);
    ctx = cl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(ContinualLearningTest, Create_Replay) {
    cl_config_t config;
    ASSERT_EQ(cl_replay_config(&config), 0);

    EXPECT_EQ(config.method, CL_REPLAY);
    EXPECT_TRUE(config.replay.enabled);
    ctx = cl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(ContinualLearningTest, ValidateConfig_InvalidMethod) {
    cl_config_t config;
    cl_default_config(&config);
    config.method = (cl_method_t)999;

    EXPECT_NE(cl_validate_config(&config), 0);
}

TEST_F(ContinualLearningTest, GetStats) {
    cl_config_t config;
    ASSERT_EQ(cl_default_config(&config), 0);
    ctx = cl_create(&config);
    ASSERT_NE(ctx, nullptr);

    cl_stats_t stats;
    EXPECT_EQ(cl_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 8: Multi-Task Learning Tests
 * ============================================================================ */

class MultiTaskLearningTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            mtl_destroy(ctx);
            ctx = nullptr;
        }
    }

    mtl_ctx_t* ctx;
};

TEST_F(MultiTaskLearningTest, DefaultConfig_ReasonableValues) {
    mtl_config_t config;
    ASSERT_EQ(mtl_default_config(&config), 0);

    EXPECT_EQ(config.method, MTL_EQUAL_WEIGHT);
    EXPECT_EQ(config.grad_conflict, MTL_GRAD_PCGRAD);
}

TEST_F(MultiTaskLearningTest, Create_EqualWeight) {
    mtl_config_t config;
    ASSERT_EQ(mtl_default_config(&config), 0);

    ctx = mtl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MultiTaskLearningTest, Create_Uncertainty) {
    mtl_config_t config;
    ASSERT_EQ(mtl_uncertainty_config(&config), 0);

    EXPECT_EQ(config.method, MTL_UNCERTAINTY);
    ctx = mtl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MultiTaskLearningTest, Create_GradNorm) {
    mtl_config_t config;
    ASSERT_EQ(mtl_gradnorm_config(&config), 0);

    EXPECT_EQ(config.method, MTL_GRADNORM);
    ctx = mtl_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(MultiTaskLearningTest, ValidateConfig_InvalidMethod) {
    mtl_config_t config;
    mtl_default_config(&config);
    config.method = (mtl_method_t)999;

    EXPECT_NE(mtl_validate_config(&config), 0);
}

TEST_F(MultiTaskLearningTest, GetStats) {
    mtl_config_t config;
    ASSERT_EQ(mtl_default_config(&config), 0);
    ctx = mtl_create(&config);
    ASSERT_NE(ctx, nullptr);

    mtl_stats_t stats;
    EXPECT_EQ(mtl_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 9: Adversarial Training Tests
 * ============================================================================ */

class AdversarialTrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            adv_destroy(ctx);
            ctx = nullptr;
        }
    }

    adv_ctx_t* ctx;
};

TEST_F(AdversarialTrainingTest, DefaultConfig_ReasonableValues) {
    adv_config_t config;
    ASSERT_EQ(adv_default_config(&config), 0);

    EXPECT_EQ(config.method, ADV_TRAIN_STANDARD);
    EXPECT_EQ(config.attack.type, ADV_ATTACK_PGD);
    EXPECT_GT(config.attack.epsilon, 0.0f);
    EXPECT_GT(config.attack.step_size, 0.0f);
}

TEST_F(AdversarialTrainingTest, Create_Standard) {
    adv_config_t config;
    ASSERT_EQ(adv_default_config(&config), 0);

    ctx = adv_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(AdversarialTrainingTest, Create_TRADES) {
    adv_config_t config;
    ASSERT_EQ(adv_trades_config(&config), 0);

    EXPECT_EQ(config.method, ADV_TRAIN_TRADES);
    EXPECT_GT(config.trades.beta, 0.0f);
    ctx = adv_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(AdversarialTrainingTest, AttackNames) {
    EXPECT_STREQ(adv_attack_name(ADV_ATTACK_FGSM), "FGSM");
    EXPECT_STREQ(adv_attack_name(ADV_ATTACK_PGD), "PGD");
    EXPECT_STREQ(adv_attack_name(ADV_ATTACK_CW), "C&W");
}

TEST_F(AdversarialTrainingTest, NormNames) {
    EXPECT_STREQ(adv_norm_name(ADV_NORM_LINF), "L-inf");
    EXPECT_STREQ(adv_norm_name(ADV_NORM_L2), "L2");
}

TEST_F(AdversarialTrainingTest, ValidateConfig_InvalidAttack) {
    adv_config_t config;
    adv_default_config(&config);
    config.attack.type = (adv_attack_t)999;

    EXPECT_NE(adv_validate_config(&config), 0);
}

TEST_F(AdversarialTrainingTest, GetStats) {
    adv_config_t config;
    ASSERT_EQ(adv_default_config(&config), 0);
    ctx = adv_create(&config);
    ASSERT_NE(ctx, nullptr);

    adv_stats_t stats;
    EXPECT_EQ(adv_get_stats(ctx, &stats), 0);
    EXPECT_EQ(stats.total_attacks, 0u);
}

TEST_F(AdversarialTrainingTest, ResetStats) {
    adv_config_t config;
    ASSERT_EQ(adv_default_config(&config), 0);
    ctx = adv_create(&config);
    ASSERT_NE(ctx, nullptr);

    adv_reset_stats(ctx);
    adv_stats_t stats;
    EXPECT_EQ(adv_get_stats(ctx, &stats), 0);
}

/* ============================================================================
 * SECTION 10: Gradient Scaling Tests
 * ============================================================================ */

class GradientScalingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = nullptr;
    }

    void TearDown() override {
        if (ctx) {
            gs_destroy(ctx);
            ctx = nullptr;
        }
    }

    gs_ctx_t* ctx;
};

TEST_F(GradientScalingTest, DefaultConfig_ReasonableValues) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);

    EXPECT_EQ(config.surrogate, GS_SURROGATE_SUPERSPIKE);
    EXPECT_GT(config.surrogate_beta, 0.0f);
    EXPECT_TRUE(config.adaptive_clipping);
}

TEST_F(GradientScalingTest, Create_SuperSpike) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);

    ctx = gs_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(GradientScalingTest, Create_Sigmoid) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);
    config.surrogate = GS_SURROGATE_SIGMOID;

    ctx = gs_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(GradientScalingTest, Create_Triangle) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);
    config.surrogate = GS_SURROGATE_TRIANGLE;

    ctx = gs_create(&config);
    ASSERT_NE(ctx, nullptr);
}

TEST_F(GradientScalingTest, ComputeSurrogate_SuperSpike) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);
    ctx = gs_create(&config);
    ASSERT_NE(ctx, nullptr);

    // Test surrogate gradient computation
    float grad = gs_surrogate_grad(ctx, 0.0f);
    EXPECT_GT(grad, 0.0f);  // Should be positive at threshold
}

TEST_F(GradientScalingTest, ComputeSurrogate_Values) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);
    ctx = gs_create(&config);
    ASSERT_NE(ctx, nullptr);

    // Gradient should be largest near threshold (v = 0)
    float grad_at_0 = gs_surrogate_grad(ctx, 0.0f);
    float grad_at_1 = gs_surrogate_grad(ctx, 1.0f);
    float grad_at_neg1 = gs_surrogate_grad(ctx, -1.0f);

    // SuperSpike: gradient decreases away from threshold
    EXPECT_GE(grad_at_0, grad_at_1);
    EXPECT_GE(grad_at_0, grad_at_neg1);
}

TEST_F(GradientScalingTest, ValidateConfig_InvalidSurrogate) {
    gs_config_t config;
    gs_default_config(&config);
    config.surrogate = (gs_surrogate_t)999;

    EXPECT_NE(gs_validate_config(&config), 0);
}

TEST_F(GradientScalingTest, GetStats) {
    gs_config_t config;
    ASSERT_EQ(gs_default_config(&config), 0);
    ctx = gs_create(&config);
    ASSERT_NE(ctx, nullptr);

    gs_stats_t stats;
    EXPECT_EQ(gs_get_stats(ctx, &stats), 0);
}

TEST_F(GradientScalingTest, SurrogateName) {
    EXPECT_STREQ(gs_surrogate_name(GS_SURROGATE_SUPERSPIKE), "SuperSpike");
    EXPECT_STREQ(gs_surrogate_name(GS_SURROGATE_SIGMOID), "Sigmoid");
    EXPECT_STREQ(gs_surrogate_name(GS_SURROGATE_TRIANGLE), "Triangle");
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

class TrainingEnhancementsIntegrationTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clean up any resources
    }
};

TEST_F(TrainingEnhancementsIntegrationTest, AllEnhancementsCreate) {
    // Test that all 10 enhancements can be created simultaneously
    mp_config_t mp_config;
    mp_default_config(&mp_config);
    mp_ctx_t* mp = mp_create(&mp_config);
    ASSERT_NE(mp, nullptr);

    dist_config_t dist_config;
    dist_default_config(&dist_config);
    dist_ctx_t* dist = dist_create(&dist_config);
    ASSERT_NE(dist, nullptr);

    kd_config_t kd_config;
    kd_default_config(&kd_config);
    kd_ctx_t* kd = kd_create(&kd_config);
    ASSERT_NE(kd, nullptr);

    qat_config_t qat_config;
    qat_default_config(&qat_config);
    qat_ctx_t* qat = qat_create(&qat_config);
    ASSERT_NE(qat, nullptr);

    meta_config_t meta_config;
    meta_default_config(&meta_config);
    meta_ctx_t* meta = meta_create(&meta_config);
    ASSERT_NE(meta, nullptr);

    hpo_config_t hpo_config;
    hpo_default_config(&hpo_config);
    hpo_ctx_t* hpo = hpo_create(&hpo_config);
    ASSERT_NE(hpo, nullptr);

    cl_config_t cl_config;
    cl_default_config(&cl_config);
    cl_ctx_t* cl = cl_create(&cl_config);
    ASSERT_NE(cl, nullptr);

    mtl_config_t mtl_config;
    mtl_default_config(&mtl_config);
    mtl_ctx_t* mtl = mtl_create(&mtl_config);
    ASSERT_NE(mtl, nullptr);

    adv_config_t adv_config;
    adv_default_config(&adv_config);
    adv_ctx_t* adv = adv_create(&adv_config);
    ASSERT_NE(adv, nullptr);

    gs_config_t gs_config;
    gs_default_config(&gs_config);
    gs_ctx_t* gs = gs_create(&gs_config);
    ASSERT_NE(gs, nullptr);

    // Cleanup
    mp_destroy(mp);
    dist_destroy(dist);
    kd_destroy(kd);
    qat_destroy(qat);
    meta_destroy(meta);
    hpo_destroy(hpo);
    cl_destroy(cl);
    mtl_destroy(mtl);
    adv_destroy(adv);
    gs_destroy(gs);
}

TEST_F(TrainingEnhancementsIntegrationTest, NullSafety) {
    // Test that all destroy functions handle NULL gracefully
    mp_destroy(nullptr);
    dist_destroy(nullptr);
    kd_destroy(nullptr);
    qat_destroy(nullptr);
    meta_destroy(nullptr);
    hpo_destroy(nullptr);
    cl_destroy(nullptr);
    mtl_destroy(nullptr);
    adv_destroy(nullptr);
    gs_destroy(nullptr);

    // Test that all get_stats functions reject NULL
    EXPECT_NE(mp_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(dist_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(kd_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(qat_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(hpo_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(cl_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(mtl_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(adv_get_stats(nullptr, nullptr), 0);
    EXPECT_NE(gs_get_stats(nullptr, nullptr), 0);
}

/* ============================================================================
 * Main Entry Point
 * ============================================================================ */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
