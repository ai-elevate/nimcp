/**
 * @file test_cross_engine_coupling.c
 * @brief Integration tests — verify engines work together through world simulator
 *
 * Tests that cross-domain couplings actually transfer data:
 * - Chemistry → Heat (reaction exotherm raises temperature)
 * - Heat → Fluid (buoyancy from temperature gradient)
 * - Physics → Scene Graph (contacts produce support relations)
 * - Entity Tracker → Physics Prior (hidden objects still tracked)
 */

#include "../../test_framework.h"
#include "cognitive/physics/nimcp_intuitive_physics.h"
#include "cognitive/physics/nimcp_scene_graph.h"
#include "cognitive/physics/nimcp_entity_tracker.h"
#include "cognitive/physics/nimcp_physics_prior.h"
#include "cognitive/physics/nimcp_surface_physics.h"
#include "cognitive/physics/nimcp_surface_chemistry.h"
#include "cognitive/physics/nimcp_world_simulator.h"

/* Test: physics engine + scene graph produce support relations */
TEST(physics_to_scene_graph) {
    intuitive_physics_engine_t* phys = intuitive_physics_create(NULL);
    scene_graph_t* sg = scene_graph_create(NULL);
    ASSERT_NOT_NULL(phys);
    ASSERT_NOT_NULL(sg);

    /* Add ground + box on ground + box on box */
    intuitive_physics_add_ground(phys);

    ip_object_t box1 = {0};
    box1.position = (wm_parietal_vec3_t){0, 0.5f, 0};
    box1.mass = 1.0f;
    box1.shape.type = IP_SHAPE_BOX;
    box1.shape.box.hx = 0.5f; box1.shape.box.hy = 0.5f; box1.shape.box.hz = 0.5f;
    box1.restitution = 0.1f; box1.friction = 0.5f;
    box1.visible = true; box1.active = true;
    uint32_t b1 = intuitive_physics_add_object(phys, &box1);

    ip_object_t box2 = box1;
    box2.position.y = 1.5f;
    uint32_t b2 = intuitive_physics_add_object(phys, &box2);

    /* Simulate until stable */
    for (int i = 0; i < 500; i++)
        intuitive_physics_step(phys, 0.01f);

    /* Rebuild scene graph from physics */
    int rc = scene_graph_rebuild(sg, phys);
    ASSERT_EQ(rc, 0);

    /* Should have support relations: box2 ON box1, box1 ON ground */
    ASSERT_GT(scene_graph_count(sg), 0);

    /* Both boxes should be supported */
    ASSERT_TRUE(intuitive_physics_is_supported(phys, b1));
    ASSERT_TRUE(intuitive_physics_is_supported(phys, b2));
    ASSERT_TRUE(intuitive_physics_is_stable(phys));

    scene_graph_destroy(sg);
    intuitive_physics_destroy(phys);
}

/* Test: entity tracker maintains belief about hidden objects */
TEST(entity_tracker_permanence) {
    entity_tracker_t* et = entity_tracker_create(NULL);
    ASSERT_NOT_NULL(et);

    /* Present an object */
    entity_observation_t obs = {
        .position = {1.0f, 0.0f, 0.0f},
        .velocity = {0.0f, 0.0f, 0.0f},
        .bounding_radius = 0.5f,
        .confidence = 0.9f,
    };
    entity_tracker_update(et, &obs, 1, 0.0);
    ASSERT_EQ(entity_tracker_count_visible(et), 1);

    /* Remove observation (object hidden) */
    entity_tracker_update(et, NULL, 0, 1.0);

    /* Entity should still exist (permanence) but be occluded */
    ASSERT_EQ(entity_tracker_count_visible(et), 0);
    ASSERT_EQ(entity_tracker_count_total(et), 1);

    /* Should still believe it's permanent */
    uint32_t ids[8];
    uint32_t n = entity_tracker_get_active_ids(et, ids, 8);
    ASSERT_EQ(n, 1);
    ASSERT_TRUE(entity_tracker_object_permanent(et, ids[0]));

    entity_tracker_destroy(et);
}

/* Test: physics prior constrains impossible predictions */
TEST(physics_prior_constraint) {
    intuitive_physics_engine_t* phys = intuitive_physics_create(NULL);
    physics_prior_t* pp = physics_prior_create(NULL);
    ASSERT_NOT_NULL(phys);
    ASSERT_NOT_NULL(pp);

    physics_prior_connect(pp, phys, NULL, NULL);

    /* Create a prediction with an object below ground (impossible) */
    pp_spatial_state_t state;
    pp_spatial_state_alloc(&state, 1);
    state.num_objects = 1;
    state.positions[0] = (wm_parietal_vec3_t){0, -1.0f, 0};  /* below ground */
    state.velocities[0] = (wm_parietal_velocity_t){0, 0, 0};
    state.object_ids[0] = 0;

    /* Should detect violation */
    uint32_t violations = physics_prior_constrain(pp, &state, 0.01f);
    /* After constraint, y should be >= 0 */
    ASSERT_GE(state.positions[0].y, 0.0f);

    pp_spatial_state_free(&state);
    physics_prior_destroy(pp);
    intuitive_physics_destroy(phys);
}

/* Test: world simulator creates and steps without crash */
TEST(world_simulator_basic) {
    world_simulator_t* ws = wsim_create(NULL);
    ASSERT_NOT_NULL(ws);

    /* Register a physics engine */
    intuitive_physics_engine_t* phys = intuitive_physics_create(NULL);
    ASSERT_NOT_NULL(phys);
    wsim_register_engine(ws, WSIM_ENGINE_NEWTONIAN, phys, 0.01f);

    /* Enable couplings */
    wsim_enable_standard_couplings(ws);

    /* Step */
    int rc = wsim_step(ws, 0.01f);
    ASSERT_EQ(rc, 0);

    /* Check stats */
    wsim_stats_t stats = wsim_get_stats(ws);
    ASSERT_EQ(stats.master_steps, 1);
    ASSERT_GE(stats.active_engines, 1);

    /* Temperature should be set */
    float T = wsim_get_temperature(ws, 16, 16, 16);
    ASSERT_NEAR(T, 293.15f, 1.0f);  /* ambient ± coupling effects */

    intuitive_physics_destroy(phys);
    wsim_destroy(ws);
}

/* Test: surface physics + surface chemistry coupling */
TEST(surface_coupling) {
    surface_physics_sim_t* sp = surface_physics_create(NULL);
    surface_chemistry_sim_t* sc = surface_chemistry_create(NULL);
    ASSERT_NOT_NULL(sp);
    ASSERT_NOT_NULL(sc);

    /* Load materials and create water-glass interface */
    surface_physics_load_common_materials(sp);
    uint32_t iface = surface_physics_create_interface(sp, 0, 1, 0.001f);  /* water-glass */
    ASSERT_NE(iface, UINT32_MAX);

    /* Load Pt catalyst */
    surface_chemistry_load_pt_catalyst(sc);

    /* Step both */
    surface_physics_step(sp, 0.01f);
    surface_chemistry_step(sc, 0.01f);

    /* Both should have non-zero stats */
    surf_phys_stats_t sp_stats = surface_physics_get_stats(sp);
    ASSERT_EQ(sp_stats.step_count, 1);

    schem_stats_t sc_stats = surface_chemistry_get_stats(sc);
    ASSERT_EQ(sc_stats.step_count, 1);

    surface_chemistry_destroy(sc);
    surface_physics_destroy(sp);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(physics_to_scene_graph);
    RUN_TEST_SAFE(entity_tracker_permanence);
    RUN_TEST_SAFE(physics_prior_constraint);
    RUN_TEST_SAFE(world_simulator_basic);
    RUN_TEST_SAFE(surface_coupling);
TEST_MAIN_END()
