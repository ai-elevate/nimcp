/**
 * @file test_topology.c
 * @brief Tests for the computational topology engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_topology.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    topology_t *t = topo_create();
    ASSERT_NOT_NULL(t);
    topo_destroy(t);
}

TEST(complex_create_destroy) {
    simplicial_complex_t *sc = topo_complex_create(10);
    ASSERT_NOT_NULL(sc);
    ASSERT_EQ(sc->num_vertices, 10);
    topo_complex_destroy(sc);
}

/* ---------- Euler characteristic: sphere = 2 ---------- */

TEST(euler_char_sphere) {
    /* Minimal triangulation of a sphere (tetrahedron boundary):
       4 vertices, 6 edges, 4 triangles => chi = 4-6+4 = 2 */
    simplicial_complex_t *sc = topo_complex_create(4);
    ASSERT_NOT_NULL(sc);

    /* 6 edges */
    topo_complex_add_edge(sc, 0, 1);
    topo_complex_add_edge(sc, 0, 2);
    topo_complex_add_edge(sc, 0, 3);
    topo_complex_add_edge(sc, 1, 2);
    topo_complex_add_edge(sc, 1, 3);
    topo_complex_add_edge(sc, 2, 3);

    /* 4 triangles */
    topo_complex_add_triangle(sc, 0, 1, 2);
    topo_complex_add_triangle(sc, 0, 1, 3);
    topo_complex_add_triangle(sc, 0, 2, 3);
    topo_complex_add_triangle(sc, 1, 2, 3);

    int32_t chi = topo_euler_characteristic(sc);
    ASSERT_EQ(chi, 2);

    topo_complex_destroy(sc);
}

/* ---------- Euler characteristic: torus = 0 ---------- */

TEST(euler_char_torus) {
    /* Minimal torus triangulation: 7 vertices, 21 edges, 14 triangles
       chi = 7 - 21 + 14 = 0 */
    simplicial_complex_t *sc = topo_complex_create(7);
    ASSERT_NOT_NULL(sc);

    /* Standard 7-vertex torus triangulation (Moebius-Kantor) */
    int tris[][3] = {
        {0,1,3}, {1,3,4}, {1,2,4}, {2,4,5}, {2,0,5}, {0,5,3},
        {3,4,6}, {4,6,0}, {4,5,0}, {5,0,1}, {5,3,1}, {3,6,1},
        {6,0,2}, {6,2,1}
    };
    /* Add all edges from triangles first */
    for (int t = 0; t < 14; t++) {
        topo_complex_add_edge(sc, tris[t][0], tris[t][1]);
        topo_complex_add_edge(sc, tris[t][1], tris[t][2]);
        topo_complex_add_edge(sc, tris[t][0], tris[t][2]);
    }
    /* Add triangles */
    for (int t = 0; t < 14; t++) {
        topo_complex_add_triangle(sc, tris[t][0], tris[t][1], tris[t][2]);
    }

    int32_t chi = topo_euler_characteristic(sc);
    ASSERT_EQ(chi, 0);

    topo_complex_destroy(sc);
}

/* ---------- Betti_0 = number of connected components ---------- */

TEST(betti_0_disconnected) {
    /* 4 vertices, 2 edges: {0-1} and {2-3} => 2 components */
    simplicial_complex_t *sc = topo_complex_create(4);
    ASSERT_NOT_NULL(sc);
    topo_complex_add_edge(sc, 0, 1);
    topo_complex_add_edge(sc, 2, 3);

    homology_result_t hom = topo_compute_homology(sc);
    ASSERT_EQ(hom.betti[0], 2);

    topo_complex_destroy(sc);
}

TEST(betti_0_single_component) {
    /* Triangle: 3 vertices, 3 edges, 1 triangle => 1 component */
    simplicial_complex_t *sc = topo_complex_create(3);
    ASSERT_NOT_NULL(sc);
    topo_complex_add_edge(sc, 0, 1);
    topo_complex_add_edge(sc, 1, 2);
    topo_complex_add_edge(sc, 0, 2);
    topo_complex_add_triangle(sc, 0, 1, 2);

    homology_result_t hom = topo_compute_homology(sc);
    ASSERT_EQ(hom.betti[0], 1);

    topo_complex_destroy(sc);
}

/* ---------- connected components ---------- */

TEST(connected_components_count) {
    /* 6 vertices: {0,1,2} connected, {3,4} connected, {5} isolated => 3 */
    simplicial_complex_t *sc = topo_complex_create(6);
    ASSERT_NOT_NULL(sc);
    topo_complex_add_edge(sc, 0, 1);
    topo_complex_add_edge(sc, 1, 2);
    topo_complex_add_edge(sc, 3, 4);

    connected_components_t cc = topo_connected_components(sc);
    ASSERT_EQ(cc.num_components, 3);

    topo_complex_destroy(sc);
}

/* ---------- simplicial complex construction ---------- */

TEST(complex_add_simplices) {
    simplicial_complex_t *sc = topo_complex_create(5);
    ASSERT_NOT_NULL(sc);

    ASSERT_TRUE(topo_complex_add_edge(sc, 0, 1));
    ASSERT_TRUE(topo_complex_add_edge(sc, 1, 2));
    ASSERT_TRUE(topo_complex_add_edge(sc, 0, 2));
    ASSERT_EQ(sc->num_edges, 3);

    ASSERT_TRUE(topo_complex_add_triangle(sc, 0, 1, 2));
    ASSERT_EQ(sc->num_triangles, 1);

    topo_complex_destroy(sc);
}

/* ---------- Betti_1: loop detection ---------- */

TEST(betti_1_loop) {
    /* Square with no diagonal: 4 vertices, 4 edges, 0 triangles
       => Betti_1 = 1 (one independent loop) */
    simplicial_complex_t *sc = topo_complex_create(4);
    ASSERT_NOT_NULL(sc);
    topo_complex_add_edge(sc, 0, 1);
    topo_complex_add_edge(sc, 1, 2);
    topo_complex_add_edge(sc, 2, 3);
    topo_complex_add_edge(sc, 3, 0);

    homology_result_t hom = topo_compute_homology(sc);
    ASSERT_EQ(hom.betti[0], 1);  /* connected */
    ASSERT_EQ(hom.betti[1], 1);  /* one loop */

    topo_complex_destroy(sc);
}

/* ---------- metric topology: open ball ---------- */

TEST(metric_open_ball) {
    topo_point_set_t ps;
    memset(&ps, 0, sizeof(ps));
    ps.num_points = 3;
    ps.dim = 2;
    ps.metric = TOPO_METRIC_EUCLIDEAN;

    /* p0 = (0,0), p1 = (1,0), p2 = (10,0) */
    ps.coords[0][0] = 0.0; ps.coords[0][1] = 0.0;
    ps.coords[1][0] = 1.0; ps.coords[1][1] = 0.0;
    ps.coords[2][0] = 10.0; ps.coords[2][1] = 0.0;

    /* B(p0, 2.0): p1 is inside (dist=1), p2 is not (dist=10) */
    ASSERT_TRUE(topo_in_open_ball(&ps, 0, 1, 2.0));
    ASSERT_FALSE(topo_in_open_ball(&ps, 0, 2, 2.0));
}

/* ---------- Euler characteristic of single edge ---------- */

TEST(euler_char_edge) {
    /* 1 edge, 2 vertices => chi = 2 - 1 = 1 */
    simplicial_complex_t *sc = topo_complex_create(2);
    ASSERT_NOT_NULL(sc);
    topo_complex_add_edge(sc, 0, 1);

    int32_t chi = topo_euler_characteristic(sc);
    ASSERT_EQ(chi, 1);

    topo_complex_destroy(sc);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(complex_create_destroy);
    RUN_TEST_SAFE(euler_char_sphere);
    RUN_TEST_SAFE(euler_char_torus);
    RUN_TEST_SAFE(betti_0_disconnected);
    RUN_TEST_SAFE(betti_0_single_component);
    RUN_TEST_SAFE(connected_components_count);
    RUN_TEST_SAFE(complex_add_simplices);
    RUN_TEST_SAFE(betti_1_loop);
    RUN_TEST_SAFE(metric_open_ball);
    RUN_TEST_SAFE(euler_char_edge);
TEST_MAIN_END()
