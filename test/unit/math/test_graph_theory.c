/**
 * @file test_graph_theory.c
 * @brief Tests for the graph theory engine
 */

#include "../../test_framework.h"
#include "cognitive/math/nimcp_graph_theory.h"

/* ---------- lifecycle ---------- */

TEST(create_destroy) {
    graph_t *g = gt_create(5, false, false);
    ASSERT_NOT_NULL(g);
    ASSERT_EQ(g->num_vertices, 5);
    ASSERT_FALSE(g->directed);
    gt_destroy(g);
}

/* ---------- edge operations ---------- */

TEST(add_edge) {
    graph_t *g = gt_create(4, false, true);
    ASSERT_TRUE(gt_add_edge(g, 0, 1, 3.0));
    ASSERT_TRUE(gt_has_edge(g, 0, 1));
    ASSERT_TRUE(gt_has_edge(g, 1, 0));  /* undirected */
    ASSERT_NEAR(gt_edge_weight(g, 0, 1), 3.0, 1e-9);
    gt_destroy(g);
}

/* ---------- BFS shortest path ---------- */

TEST(bfs_shortest_path) {
    /* 0-1-2-3 path graph, also 0-3 direct edge */
    graph_t *g = gt_create(4, false, false);
    gt_add_edge(g, 0, 1, 1.0);
    gt_add_edge(g, 1, 2, 1.0);
    gt_add_edge(g, 2, 3, 1.0);
    gt_add_edge(g, 0, 3, 1.0); /* shortcut */

    gt_traversal_t bfs = gt_bfs(g, 0);
    ASSERT_EQ(bfs.count, 4);

    /* BFS parent of 3 should be 0 (direct edge) */
    ASSERT_EQ(bfs.parent[3], 0);
    gt_destroy(g);
}

/* ---------- Dijkstra ---------- */

TEST(dijkstra_weighted) {
    /* 0->1 (1), 0->2 (4), 1->2 (2), 2->3 (1) */
    graph_t *g = gt_create(4, true, true);
    gt_add_edge(g, 0, 1, 1.0);
    gt_add_edge(g, 0, 2, 4.0);
    gt_add_edge(g, 1, 2, 2.0);
    gt_add_edge(g, 2, 3, 1.0);

    gt_shortest_path_t sp = gt_dijkstra(g, 0);
    ASSERT_NEAR(sp.dist[0], 0.0, 1e-9);
    ASSERT_NEAR(sp.dist[1], 1.0, 1e-9);
    ASSERT_NEAR(sp.dist[2], 3.0, 1e-9);   /* 0->1->2 = 3, not 0->2 = 4 */
    ASSERT_NEAR(sp.dist[3], 4.0, 1e-9);   /* 0->1->2->3 = 4 */

    gt_destroy(g);
}

/* ---------- Kruskal MST ---------- */

TEST(kruskal_mst) {
    /*  0-1 (1), 0-2 (3), 1-2 (2), 1-3 (4), 2-3 (5)
        MST edges: 0-1(1), 1-2(2), 1-3(4) => total = 7 */
    graph_t *g = gt_create(4, false, true);
    gt_add_edge(g, 0, 1, 1.0);
    gt_add_edge(g, 0, 2, 3.0);
    gt_add_edge(g, 1, 2, 2.0);
    gt_add_edge(g, 1, 3, 4.0);
    gt_add_edge(g, 2, 3, 5.0);

    gt_mst_result_t mst = gt_kruskal(g);
    ASSERT_EQ(mst.num_edges, 3);
    ASSERT_NEAR(mst.total_weight, 7.0, 1e-9);

    gt_destroy(g);
}

/* ---------- chromatic number of K4 ---------- */

TEST(chromatic_number_k4) {
    /* Complete graph K4: every pair connected => chi(K4) = 4 */
    graph_t *g = gt_create(4, false, false);
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            gt_add_edge(g, i, j, 1.0);

    gt_coloring_t col = gt_chromatic_number(g);
    ASSERT_EQ(col.num_colors, 4);
    ASSERT_TRUE(col.optimal);

    gt_destroy(g);
}

/* ---------- bipartite detection ---------- */

TEST(bipartite_detection) {
    /* Even cycle is bipartite */
    graph_t *g = gt_create(4, false, false);
    gt_add_edge(g, 0, 1, 1.0);
    gt_add_edge(g, 1, 2, 1.0);
    gt_add_edge(g, 2, 3, 1.0);
    gt_add_edge(g, 3, 0, 1.0);
    ASSERT_TRUE(gt_is_bipartite(g));
    gt_destroy(g);

    /* Odd cycle (triangle) is not bipartite */
    graph_t *g2 = gt_create(3, false, false);
    gt_add_edge(g2, 0, 1, 1.0);
    gt_add_edge(g2, 1, 2, 1.0);
    gt_add_edge(g2, 2, 0, 1.0);
    ASSERT_FALSE(gt_is_bipartite(g2));
    gt_destroy(g2);
}

/* ---------- max flow ---------- */

TEST(max_flow_simple) {
    /* s=0, t=3: 0->1(10), 0->2(10), 1->3(10), 2->3(10), 1->2(1) */
    /* Max flow = 20 */
    graph_t *g = gt_create(4, true, true);
    gt_add_edge(g, 0, 1, 10.0);
    gt_add_edge(g, 0, 2, 10.0);
    gt_add_edge(g, 1, 3, 10.0);
    gt_add_edge(g, 2, 3, 10.0);
    gt_add_edge(g, 1, 2, 1.0);

    gt_flow_result_t flow = gt_max_flow(g, 0, 3);
    ASSERT_NEAR(flow.max_flow, 20.0, 1e-9);

    gt_destroy(g);
}

/* ---------- connected components ---------- */

TEST(connected_components) {
    /* Two disconnected pairs: {0,1} and {2,3} */
    graph_t *g = gt_create(4, false, false);
    gt_add_edge(g, 0, 1, 1.0);
    gt_add_edge(g, 2, 3, 1.0);

    gt_components_t cc = gt_connected_components(g);
    ASSERT_EQ(cc.num_components, 2);
    ASSERT_EQ(cc.component[0], cc.component[1]);
    ASSERT_EQ(cc.component[2], cc.component[3]);
    ASSERT_NE(cc.component[0], cc.component[2]);

    gt_destroy(g);
}

/* ---------- DFS traversal ---------- */

TEST(dfs_traversal) {
    graph_t *g = gt_create(4, false, false);
    gt_add_edge(g, 0, 1, 1.0);
    gt_add_edge(g, 0, 2, 1.0);
    gt_add_edge(g, 1, 3, 1.0);

    gt_traversal_t dfs = gt_dfs(g, 0);
    ASSERT_EQ(dfs.count, 4);
    ASSERT_EQ(dfs.order[0], 0);  /* source is visited first */

    gt_destroy(g);
}

TEST_MAIN_BEGIN()
    RUN_TEST_SAFE(create_destroy);
    RUN_TEST_SAFE(add_edge);
    RUN_TEST_SAFE(bfs_shortest_path);
    RUN_TEST_SAFE(dijkstra_weighted);
    RUN_TEST_SAFE(kruskal_mst);
    RUN_TEST_SAFE(chromatic_number_k4);
    RUN_TEST_SAFE(bipartite_detection);
    RUN_TEST_SAFE(max_flow_simple);
    RUN_TEST_SAFE(connected_components);
    RUN_TEST_SAFE(dfs_traversal);
TEST_MAIN_END()
