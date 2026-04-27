from __future__ import annotations

import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from fscd.graphs import adjacency_to_skeleton_upper, is_dag, pdag_to_dag_adjacency


def test_pdag_to_dag_preserves_skeleton_and_is_acyclic() -> None:
    nodes = [GraphNode(f"X{i}") for i in range(3)]
    pdag = GeneralGraph(nodes)
    pdag.add_edge(Edge(nodes[0], nodes[1], Endpoint.TAIL, Endpoint.TAIL))
    pdag.add_edge(Edge(nodes[1], nodes[2], Endpoint.TAIL, Endpoint.ARROW))

    dag = pdag_to_dag_adjacency(pdag)

    assert is_dag(dag)
    expected_skeleton = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(adjacency_to_skeleton_upper(dag), expected_skeleton)


def test_pdag_to_dag_handles_undirected_four_cycle() -> None:
    nodes = [GraphNode(f"X{i}") for i in range(5)]
    pdag = GeneralGraph(nodes)
    pdag.add_edge(Edge(nodes[0], nodes[3], Endpoint.TAIL, Endpoint.TAIL))
    pdag.add_edge(Edge(nodes[0], nodes[4], Endpoint.TAIL, Endpoint.TAIL))
    pdag.add_edge(Edge(nodes[2], nodes[3], Endpoint.TAIL, Endpoint.TAIL))
    pdag.add_edge(Edge(nodes[2], nodes[4], Endpoint.TAIL, Endpoint.TAIL))

    dag = pdag_to_dag_adjacency(pdag)

    assert is_dag(dag)
    expected_skeleton = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(adjacency_to_skeleton_upper(dag), expected_skeleton)
