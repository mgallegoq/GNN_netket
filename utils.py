"""
Graph Utilities for Directed Weighted Connectivity Analysis

This module provides utility functions for constructing, analyzing, and visualizing
directed graphs derived from weighted adjacency matrices. It is intended for applications
in scientific computing and machine learning that involve asymmetric connectivity structures.

Key functionalities include:

- Extraction of the top-M strongest directed edges based on edge weight magnitude.
- Connectivity checks in both directed and undirected contexts using NetworkX.
- Visualization of directed graphs with optional edge weight annotations.
"""

from typing import Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import netket as nk


def top_m_edges_directed(weights: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the top-m strongest directed edges from a weighted adjacency matrix.

    Args:
        weights (np.ndarray): A square 2D array of shape (n, n), where weights[i, j]
                              denotes the weight of the directed edge from node i to node j.
        m (int): The number of strongest edges to select, based on absolute weight.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - edges: An array of shape (m, 2) containing (source, target) node pairs.
            - weights: An array of shape (m,) with the corresponding edge weights.
    """
    n = weights.shape[0]
    assert weights.shape == (n, n)

    senders, receivers = np.nonzero(~np.eye(n, dtype=bool))
    strengths = np.abs(weights[senders, receivers])

    top_indices = np.argpartition(-strengths, m)[:m]
    top_edges = np.stack([senders[top_indices], receivers[top_indices]], axis=1)
    top_weights = weights[senders[top_indices], receivers[top_indices]]

    return top_edges, top_weights


def is_connected(n: int, edges: np.ndarray) -> bool:
    """
    Checks whether the undirected version of a graph is connected.

    Args:
        n (int): The number of nodes in the graph.
        edges (np.ndarray): An array of shape (E, 2) representing directed edges.

    Returns:
        bool: True if the corresponding undirected graph is connected; False otherwise.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return nx.is_connected(g)


def plot_directed_graph(n: int, edges: np.ndarray, edge_weights: np.ndarray = None):
    """
    Visualizes a directed graph using a spring layout.

    Args:
        n (int): The number of nodes in the graph.
        edges (np.ndarray): An array of shape (E, 2) representing directed edges.
        edge_weights (np.ndarray, optional): An array of shape (E,) with edge weights
                                             to annotate the graph.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    pos = nx.spring_layout(g, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(g, pos, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(g, pos, font_size=12)

    if edge_weights is not None:
        edge_labels = {tuple(edge): f"{w:.2f}" for edge, w in zip(edges, edge_weights)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Directed Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("graph_test.png")


def get_connected_graph(weights: np.ndarray, min_edges: int = 2) -> Tuple["ReducedGraph", np.ndarray]:
    """
    Iteratively selects the smallest set of strongest edges that results in a connected undirected graph.

    Args:
        weights (np.ndarray): A square 2D array representing the weighted adjacency matrix.
        min_edges (int): Minimum number of edges to start the connectivity search.

    Returns:
        Tuple[ReducedGraph, np.ndarray]:
            - ReducedGraph: An object encapsulating the selected edge set.
            - np.ndarray: The associated edge weights.
    """
    n = weights.shape[0]
    max_edges = min_edges
    while True:
        reduced_edges, reduced_weights = top_m_edges_directed(weights, max_edges)
        if is_connected(n, reduced_edges):
            break
        max_edges += 1
    return ReducedGraph(reduced_edges), reduced_weights


class ReducedGraph:
    """
    A lightweight container class for storing and exposing a reduced edge set.

    Attributes:
        _edges (Any): Internal storage of the edge array.
    """

    def __init__(self, edges: Any) -> None:
        self._edges = edges
        self.n_nodes = int(np.max(self._edges)) + 1
    
    def edges(self) -> Any:
        """
        Returns the stored edges.

        Returns:
            Any: The array of edges.
        """
        return self._edges
    def adjacency_matrix(self) -> Any:
        adj = np.zeros((self.n_nodes, self.n_nodes))
        for i, j in self._edges:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj   
