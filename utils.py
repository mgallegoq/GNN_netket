"""
This module provides utility functions for constructing and analyzing directed graphs
from weighted adjacency matrices. It includes:

- Selection of the top-M strongest directed edges.
- Checks for strong and undirected connectivity using NetworkX.
- Visualization tools for directed graphs.

Typical use case involves working with asymmetric (directed) connectivity matrices
in scientific computing or machine learning applications.
"""

import time
from typing import Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def top_m_edges_directed(weights: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects the top-m strongest directed edges from a weighted adjacency matrix.

    Args:
        weights (np.ndarray): A square 2D array of shape (n, n) where J[i, j] is
        the edge weight from i to j.
        m (int): number of strongest edges to retain.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - edges: Array of shape (m, 2) with selected (sender, receiver) node pairs.
            - weights: Array of shape (m,) with corresponding edge weights.
    """
    n = weights.shape[0]
    assert weights.shape == (n, n)

    all_senders, all_receivers = np.nonzero(~np.eye(n, dtype=bool))
    strengths = np.abs(weights[all_senders, all_receivers])

    top_indices = np.argpartition(-strengths, m)[:m]
    top_edges = np.stack([all_senders[top_indices], all_receivers[top_indices]], axis=1)
    top_weights = weights[all_senders[top_indices], all_receivers[top_indices]]

    return top_edges, top_weights


def is_connected(n: int, edges: np.ndarray) -> bool:
    """
    Checks whether a directed graph is strongly connected using networkX.

    Args:
        n (int): number of nodes in the graph.
        edges (np.ndarray): Array of shape (E, 2) representing directed edges (sender, receiver).

    Returns:
        bool: True if the graph is strongly connected, False otherwise.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    return nx.is_connected(g)


def plot_directed_graph(n: int, edges: np.ndarray, edge_weights: np.ndarray = None):
    """
    Plots a directed graph given node count and a list of edges.

    Args:
        n (int): number of nodes.
        edges (np.ndarray): Array of shape (E, 2) with directed edges (sender, receiver).
        edge_weights (np.ndarray, optional): Array of shape (E,) with edge weights to display.
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

def get_connected_graph(weights: np.ndarray, min_edges: int = 2):
    n = weights.shape[0]
    max_edges = min_edges
    while True:
        reduced_edges, reduced_weights = top_m_edges_directed(weights, max_edges)
        if is_connected(n, reduced_edges):
            break
        max_edges += 1
    return ReducedGraph(reduced_edges), reduced_weights

class ReducedGraph():
    edges_value: Any
    def __init__(self, edges_value) -> None:
        self.edges_value = edges_value
    def edges(self):
        return self.edges_value

