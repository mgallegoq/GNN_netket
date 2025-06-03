"""
Graph Neural Network with Attention for NetKet

This module defines a flexible graph neural network (GNN) architecture with
optional attention mechanisms for use as a variational ansatz in quantum
many-body problems using the NetKet library. It is implemented using Flax
and JAX and is compatible with NetKet ≥3.10.

Classes:
    - AttentionGNNLayer: Single message-passing layer with optional attention.
    - GraphAttentionGNN: Full GNN model with stacked attention layers that maps
                         spin configurations to log wavefunction amplitudes.
"""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class AttentionGNNLayer(nn.Module):
    """
    Single GNN message-passing layer with optional self-attention.

    Attributes:
        out_features (int): Number of output features per node.
        use_attention (bool): Whether to use an attention mechanism on edges.

    Inputs:
        h (jnp.ndarray): Node features of shape (n_nodes, features).
        senders (jnp.ndarray): Indices of sending nodes for edges.
        receivers (jnp.ndarray): Indices of receiving nodes for edges.

    Returns:
        jnp.ndarray: Updated node features of shape (n_nodes, out_features).
    """

    out_features: int
    senders: Any
    receivers: Any
    use_attention: bool = True

    @nn.compact
    def __call__(self, h):
        w = self.param(
            "W", nn.initializers.xavier_uniform(), (h.shape[-1], self.out_features)
        )
        assert h.ndim == 2, f"Expected h to be 2D, got shape {h.shape}"
        h_proj = jnp.dot(h, w)

        messages = h_proj[:, self.senders]

        if self.use_attention:
            q = nn.Dense(self.out_features)(h)
            k = nn.Dense(self.out_features)(h)
            a = jnp.dot(q, k.transpose())

            messages = messages * a[..., None]
        print(h, messages)
        aggregated = jax.ops.segment_sum(messages, self.receivers, h.shape[0])
        return nn.relu(aggregated)


class GraphAttentionGNN(nn.Module):
    """
    Graph Neural Network with optional attention layers for NetKet.

    This GNN takes spin configurations and computes the log-amplitude (and optionally
    the phase) of the quantum wavefunction defined on a graph.

    Attributes:
        graph (Any): NetKet graph object with .senders and .receivers fields.
        layers (int): Number of GNN layers to stack.
        features (int): Number of hidden features per layer.
        use_attention (bool): Whether to use attention in each layer.
        output_phase (bool): Whether to return a complex wavefunction (log_amp + i * phase).

    Inputs:
        x (jnp.ndarray): Spin configurations of shape (n_samples, n_spins) or (n_spins,).

    Returns:
        jnp.ndarray: Logarithm of wavefunction amplitude, or complex log amplitude
                     if output_phase is True.
    """

    graph: Any
    layers: int = 1
    features: int = 64
    use_attention: bool = True
    output_phase: bool = True

    @nn.compact
    def __call__(self, x):

        x = x.astype(jnp.float32)
        batch = x.ndim == 2
        h = x if batch else x[None, :]
        senders = jnp.concatenate(
            (jnp.array(self.graph.edges())[:, 0], jnp.array(self.graph.edges())[:, 1])
        )
        receivers = jnp.concatenate(
            (jnp.array(self.graph.edges())[:, 1], jnp.array(self.graph.edges())[:, 0])
        )

        worker = AttentionGNNLayer(
            self.features, senders, receivers, self.use_attention
        )
        h = worker(h)

        h_sum = jnp.sum(h, axis=1 if batch else 0)
        log_amp = nn.Dense(1)(h_sum).squeeze(-1)

        if self.output_phase:
            phase = nn.Dense(1)(h_sum).squeeze(-1)
            return log_amp + 1j * phase
        return log_amp
