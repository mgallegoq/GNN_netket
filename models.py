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


import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import logsumexp
from typing import Sequence, Callable, Any
from jax.scipy.special import logsumexp

REAL_DTYPE = jnp.asarray(1.0).dtype


class MLP(nn.Module):
    """
    A simple single-layer Multilayer Perceptron (MLP) with ReLU activation.

    Attributes:
        features (int): Number of output features for the dense layer.

    Methods:
        __call__(x): Applies a Dense layer followed by a ReLU activation.
    """

    features: int

    @nn.compact
    def __call__(self, x):
        return nn.relu(nn.Dense(self.features)(x))


class FFN(nn.Module):
    """
    Feed Forward Network
    alpha represents the number of hidden units in each layer and mu the number of layers
    The parameters (weights and biases) are defined as complex.
    """
    alpha: int = 1
    mu: int = 1
    output_size: int = 0

    @nn.compact
    def __call__(self, x):
        for _ in range(self.mu):
            x = nn.Dense(
                features=self.alpha * x.shape[-1],
                param_dtype=REAL_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.1),
                bias_init=nn.initializers.normal(stddev=0.1),
            )(x)
            x = nn.selu(x)
        if self.output_size == 0:
            return x
        else:
            x = nn.Dense(
                features=self.output_size,
                param_dtype=REAL_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.1),
                bias_init=nn.initializers.normal(stddev=0.1),
            )(x)
            return nn.selu(x)


class BatchedFFN(nn.Module):
    """
    Applies a feedforward network (FFN) to a batch of input vectors using JAX vectorization.

    Attributes:
        alpha (int): Scaling factor for the width of hidden layers in the FFN.
        mu (int): Number of hidden layers in the FFN.

    Methods:
        __call__(batched_x): Applies the FFN to each input in the batch independently.
    """

    alpha: int = 1
    mu: int = 1

    @nn.compact
    def __call__(self, batched_x):
        worker = FFN(alpha=self.alpha, mu=self.mu)

        return jax.vmap(worker, in_axes=0)(batched_x)


class FFNClassifier(nn.Module):
    """
    Feed-Forward Network (FFN) for binary classification.
    Outputs +1 or -1 using a final sign activation.

    Attributes:
        alpha (int): Width multiplier for hidden layers.
        mu (int): Number of hidden layers.
    """

    alpha: int = 1
    mu: int = 1

    @nn.compact
    def __call__(self, x):
        for _ in range(self.mu):
            x = nn.Dense(
                features=self.alpha * x.shape[-1],
                param_dtype=REAL_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.1),
                bias_init=nn.initializers.normal(stddev=0.1),
            )(x)
            x = nn.selu(x)
        # Final classification layer
        x = nn.Dense(
            features=2,
            param_dtype=REAL_DTYPE,
            kernel_init=nn.initializers.normal(stddev=0.1),
            bias_init=nn.initializers.normal(stddev=0.1),
        )(x)
        x = nn.softmax(x)
        return jnp.squeeze(x[..., 1] * jnp.pi)


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


import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import logsumexp
from typing import Sequence, Callable, Any
from jax.scipy.special import logsumexp

REAL_DTYPE = jnp.asarray(1.0).dtype



class AttentionGNNLayer(nn.Module):
    """
    Single GNN message-passing layer with optional self-attention.

    Attributes:
        out_features (int): Number of output features per node.
        use_attention (bool): Whether to use an attention mechanism on edges.

    Inputs:
        h (jnp.ndarray): input of shape (n_batch, n_nodes).
        senders (jnp.ndarray): Indices of sending nodes for edges.
        receivers (jnp.ndarray): Indices of receiving nodes for edges.

    Returns:
        jnp.ndarray: Updated node features of shape (n_nodes, out_features).
    """

    out_features: int
    couplings: Sequence
    senders: Any
    receivers: Any
    layers: int = 1

    use_attention: bool = True
    n_embd = 5

    @nn.compact
    def __call__(self, h_embd):
        for _ in range(self.layers):
            receiver_features = h_embd[self.receivers, :]  # shape (num_edges, n_embd)
            sender_features = h_embd[self.senders, :]  # same shape
            mod_couplings = jnp.concatenate(
                (jnp.array(self.couplings), jnp.array(self.couplings))
            )[
                :, None
            ]  # (num_edges, 1)
            edge_features = jnp.concatenate(
                [receiver_features, sender_features, mod_couplings], axis=-1
            )  # (num_edges, 2 * n_embd)

            # Apply MLP once on whole array:
            messages = FFN(1, 1, self.out_features)(edge_features)  # (num_edges, out_features)
            if self.use_attention:
                q = nn.Dense(self.out_features)(sender_features)
                k = nn.Dense(self.out_features)(receiver_features)
                a = jnp.sum(q * k, axis=-1, keepdims=True)  # shape: (n_edges, 1)
                messages = messages * nn.softmax(a)  # or use softmax over edges

            aggregated = self.aggregate_messages(h_embd, messages, self.receivers)
            h_embd = aggregated

        return nn.relu(aggregated)

    def aggregate_messages(self, h_batch, messages_batch, receivers):
        """
        Aggregates messages for each node by summing over incoming messages.

        Args:
            h_batch (jnp.ndarray): Node features of shape (num_nodes, feature_dim).
            messages_batch (jnp.ndarray): Message features corresponding to edges.
            receivers (jnp.ndarray): Indices indicating the receiving node for each message.

        Returns:
            jnp.ndarray: Aggregated node features after message passing.
        """
        return jax.ops.segment_sum(
            messages_batch, receivers, num_segments=h_batch.shape[0]
        )


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
    couplings: Sequence
    layers: int = 1
    features: int = 64
    use_attention: bool = True
    output_phase: bool = True
    n_embd = 5

    @nn.compact
    def __call__(self, h):
        h = (h + 1) // 2
        h = h.astype(int)
        h_embd = nn.Embed(2, self.n_embd)(h)
        senders = jnp.concatenate(
            (jnp.array(self.graph.edges())[:, 0], jnp.array(self.graph.edges())[:, 1])
        )
        receivers = jnp.concatenate(
            (jnp.array(self.graph.edges())[:, 1], jnp.array(self.graph.edges())[:, 0])
        )
        h_embd = AttentionGNNLayer(
            self.features,
            self.couplings,
            senders,
            receivers,
            layers=self.layers,
            use_attention=self.use_attention,
        )(h_embd)
        h_sum = jnp.sum(h_embd, axis=1).squeeze()
        log_amp = jnp.sum(FFN(1, 2)(h_sum))
        if self.output_phase:
            phase = FFNClassifier()(h)
            return log_amp + 1j * phase
        return log_amp


class BatchGNN(nn.Module):
    """
    Applies a GraphAttentionGNN model to a batch of spin configurations using JAX vectorization.

    Attributes:
        graph (Any): A NetKet-compatible graph structure.
        couplings (Sequence): Coupling values for the edges.
        layers (int): Number of GNN layers.
        features (int): Number of output features per GNN layer.
        use_attention (bool): Whether to use attention in message passing.
        output_phase (bool): Whether to compute a phase output.

    Methods:
        __call__(x): Applies the GraphAttentionGNN over a batch of configurations.
    """

    graph: Any
    couplings: Sequence
    layers: int = 1
    features: int = 64
    use_attention: bool = True
    output_phase: bool = True

    @nn.compact
    def __call__(self, x):
        worker = GraphAttentionGNN(
            self.graph,
            self.couplings,
            self.layers,
            self.features,
            self.use_attention,
            self.output_phase,
        )
        return jax.vmap(worker, in_axes=0)(x)


class SymGNN(nn.Module):
    """
    Symmetrized Graph Neural Network (GNN) that applies a GNN to both input and its inversion,
    and combines them based on a symmetry (trivial or sign-changing).

    Attributes:
        graph (Any): A NetKet-compatible graph structure.
        couplings (Sequence): Coupling values for the edges.
        layers (int): Number of GNN layers.
        features (int): Number of output features per GNN layer.
        use_attention (bool): Whether to use attention in message passing.
        output_phase (bool): Whether to compute a phase output.
        trivial (bool): If True, performs symmetric averaging; else antisymmetric combination.

    Methods:
        __call__(x): Computes the log-amplitudes (and optionally phase) with symmetry enforcement.
    """

    graph: Any
    couplings: Sequence
    layers: int = 1
    features: int = 64
    use_attention: bool = True
    output_phase: bool = True
    trivial: bool = True

    @nn.compact
    def __call__(self, x) -> Any:
        model = BatchGNN(
            self.graph,
            self.couplings,
            self.layers,
            self.features,
            self.use_attention,
            self.output_phase,
        )

        output_x = model(x)
        output_inv_x = model(-1 * x)
        if self.trivial:
            return logsumexp(jnp.array([output_x, output_inv_x]), axis=0)
        return logsumexp(
            jnp.array([output_x, output_inv_x]),
            b=jnp.asarray(
                [jnp.ones(output_x.shape[-1]), -jnp.ones(output_x.shape[-1])]
            ),
            axis=0,
        )



