import netket as nk
import numpy as np

from models import GraphAttentionGNN

N = 20
g = nk.graph.Hypercube(length=N, n_dim=1)
print(g.n_nodes)
hi = nk.hilbert.Spin(s=0.5, N=N)
model = GraphAttentionGNN(
    graph = g,
    layers=3,
    features=N,
    use_attention=True,
    output_phase=False
)

# Sampler
sa = nk.sampler.MetropolisLocal(hi)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)

# Disordered Heisenberg model (example)
J = np.random.normal(0, 1, size=(g.n_edges,))
H = nk.operator.Heisenberg(hi, graph=g, J=1)

# Optimizer and VMC
opt = nk.optimizer.Adam(learning_rate=1e-3)
gs = nk.VMC(H, opt, variational_state=vs)

gs.run(300)
