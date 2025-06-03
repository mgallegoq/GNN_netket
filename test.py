import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import BatchGNN

N = 20
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=N)
model = BatchGNN(
    graph = g,
    layers=3,
    features=10,
    use_attention=True,
    output_phase=False
)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)

# Disordered Heisenberg model (example)
J = np.random.normal(0, 1, size=(g.n_edges,))
H = -sum(sigmaz(hi, i) for i in range(N))

# Optimizer and VMC
opt = nk.optimizer.Adam(learning_rate=1e-5)
gs = nk.VMC(H, opt, variational_state=vs)

gs.run(300)
