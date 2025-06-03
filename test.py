import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import BatchGNN

N = 17
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=N)
model = BatchGNN(
    graph = g,
    layers= 3,
    features = 50,
    use_attention=True,
    output_phase=False
)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)

# Disordered Heisenberg model (example)
H = -sum(sigmaz(hi, i) for i in range(N))

# Optimizer and VMC
opt = nk.optimizer.Adam(learning_rate=1e-4)
gs = nk.VMC(H, opt, variational_state=vs)

gs.run(1000)
