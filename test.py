import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import BatchGNN

N = 17
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=N)
model = BatchGNN(
    graph = g,
    layers= 2,
    features = 30,
    use_attention=True,
    output_phase=False
)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)
print(f'number of parameters: {vs.n_parameters}')
# Disordered Heisenberg model (example)
H = -sum(sigmaz(hi, i)*sigmaz(hi, i+1) for i in range(N-1))
H += -0.5*sum(sigmax(hi, i) for i in range(N))

# RBM Energy=-17.1757+0.0000j ± 0.0080 σ²=0.0646

# Optimizer and VMC
opt = nk.optimizer.Adam(learning_rate = 1e-3)
gs = nk.VMC(H, opt, variational_state=vs)

gs.run(300)

