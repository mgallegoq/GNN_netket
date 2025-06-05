import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import SymGNN
from utils import get_connected_graph
import optax

N = 18
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=1/2, N=N)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)


# Disordered Heisenberg model (example)
J = np.load('reference_couplings.npy')
g, reduced_weights = get_connected_graph(J, min_edges=N // 10)
H = -sum(sum(J[i, j]*sigmaz(hi, i)*sigmaz(hi, j) for i in range(N)) for j in range(N))
H += -0.5*sum(sigmax(hi, i) for i in range(N))

# SymGNN 2.45s/it, Energy=-66.04 ± 0.0034 [σ²=0.0033, R̂=1.0123]]]
model = SymGNN(
graph = g,
couplings = tuple(reduced_weights),
layers= 3,
features = 20,
use_attention=False,
output_phase=True
)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)
print(f'number of parameters: {vs.n_parameters}')
lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value=1e-3,
            peak_value=1e-2,
            warmup_steps=30,
            transition_steps=1,
            decay_rate=0.95,
        )
# Optimizer and VMC
sr = nk.optimizer.SR(diag_shift=1e-3)
opt = nk.optimizer.Sgd(learning_rate = lr_schedule)
gs = nk.VMC(H, opt, variational_state=vs, preconditioner=sr)

gs.run(300)

