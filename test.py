import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import SymGNN
import optax

N = 18
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=1/2, N=N)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)


# Disordered Heisenberg model (example)
J = [1.48546337,  1.42617448, -0.52427966, -0.35561696,  0.2325699,   0.4986678,
  1.86363023, -2.06921317, -1.2083311,  -0.29168546, -0.93841295, -0.00382017,
 -1.26814365,  0.30772702,  0.28159215,  0.2291237,   0.68785645,  0.61345415]
H = -sum(J[i]*sigmaz(hi, i)*sigmaz(hi, i+1) for i in range(N-1))
H += -0.5*sum(sigmax(hi, i) for i in range(N))
H += -J[-1]*sigmaz(hi, 0)*sigmaz(hi, N-1)

# SymGNN 1.04s/it, Energy=-16.4106 ± 0.0034 [σ²=0.0093, R̂=1.0183]]]
model = SymGNN(
graph = g,
couplings = tuple(J),
layers= 2,
features = 30,
use_attention=False,
output_phase=True
)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)
print(f'number of parameters: {vs.n_parameters}')
lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value=1e-2,
            peak_value=3e-2,
            warmup_steps=30,
            transition_steps=1,
            decay_rate=0.98,
        )
# Optimizer and VMC
sr = nk.optimizer.SR(diag_shift=1e-3)
opt = nk.optimizer.Sgd(learning_rate = lr_schedule)
gs = nk.VMC(H, opt, variational_state=vs, preconditioner=sr)

gs.run(300)

