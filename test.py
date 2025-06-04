import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import BatchGNN

N = 17
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=N)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)


# Disordered Heisenberg model (example)
J = [-0.47129921,  1.07560384, -0.7645227,  -1.27589492, -0.07389608,  1.2722987,
 -1.13396229, -1.24706351, -0.03269763, -0.69614584, -1.43928219, -1.47375201,
  0.82669483, -0.4285763,   0.10482252,  0.49918111,  0.11814547]

H = -sum(J[i]*sigmaz(hi, i)*sigmaz(hi, i+1) for i in range(N-1))
H += -0.5*sum(sigmax(hi, i) for i in range(N))
H += -J[-1]*sigmaz(hi, 0)*sigmaz(hi, N-1)

# RBM Energy=-14.554+0.000j ± 0.018 [σ²=0.306, R̂=1.0068]]]
model = BatchGNN(
    graph = g,
    couplings = tuple(J),
    layers= 2,
    features = 30,
    use_attention=False,
    output_phase=False
)

# Variational State
vs = nk.vqs.MCState(sa, model, n_samples=1024)
print(f'number of parameters: {vs.n_parameters}')

# Optimizer and VMC
opt = nk.optimizer.Sgd(learning_rate = 5e-3)
gs = nk.VMC(H, opt, variational_state=vs)

gs.run(300)

