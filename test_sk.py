import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from models import SymGNN
from utils import get_connected_graph
import optax
import networkx as nx
import flax
import copy

class BestStateTracker:
    def __init__(self):
        self.best_variance = np.inf
        self.best_energy = np.inf
        self.best_state = None
        self.filename = None

    def update(self, step, log_data, driver):
        vstate = driver.state
        energy = np.real(vstate.expect(H).mean)
        variance = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        variance_score = variance / mean**2

        if variance_score < self.best_variance:
            self.best_energy = energy
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(driver.state.parameters)
            self.best_variance = variance_score
            with open(self.filename, "wb") as f:
                f.write(flax.serialization.to_bytes(driver.state))

        return self.best_variance > 0
    

N = 18
g = nk.graph.Hypercube(length=N, n_dim=1)
hi = nk.hilbert.Spin(s=1/2, N=N)

# model = nk.models.RBMModPhase()

# Sampler
sa = nk.sampler.MetropolisLocal(hi)


# Disordered Heisenberg model (example)
J = np.load('reference_couplings.npy')
g, reduced_weights = get_connected_graph(J, min_edges= N**2 // 10)
red_g = nx.DiGraph()
red_g.add_nodes_from(range(N))
red_g.add_edges_from(g.edges())
print(f'Graphs density: {nx.density(red_g)}')
H = -sum(sum(J[i, j]*sigmaz(hi, i)*sigmaz(hi, j) for i in range(N)) for j in range(N))
H += -0.5*sum(sigmax(hi, i) for i in range(N))

# SymGNN 3s/it, Energy=-66.9443 ± 0.0034 [σ²=0.0034, R̂=1.0183]]]
model = SymGNN(
graph = g,
couplings = tuple(reduced_weights),
layers= 3,
features = 15,
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
            decay_rate=0.99,
        )
tracker = BestStateTracker()
tracker.filename = "states/test.mpack"


# Optimizer and VMC
sr = nk.optimizer.SR(diag_shift=1e-3)
opt = nk.optimizer.Sgd(learning_rate = lr_schedule)
gs = nk.VMC(H, opt, variational_state=vs, preconditioner=sr)

gs.run(out="ViT", n_iter=300, callback=[tracker.update], show_progress=True)

# Compute and log results
vscore = tracker.best_variance * N
energy = tracker.best_energy
print(f'vscore={vscore:.1E} {energy=}')