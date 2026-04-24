import numpy as np
import networkx as nx
from scipy.sparse import csr_array

"""
def sis_step(states, adjacency_list, beta, mu, rng):
    N = len(states)
    new_states = states.copy()

    for i in range(N):
        if states[i] == 1:
            # Infected node: recover with probability mu
            if rng.random() < mu:
                new_states[i] = 0
        else:
            # Susceptible node: each infected neighbor independently tries to infect
            neighbors = adjacency_list[i]
            infected_neighbors = neighbors[states[neighbors] == 1]
            for _ in infected_neighbors:
                if rng.random() < beta:
                    new_states[i] = 1
                    break  # stop as soon as infected

    return new_states
"""

def sis_step_fast(states, A_csr, beta, mu, rng):
    """
    Vectorised synchronous SIS time step using a sparse adjacency matrix.

    The infection rule is mathematically equivalent to sis_step():
    a susceptible node i with k_inf infected neighbors stays susceptible with
    probability (1-beta)^k_inf, i.e. it escapes infection from every neighbor.

    Parameters
    ----------
    states  : np.ndarray of int, shape (N,)  — 0=S, 1=I
    A_csr   : scipy sparse matrix (N x N)    — adjacency matrix
    beta    : float
    mu      : float
    rng     : np.random.Generator

    Returns
    -------
    new_states : np.ndarray of int, shape (N,)
    """
    N = len(states)
    new_states = states.copy()
    infected_mask = states == 1

    # Recovery: each infected node recovers independently with prob mu
    recovering = infected_mask & (rng.random(N) < mu)
    new_states[recovering] = 0

    # Infection: number of infected neighbors for each node
    n_inf_neighbors = A_csr.dot(infected_mask.astype(np.float64))

    # Probability of escaping all infected neighbors = (1-beta)^n_inf_neighbors
    prob_escape = (1.0 - beta) ** n_inf_neighbors

    # A susceptible node gets infected if it fails to escape
    susceptible_mask = states == 0
    new_infections = susceptible_mask & (rng.random(N) > prob_escape)
    new_states[new_infections] = 1

    return new_states


def build_sparse_adj(G):
    return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)


def run_simulation(G, beta, mu, rho0=0.05, Tmax=1000, Ttrans=900, seed=None):
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    A = build_sparse_adj(G)

    # Initialise: infect rho0 fraction of nodes (at least 50 for robustness)
    n_infected_0 = max(50, int(rho0 * N))
    states = np.zeros(N, dtype=int)
    states[rng.choice(N, size=n_infected_0, replace=False)] = 1

    rho_t = np.empty(Tmax)
    for t in range(Tmax):
        rho_t[t] = states.sum() / N
        states = sis_step_fast(states, A, beta, mu, rng)

    rho_mean = rho_t[Ttrans:].mean()
    return rho_mean, rho_t


def simulate_epidemic(G, beta_values, mu,
                      rho0=0.05, Tmax=1000, Ttrans=900,
                      Nrep=100, seed=None, verbose=True):
    rng = np.random.default_rng(seed)
    beta_values = np.asarray(beta_values)
    rho_mean = np.zeros(len(beta_values))
    rho_std  = np.zeros(len(beta_values))

    for i, beta in enumerate(beta_values):
        rep_means = np.empty(Nrep)
        for rep in range(Nrep):
            rep_seed = int(rng.integers(0, 2**31))
            rep_means[rep], _ = run_simulation(
                G, beta, mu,
                rho0=rho0, Tmax=Tmax, Ttrans=Ttrans,
                seed=rep_seed
            )
        rho_mean[i] = rep_means.mean()
        rho_std[i]  = rep_means.std()

        if verbose:
            print(f"  beta={beta:.3f}  <rho>={rho_mean[i]:.4f} ± {rho_std[i]:.4f}")

    return rho_mean, rho_std