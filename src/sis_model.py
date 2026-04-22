"""
sis_model.py
------------
Core implementation of the SIS epidemic model on complex networks.

Contains:
  - sis_step()           : one synchronous time step of the SIS dynamics (reference, readable)
  - sis_step_fast()      : vectorised version using sparse matrix (x30 faster, used in practice)
  - run_simulation()     : full Monte Carlo run for a single (network, beta, mu)
  - simulate_epidemic()  : sweeps over beta values and returns <rho>(beta)
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_array


# ── Single time step (readable reference implementation) ──────────────────────

def sis_step(states, adjacency_list, beta, mu, rng):
    """
    Performs one synchronous time step of the SIS dynamics.

    Parameters
    ----------
    states         : np.ndarray of int, shape (N,)
                     Current state of each node: 0 = Susceptible, 1 = Infected
    adjacency_list : list of np.ndarray
                     adjacency_list[i] contains the indices of node i's neighbors
    beta           : float
                     Infection probability per infected neighbor contact
    mu             : float
                     Spontaneous recovery probability
    rng            : np.random.Generator

    Returns
    -------
    new_states : np.ndarray of int, shape (N,)
    """
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


# ── Vectorised time step (fast, used in all simulations) ──────────────────────

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


# ── Helper: build sparse adjacency matrix ─────────────────────────────────────

def build_sparse_adj(G):
    """Returns scipy CSR sparse adjacency matrix."""
    return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)


# ── Single Monte Carlo run ────────────────────────────────────────────────────

def run_simulation(G, beta, mu, rho0=0.05, Tmax=1000, Ttrans=900, seed=None):
    """
    Runs one Monte Carlo realization of the SIS model on graph G.

    Parameters
    ----------
    G      : networkx.Graph  (nodes must be 0..N-1)
    beta   : float  – infection probability per contact
    mu     : float  – spontaneous recovery probability
    rho0   : float  – initial fraction of infected nodes (default 0.05 → 50/1000)
    Tmax   : int    – total number of time steps
    Ttrans : int    – transient steps to discard before averaging
    seed   : int or None

    Returns
    -------
    rho_mean : float
               Average fraction of infected nodes over [Ttrans, Tmax)
    rho_t    : np.ndarray, shape (Tmax,)
               Full rho(t) time series (useful for checking transient)
    """
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


# ── Monte Carlo epidemic diagram ──────────────────────────────────────────────

def simulate_epidemic(G, beta_values, mu,
                      rho0=0.05, Tmax=1000, Ttrans=900,
                      Nrep=100, seed=None, verbose=True):
    """
    Computes <rho>(beta) by Monte Carlo for a sweep of beta values.

    For each beta, runs Nrep independent repetitions and averages rho over the
    stationary window [Ttrans, Tmax) of each repetition.

    Parameters
    ----------
    G           : networkx.Graph
    beta_values : array-like of floats
    mu          : float
    rho0        : float  – initial fraction of infected nodes
    Tmax        : int
    Ttrans      : int
    Nrep        : int    – number of independent repetitions per beta value
    seed        : int or None
    verbose     : bool

    Returns
    -------
    rho_mean : np.ndarray, shape (len(beta_values),)
    rho_std  : np.ndarray, shape (len(beta_values),)
    """
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