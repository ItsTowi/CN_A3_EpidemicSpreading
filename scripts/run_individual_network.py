import os
import sys
import time
import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sis_model import simulate_epidemic


BETA_VALUES = np.round(np.arange(0.00, 0.31, 0.01), 3)

N      = 1000
RHO0   = 0.05
TMAX   = 1000
TTRANS = 900
NREP   = 100 
SEED   = 42

BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
NETWORKS_DIR = os.path.join(BASE_DIR, "data", "networks")
RESULTS_DIR  = os.path.join(BASE_DIR, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NETWORK_OPTIONS = {
    "1": ("ER_N1000_k4", "ER_N1000_k4.edgelist"),
    "2": ("ER_N1000_k6", "ER_N1000_k6.edgelist"),
    "3": ("BA_N1000_k4", "BA_N1000_k4.edgelist"),
    "4": ("BA_N1000_k6", "BA_N1000_k6.edgelist"),
}

MU_OPTIONS = {
    "1": 0.2,
    "2": 0.4,
}


def prompt_network() -> tuple[str, str]:
    print("\nAvailable networks:")
    for key, (name, _) in NETWORK_OPTIONS.items():
        print(f"  [{key}] {name}")
    print()

    while True:
        choice = input("Select network (1-4): ").strip()
        if choice in NETWORK_OPTIONS:
            return NETWORK_OPTIONS[choice]
        print("  Invalid choice. Please enter a number between 1 and 4.")


def prompt_mu() -> float:
    print("\nAvailable mu values:")
    for key, val in MU_OPTIONS.items():
        print(f"  [{key}] mu = {val}")
    print("  [3] Custom value")
    print()

    while True:
        choice = input("Select mu (1-3): ").strip()
        if choice in MU_OPTIONS:
            return MU_OPTIONS[choice]
        if choice == "3":
            while True:
                raw = input("  Enter custom mu (0 < mu ≤ 1): ").strip()
                try:
                    val = float(raw)
                    if 0 < val <= 1:
                        return val
                    print("  Value must be in (0, 1].")
                except ValueError:
                    print("  Please enter a valid number.")
        print("  Invalid choice. Please enter 1, 2, or 3.")


def load_network(name: str, filename: str) -> nx.Graph:
    path = os.path.join(NETWORKS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Network file not found: {path}\n"
            f"Run scripts/generate_networks.py first."
        )
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)
    print(f"  Loaded {name}: N={G.number_of_nodes()}, edges={G.number_of_edges()}, "
          f"<k>={2*G.number_of_edges()/G.number_of_nodes():.3f}")
    return G



def main():
    print("=" * 60)
    print("  SIS Epidemic Simulation — Single Network Mode")
    print("=" * 60)

    # Interactive prompts
    net_name, net_file = prompt_network()
    mu = prompt_mu()

    print(f"\nConfiguration:")
    print(f"  Network : {net_name}")
    print(f"  mu      : {mu:.4f}")
    print(f"  beta    : {BETA_VALUES[0]:.2f} … {BETA_VALUES[-1]:.2f} "
          f"({len(BETA_VALUES)} values)")
    print(f"  Nrep    : {NREP}")
    print()

    confirm = input("Start simulation? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Aborted.")
        sys.exit(0)

    print(f"\nLoading network…")
    G = load_network(net_name, net_file)

    print(f"\nRunning simulation ({len(BETA_VALUES)} beta values × {NREP} reps)…")
    t_start = time.time()

    rho_mean, rho_std = simulate_epidemic(
        G,
        beta_values=BETA_VALUES,
        mu=mu,
        rho0=RHO0,
        Tmax=TMAX,
        Ttrans=TTRANS,
        Nrep=NREP,
        seed=SEED,
        verbose=True,
    )

    elapsed = time.time() - t_start

    out_file = os.path.join(RESULTS_DIR, f"{net_name}_mu{mu:.2f}.csv")
    df = pd.DataFrame({
        "beta":     BETA_VALUES,
        "rho_mean": rho_mean,
        "rho_std":  rho_std,
    })
    df.to_csv(out_file, index=False)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min).")
    print(f"Results saved → {out_file}")


if __name__ == "__main__":
    main()
