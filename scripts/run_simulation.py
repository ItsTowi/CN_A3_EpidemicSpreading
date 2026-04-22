import os
import sys
import time
import numpy as np
import networkx as nx
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sis_model import simulate_epidemic

BETA_VALUES = np.round(np.arange(0.00, 0.31, 0.01), 3)   # 0.00 … 0.30, step 0.01
MU_VALUES   = [0.2, 0.4]

N    = 1000
RHO0 = 0.05
TMAX   = 1000
TTRANS = 900
NREP   = 100
SEED   = 42


BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
NETWORKS_DIR = os.path.join(BASE_DIR, "data", "networks")
RESULTS_DIR  = os.path.join(BASE_DIR, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NETWORK_FILES = {
    "ER_N1000_k4": "ER_N1000_k4.edgelist",
    "ER_N1000_k6": "ER_N1000_k6.edgelist",
    "BA_N1000_k4": "BA_N1000_k4.edgelist",
    "BA_N1000_k6": "BA_N1000_k6.edgelist",
}


def load_network(name, filename):

    path = os.path.join(NETWORKS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Network file not found: {path}\n"
            f"Run scripts/generate_networks.py first."
        )
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)   # fix non-contiguous node ids
    print(f"  Loaded {name}: N={G.number_of_nodes()}, edges={G.number_of_edges()}, "
          f"<k>={2*G.number_of_edges()/G.number_of_nodes():.3f}")
    return G



def main():
    total_configs = len(NETWORK_FILES) * len(MU_VALUES)
    config_idx = 0
    t_global_start = time.time()

    for net_name, net_file in NETWORK_FILES.items():
        print(f"\n{'='*60}")
        print(f"Network: {net_name}")
        print(f"{'='*60}")
        G = load_network(net_name, net_file)

        for mu in MU_VALUES:
            config_idx += 1
            print(f"\n[{config_idx}/{total_configs}]  mu={mu:.2f}  "
                  f"({len(BETA_VALUES)} beta values x {NREP} reps)")

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

            # Save results
            out_file = os.path.join(RESULTS_DIR, f"{net_name}_mu{mu:.2f}.csv")
            df = pd.DataFrame({
                "beta":     BETA_VALUES,
                "rho_mean": rho_mean,
                "rho_std":  rho_std,
            })
            df.to_csv(out_file, index=False)
            print(f"  Saved → {out_file}  ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"All simulations done in {total_elapsed/60:.1f} min.")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()