"""
plot_temporal.py
----------------
Plots the temporal evolution rho(t) of the SIS epidemic for a single
(network, beta, mu) combination.

Shows:
  - Individual trajectories (faint lines)
  - Mean rho(t) averaged over all repetitions (bold line)
  - Shaded ±1 std band
  - Vertical dashed line marking Ttrans (start of stationary window)
  - Horizontal dashed line for the stationary mean rho
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sis_model import run_simulation, build_sparse_adj

BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
NETWORKS_DIR = os.path.join(BASE_DIR, "data", "networks")
FIGURES_DIR  = os.path.join(BASE_DIR, "data", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

NETWORK_NAME = "ER_N1000_k4"        # ER_N1000_k4 | ER_N1000_k6 | BA_N1000_k4 | BA_N1000_k6
NETWORK_FILE = f"{NETWORK_NAME}.edgelist"

BETA   = 0.10
MU     = 0.20
RHO0   = 0.05
TMAX   = 1000
TTRANS = 900
NREP   = 30
SEED   = 42

BG_COLOR       = "#0D1117"
GRID_COLOR     = "#21262D"
SPINE_COLOR    = "#30363D"
TEXT_COLOR     = "#C9D1D9"
MUTED_COLOR    = "#8B949E"
TRAJ_COLOR     = "#457B9D"
MEAN_COLOR     = "#E63946"
BAND_COLOR     = "#E63946"
TTRANS_COLOR   = "#E9C46A"
STAT_COLOR     = "#2A9D8F"


def load_network(name, filename):
    path = os.path.join(NETWORKS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Network file not found: {path}\n"
            f"Run scripts/generate_networks.py first."
        )
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)
    k_mean = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"  Loaded {name}: N={G.number_of_nodes()}, "
          f"edges={G.number_of_edges()}, <k>={k_mean:.3f}")
    return G, k_mean


def run_trajectories(G, beta, mu, rho0, Tmax, Ttrans, Nrep, seed):
    rng = np.random.default_rng(seed)
    all_rho = np.zeros((Nrep, Tmax))

    for rep in range(Nrep):
        rep_seed = int(rng.integers(0, 2**31))
        _, rho_t = run_simulation(
            G, beta, mu,
            rho0=rho0, Tmax=Tmax, Ttrans=Ttrans,
            seed=rep_seed,
        )
        all_rho[rep] = rho_t
        print(f"  rep {rep+1:2d}/{Nrep}  stationary rho = {rho_t[Ttrans:].mean():.4f}")

    t_axis   = np.arange(Tmax)
    rho_mean = all_rho.mean(axis=0)
    rho_std  = all_rho.std(axis=0)
    rho_stat = all_rho[:, Ttrans:].mean()   # grand stationary average

    return t_axis, rho_mean, rho_std, all_rho, rho_stat


def make_plot(t_axis, rho_mean, rho_std, all_rho, rho_stat,
              net_name, k_mean, beta, mu, Ttrans):

    fig, ax = plt.subplots(figsize=(10, 5.5))

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=10)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(True, linestyle="--", linewidth=0.4, color=GRID_COLOR, zorder=0)

    Tmax = len(t_axis)

    for rep in range(len(all_rho)):
        ax.plot(t_axis, all_rho[rep],
                color=TRAJ_COLOR, linewidth=0.5, alpha=0.18, zorder=1)

    ax.fill_between(t_axis,
                    np.clip(rho_mean - rho_std, 0, 1),
                    np.clip(rho_mean + rho_std, 0, 1),
                    color=BAND_COLOR, alpha=0.20, zorder=2)

    ax.plot(t_axis, rho_mean,
            color=MEAN_COLOR, linewidth=2.2, zorder=3,
            label=f"Mean  rho(t)  (N={len(all_rho)} reps)")

    ax.axvline(Ttrans, color=TTRANS_COLOR, linewidth=1.2,
               linestyle="--", alpha=0.85, zorder=4,
               label=f"Transient cutoff  t = {Ttrans}")

    ax.axhline(rho_stat, color=STAT_COLOR, linewidth=1.1,
               linestyle=":", alpha=0.90, zorder=4,
               label=f"Stationary  <rho> = {rho_stat:.4f}")

    beta_c = mu / k_mean
    above = beta > beta_c
    regime = "above threshold  (endemic)" if above else "below threshold  (absorbing)"
    ax.text(0.98, 0.05,
            f"beta_c (HMF) = mu/<k> = {beta_c:.3f}\nbeta = {beta:.3f}  →  {regime}",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            color=MUTED_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22",
                      edgecolor=SPINE_COLOR, alpha=0.8))

    ax.set_xlabel("Time step  t", fontsize=12)
    ax.set_ylabel("Infected fraction  rho(t)", fontsize=12)
    ax.set_title(
        f"SIS Temporal Evolution — {net_name}  |  beta={beta:.3f},  mu={mu:.2f}\n"
        f"N=1000,  <k>={k_mean:.1f},  rho0={RHO0},  {len(all_rho)} independent runs",
        color=TEXT_COLOR, fontsize=11, pad=12
    )
    ax.set_xlim(0, Tmax - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.3,
        facecolor="#161B22",
        edgecolor=SPINE_COLOR,
        labelcolor=TEXT_COLOR,
    )

    plt.tight_layout()

    stem = os.path.join(
        FIGURES_DIR,
        f"rho_t_{net_name}_beta{beta:.3f}_mu{mu:.2f}"
    )
    fig.savefig(stem + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved --> {stem}.pdf / .png")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  SIS Temporal Evolution — rho(t) plot")
    print("=" * 60)
    print(f"  Network : {NETWORK_NAME}")
    print(f"  beta    : {BETA}")
    print(f"  mu      : {MU}")
    print(f"  Tmax    : {TMAX}  |  Ttrans : {TTRANS}  |  Nrep : {NREP}")
    print()

    print("Loading network...")
    G, k_mean = load_network(NETWORK_NAME, NETWORK_FILE)

    print(f"\nRunning {NREP} trajectories...")
    t_axis, rho_mean, rho_std, all_rho, rho_stat = run_trajectories(
        G, BETA, MU, RHO0, TMAX, TTRANS, NREP, SEED
    )

    print("\nPlotting...")
    make_plot(t_axis, rho_mean, rho_std, all_rho, rho_stat,
              NETWORK_NAME, k_mean, BETA, MU, TTRANS)

    print("\nDone.")


if __name__ == "__main__":
    main()