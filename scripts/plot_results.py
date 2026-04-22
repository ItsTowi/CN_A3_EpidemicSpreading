"""
plot_results.py
---------------
Plots the SIS epidemic diagrams from Monte Carlo simulation results.

For each value of mu (0.2, 0.4), produces ONE figure with:
  - rho(lambda) curves for the 4 networks (ER k4, ER k6, BA k4, BA k6)
  - Shaded error bands (±1 std)
  - Vertical dashed lines for the HMF theoretical threshold (beta_c = mu/<k>)

Saves figures to:
    data/figures/SIS_mu0.20.pdf  (and .png)
    data/figures/SIS_mu0.40.pdf  (and .png)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MU_VALUES = [0.2, 0.4]

NETWORKS = [
    ("ER_N1000_k4", "ER  <k>=4", 4),
    ("ER_N1000_k6", "ER  <k>=6", 6),
    ("BA_N1000_k4", "BA  <k>=4", 4),
    ("BA_N1000_k6", "BA  <k>=6", 6),
]

COLORS  = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
MARKERS = ["o", "s", "^", "D"]
MSIZE   = 4


def load_csv(net_name, mu):
    path = os.path.join(RESULTS_DIR, f"{net_name}_mu{mu:.2f}.csv")
    if not os.path.exists(path):
        print(f"  [warning] Not found: {os.path.basename(path)} -- skipping.")
        return None
    return pd.read_csv(path)


def make_figure(mu):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.tick_params(colors="#8B949E", labelsize=10)
    ax.xaxis.label.set_color("#C9D1D9")
    ax.yaxis.label.set_color("#C9D1D9")
    ax.grid(True, linestyle="--", linewidth=0.5, color="#21262D", zorder=0)

    legend_handles = []

    for idx, (net_name, net_label, k_mean) in enumerate(NETWORKS):
        color  = COLORS[idx]
        marker = MARKERS[idx]

        df = load_csv(net_name, mu)
        if df is None:
            continue

        betas    = df["beta"].values
        rho_mean = df["rho_mean"].values
        rho_std  = df["rho_std"].values

        ax.plot(betas, rho_mean,
                marker=marker, markersize=MSIZE,
                color=color, linewidth=1.6,
                linestyle="-", zorder=3)

        ax.fill_between(betas,
                        np.clip(rho_mean - rho_std, 0, 1),
                        np.clip(rho_mean + rho_std, 0, 1),
                        color=color, alpha=0.15, zorder=2)


        beta_c = mu / k_mean
        ax.axvline(beta_c, color=color, linewidth=0.9,
                   linestyle=":", alpha=0.7, zorder=1)

        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker,
                   markersize=MSIZE, linewidth=1.6,
                   label=f"{net_label}  (beta_c = {beta_c:.3f})")
        )

    legend_handles.append(
        Line2D([0], [0], color="white", linewidth=0.9,
               linestyle=":", alpha=0.7,
               label="HMF threshold  beta_c = mu / <k>")
    )

    ax.set_xlabel("Infection probability  lambda", fontsize=12)
    ax.set_ylabel("Infected fraction  rho", fontsize=12)
    ax.set_title(
        f"SIS Epidemic Spreading -- Monte Carlo Simulation\n"
        f"N = 1000,  mu = {mu:.1f},  rho0 = 0.05,  T = 1000,  N_rep = 100",
        color="#C9D1D9", fontsize=11, pad=12
    )
    ax.set_xlim(-0.005, 0.305)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.25,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
    )

    plt.tight_layout()

    stem = os.path.join(FIGURES_DIR, f"SIS_mu{mu:.2f}")
    fig.savefig(stem + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    print(f"  Saved --> {stem}.pdf / .png")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  SIS Epidemic Simulation -- Plotting")
    print("=" * 60)

    for mu in MU_VALUES:
        print(f"\nPlotting mu={mu:.2f} ...")
        make_figure(mu)

    print(f"\nAll figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
