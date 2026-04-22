import os
import networkx as nx
import numpy as np


N = 1000
MEAN_DEGREES = [4, 6]
SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "networks")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def mean_degree(G):
    degrees = [d for _, d in G.degree()]
    return np.mean(degrees)


def save_network(G, filepath):
    nx.write_edgelist(G, filepath, data=False)
    print(f"  Saved: {filepath}  |  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  <k>={mean_degree(G):.3f}")


print("Generating Erdos-Renyi networks...")
for k_target in MEAN_DEGREES:

    p = k_target / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=SEED)

    G.remove_edges_from(nx.selfloop_edges(G))

    filename = f"ER_N{N}_k{k_target}.edgelist"
    save_network(G, os.path.join(OUTPUT_DIR, filename))



print("\nGenerating Barabasi-Albert networks...")
for k_target in MEAN_DEGREES:

    m = k_target // 2
    G = nx.barabasi_albert_graph(N, m, seed=SEED)

    filename = f"BA_N{N}_k{k_target}.edgelist"
    save_network(G, os.path.join(OUTPUT_DIR, filename))


print("\nAll networks generated successfully.")