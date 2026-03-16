import os

import numpy as np
import networkx  as nx

os.system('clear')

import paprint as pa

# from Graph import Graph

# ═══ Parameters ═══════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════

A = np.eye(15, dtype=int)

pa.matrix(A, highlight=A)

# def create_erdos_renyi_graph(n=20, p=0.15, directed=True, seed=42):
# 	random_state = np.random.RandomState(seed)
# 	return nx.gnp_random_graph(n=n, p=p, seed=random_state, directed=directed)


# def main():
# 	os.system('clear')

# 	n = 20
# 	p = 0.15
# 	directed = True
# 	seed = 42

# 	G_nx = create_erdos_renyi_graph(n=n, p=p, directed=directed, seed=seed)
# 	G = Graph(nx=G_nx)

# 	print(f"Graphe d'Erdos-Renyi cree avec NetworkX : G(n={n}, p={p})")
# 	print(f"Oriente : {directed}")
# 	print(f"Nombre de sommets : {G_nx.number_of_nodes()}")
# 	print(f"Nombre d'aretes : {G_nx.number_of_edges()}")
# 	print()
# 	print(G)


# if __name__ == '__main__':
# 	main()


