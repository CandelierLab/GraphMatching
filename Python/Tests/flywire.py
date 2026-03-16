import os

from Applications.FlyWire import FlyWire

os.system('clear')

# === Parameters ===========================================================

ddir = '/home/raphael/Science/Projects/Misc/GraphMatching/Files/FlyWire/'

# ==========================================================================

F = FlyWire()

# male_edges = {(r[0], r[1]): int(r[2]) for r in df}


# male_edges = {(r[0], r[1]): int(r[2]) for r in pd.read_csv(ddir + "male_connectome_graph.csv.gz")[1:]}q
# female_edges = {(r[0], r[1]): int(r[2]) for r in pd.read_csv(ddir + "female_connectome_graph.csv.gz")[1:]}
# matching = {r[0]: r[1] for r in pd.read_csv(ddir + "vnc_matching_submission_benchmark_5154247.csv.gz")[1:]}
# alignment = 0
# for male_nodes, edge_weight in male_edges.items():
#   female_nodes = (matching[male_nodes[0]], matching[male_nodes[1]])
#   alignment += min(edge_weight, female_edges.get(female_nodes, 0))
# print(f"{alignment=}") 