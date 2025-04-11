import pandas as pd
import igraph as ig
import numpy as np
from collections import defaultdict


def load_snap(graph_path, coms_path, drop_outliers=False):
    # Read edges
    edges = pd.read_csv(graph_path, sep='\t', comment="#")
    g = ig.Graph.DataFrame(edges, directed=False)

    # Read Communities
    coms = []
    with open(coms_path, "r") as infile:
        for line in infile:
            x = line[:-1]  # drop trailing newline
            x = x.split('\t')
            coms.append([int(y) for y in x]) ## map to 0-based

    # Store each nodes communities
    c = [set() for _ in range(g.vcount())]
    for i, com in enumerate(coms):
        for v in com:
            c[v].add(i)
    c = [frozenset(i) for i in c]
    g.vs["comms"] = c

    # Vertex list is not continugous, drop degree 0 vertices
    to_drop = np.array(g.degree()) == 0
    if drop_outliers:
        is_outlier = np.array([len(comms) == 0 for comms in g.vs["comms"]])
        to_drop = np.bitwise_or(to_drop, is_outlier)
    #print(f"To drop {np.sum(to_drop)} vertices.")
    indices_to_keep = np.argwhere(~to_drop).reshape(-1)
    #print(f"Number to keeep: {len(indices_to_keep)}")
    g = g.subgraph(indices_to_keep)
    assert(g.vcount() == len(to_drop) - np.sum(to_drop))

    new_indices = np.full(len(to_drop), -1, dtype="int64")
    new_indices[indices_to_keep] = np.arange(len(indices_to_keep))

    # Reindex coms list.
    for i in range(len(coms)):
        new_com_ids = new_indices[coms[i]]
        coms[i] = frozenset(new_com_ids)

    return g, coms


def load_abcdoo(graph_path, coms_path, has_outliers=True):
    # Read Edges
    edges = pd.read_csv(graph_path, sep='\t', header=None)-1
    g = ig.Graph.DataFrame(edges, directed=False)

    # Read Communities
    node_coms = []
    with open(coms_path, "r") as infile:
        for line in infile:
            # each line has the format: node_id\t[a,b,c]\n
            # for this node belonging to communities a,v,c
            # if there are outliers then they have community [1]
            # node_id counts up starting at 1
            x = line.split('\t')[1].rstrip()[1:-1] # gets a,b,c
            if has_outliers:
                coms = [int(y)-2 for y in x.split(',') if int(y)>1] # map to 0-based and remove outlier
            else:
                coms = [int(y)-1 for y in x.split(',')]  # map to 0-based
            node_coms.append(frozenset(coms))
    g.vs['comms'] = node_coms
    
    # Build list of communities
    coms = defaultdict(set)
    for i, c in enumerate(g.vs['comms']):
        for com in c:
            coms[com].add(i)
    #coms = list(coms.values())
    return g, coms


def load_coms(coms_path, has_outliers):
    # Read Communities
    node_coms = []
    with open(coms_path, "r") as infile:
        for line in infile:
            # each line has the format: node_id\t[a,b,c]\n
            # for this node belonging to communities a,v,c
            # if there are outliers then they have community [1]
            # node_id counts up starting at 1
            x = line.split('\t')[1].rstrip()[1:-1] # gets a,b,c
            if has_outliers:
                coms = [int(y)-2 for y in x.split(',') if int(y)>1] # map to 0-based and remove outlier
            else:
                coms = [int(y)-1 for y in x.split(',')]  # map to 0-based
            node_coms.append(frozenset(coms))

    coms = defaultdict(set)
    for i, c in enumerate(node_coms):
        for com in c:
            coms[com].add(i)
    coms = list(coms.values())
    return coms, node_coms