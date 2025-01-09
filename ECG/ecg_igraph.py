import numpy as np
import igraph as ig
#from scipy.stats import binom
import sknetwork as sn

from numba_stats import binom
from numba import jit

import sys
sys.path.append('../')
import CAS


EPS = 1e-6


@jit
def is_internal_edge(indptr, indices, clusters):
    is_internal = np.zeros(indptr[-1], dtype="bool")
    i = 0
    for source in range(len(indptr)-1):
        for target in indices[indptr[source]:indptr[source+1]]:
            is_internal[i] = clusters[source] == clusters[target]
            i += 1
    return is_internal


@jit
def ief_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols, eps=1e-6):
    neighbors = indices[indptr[node]:indptr[node+1]]
    deg = len(neighbors)
    in_com_deg = np.sum(clusters[neighbors] == target_cluster)
    ief = in_com_deg/(deg + eps)
    return ief


@jit
def nief_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols):
    nief = np.maximum(ief_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols) - cluster_vols[target_cluster], 0.0)
    return nief


@jit
def p_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols):
    neighbors = indices[indptr[node]:indptr[node+1]]
    deg = np.array([len(neighbors)], dtype="int64")
    in_com_deg = np.array([np.sum(clusters[neighbors] == target_cluster) - 1], dtype="float64")
    p = binom._cdf(in_com_deg, deg, cluster_vols[target_cluster])[0]
    return p


@jit
def ecg_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols):
    return int(clusters[node] == target_cluster)


@jit
def get_cluster_vols(indptr, clusters):
    cluster_vols = np.zeros(np.max(clusters)+1, dtype="int32")
    degrees = np.array([indptr[i+1]-indptr[i] for i in range(len(indptr)-1)])
    for c, deg in zip(clusters, degrees):
        cluster_vols[c] += deg
    return cluster_vols / indptr[-1]  # indptr[-1] is the total volume of the graph


@jit
def compute_edge_weights(indptr, indices, clusters, cas):
    weights = np.empty(len(indices), dtype="float64")
    cluster_vols = get_cluster_vols(indptr, clusters)
    i = 0
    for source in range(len(indptr)-1):
        for target in indices[indptr[source]:indptr[source+1]]:
            weights[i] = cas(indptr, indices, clusters, source, clusters[target], cluster_vols)
            i += 1
    return weights


@jit
def get_edge_data_ids(edge_endpoint_ids, indptr, indices):
    data_ids = np.empty_like(edge_endpoint_ids)
    for i in range(edge_endpoint_ids.shape[0]):
        source = edge_endpoint_ids[i, 0]
        target = edge_endpoint_ids[i, 1]
        for j, end in enumerate(indices[indptr[source]:indptr[source+1]]):
            if end == target:
                data_ids[i, 0] = indptr[source] + j
                break
        for j, end in enumerate(indices[indptr[target]:indptr[target+1]]):
            if end == source:
                data_ids[i, 1] = indptr[target] + j 
    return data_ids


@jit
def add_weights(weights, edge_data_ids, data, combine):
    for i in range(weights.shape[0]):
        source_to_target = data[edge_data_ids[i, 0]]
        target_to_source = data[edge_data_ids[i, 1]]
        if combine == "and":
            weights[i] += source_to_target * target_to_source
        elif combine == "or":
            weights[i] += source_to_target + target_to_source - source_to_target * target_to_source
        elif combine == "min":
            weights[i] += np.minimum(source_to_target, target_to_source)
        elif combine == "mean":
            weights[i] += (source_to_target + target_to_source)/2


def ensemble_edge_weights(g: ig.Graph, ens_size=16, cas=p_to_cluster, combine="and", method="first_louvain", resolution=1.0):
    if combine not in ["and", "or", "min", "mean"]:
        raise ValueError(f"combine_function expected one of and, or, min or mean. Got {combine}")
    if method not in ["louvain", "first_louvain", "leiden", "first_leiden"]:
        raise ValueError(f"clustering_method expected one of leiden, louvain, or first_louvain. Got {method}")

    weights = np.zeros(g.ecount())
    adj = g.get_adjacency_sparse()
    edge_endpoint_ids = np.array([(e.source, e.target) for e in g.es])
    edge_adj_data_ids = get_edge_data_ids(edge_endpoint_ids, adj.indptr, adj.indices)

    for _ in range(ens_size):
        permutation = np.random.permutation(g.vcount())
        g_shuffled = g.permute_vertices(permutation)
        if method == "leiden":
            clusters = np.array(g_shuffled.community_leiden(resoltuion=resolution, objective_function="modularity").membership)
        elif method == "first_leiden":
            clusters = np.array(g_shuffled.community_leiden(resoltuion=resolution, objective_function="modularity", n_iterations=1).membership)
        elif method == "louvain":
            clusters = np.array(g_shuffled.community_multilevel(resoltuion=resolution).membership)
        elif method == "first_louvain":
            clusters = np.array(g_shuffled.community_multilevel(return_levels=True)[0].membership)

        clusters = clusters[permutation]
        add_weights(weights, edge_adj_data_ids, compute_edge_weights(adj.indptr, adj.indices, clusters, cas), combine)

    return weights/ens_size


def cluster_edges(G, edge_weights, min_weight=0.05, twocore=True, final="leiden", resolution=1.0):
    if min_weight > 0:
        edge_weights = (1-min_weight) * edge_weights + min_weight
    
    if twocore:
        core = G.shell_index()
        ecore = np.array([min(core[e.source],core[e.target]) for e in G.es])
        edge_weights[ecore == 1] = min_weight

    if final == "leiden":
        return G.community_leiden(weights=edge_weights, objective_function='modularity', resolution=resolution)
    elif final == "louvain":
        return G.community_multilevel(weights=edge_weights)
    else:
        raise ValueError(f"final expected one of leiden or louvain. Got {final}")


def outlier_scores(G, edge_weights, clustering):
    degs = np.array(G.degree())
    good = np.zeros(G.vcount())
    bad = np.zeros(G.vcount())
    for i, e in enumerate(G.es):
        if clustering[e.source] == clustering[e.target]:
            good[e.source] += edge_weights[i]
            good[e.target] += edge_weights[i]
        else:
            bad[e.source] += edge_weights[i]
            bad[e.target] += edge_weights[i]       

    overall = (degs - bad - good)/degs
    community = bad/(bad + good)
    return overall, community


########################
# Recursively pruneing #
########################

def get_com_score(scores, com):
    return np.array([scores[i, com[i]] for i in range(len(com))])


def attatchment_scores(G, edge_weights, clustering, eps=1e-6):
    # 1 - outlier scores from ECG. So that low attatchment are likely outliers to match with CAS scores.
    degs = sn.utils.get_degrees(G)
    good = np.zeros(len(degs))
    bad = np.zeros(len(degs))
    i = 0
    for source in range(len(G.indptr)-1):
        for target in G.indices[G.indptr[source]:G.indptr[source+1]]:
            if clustering[source] == clustering[target]:
                good[source] += edge_weights[i]
            else:
                bad[source] += edge_weights[i]
            i+=1
    
    overall = 1 - (degs - bad - good)/(degs + eps)
    community = 1 - bad/(bad + good + eps)
    return overall, community


def prune(g, edge_weights, coms, thresh, score="beta", max_per_round=500, recursive=True, eps=1e-6):
    # Prune recursively. The graph is not reclustered, and the CAS scores are computed with edge weights of the original graph.
    n = g.vcount()

    # Make sparse matrix versions of weighted graph and unweighted graph for CAS scores.
    pruned = np.zeros(n, dtype="bool")
    adj = sp.dok_matrix((n, n), dtype="int32")
    sn_edgeweights = sp.dok_matrix((n, n))
    for e, w in zip(g.es, edge_weights):
        adj[e.source, e.target] = 1
        adj[e.target, e.source] = 1
        sn_edgeweights[e.source, e.target] = w + eps  # in case edge_weight is 0 sparse matrix missing edges
        sn_edgeweights[e.target, e.source] = w + eps
    adj = adj.tocsr()
    sn_edgeweights = sn_edgeweights.tocsr()

    done = False
    round_counter = 1
    while not done:
        # Remove pruned nodes to make subgraph
        index_map = np.arange(n)[~pruned]
        subadj = adj[~pruned][:, ~pruned]
        subedgeweights = sn_edgeweights[~pruned][:, ~pruned]
        subcoms = coms[~pruned]

        # get score
        if score in ["ief", "beta", "c", "p"]:
            ief, beta, c, p, degs = CAS.CAS(subadj, CAS.partition2sparse(subcoms))
            if score == "ief":
                s = get_com_score(ief, subcoms)
            elif score == "beta":
                s = get_com_score(beta, subcoms)
            elif score == "c":
                s = get_com_score(c, subcoms)
            elif score == "p":
                s = get_com_score(p, subcoms)
        elif score in ["oout", "cout"]:
            oout, cout = attatchment_scores(subadj, subedgeweights.data, subcoms)
            if score == "oout":
                s = oout
            elif score == "cout":
                s = cout
        else:
            raise ValueError(f"expected score to be one of: ief, beta, c, p, oout, cout. Got {score}.")
        
        # prune below thresh
        argsort = np.argsort(s)
        if len(s) > max_per_round and s[argsort[max_per_round]] < thresh:
            #prune max
            to_prune = argsort[:max_per_round]
            pruned[index_map[to_prune]] = True
        elif len(s) > 1 and s[argsort[0]] < thresh:
            # prune some
            for i in argsort:
                if s[i] < thresh:
                    pruned[index_map[i]] = True
                else:
                    break
        else:
            # nothing to prune
            done = True
        
        if not recursive:
            done = True
        round_counter += 1
        #print(f"{np.sum(pruned)} nodes pruned after round {round_counter}")
    return pruned