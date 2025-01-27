import numpy as np
import sknetwork as sn
import scipy.sparse as sp
from numba_stats import binom
from numba import jit

import sys
sys.path.append('../')
import CAS


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
def compute_edge_weights(indptr, indices, cas_function, clusters, combine_function="and", normalize=True, eps=1e-06):
    weights = np.empty(len(indices), dtype="float64")
    cluster_vols = get_cluster_vols(indptr, clusters)
    i = 0
    for source in range(len(indptr)-1):
        for target in indices[indptr[source]:indptr[source+1]]:
            source_to_target = cas_function(indptr, indices, clusters, source, clusters[target], cluster_vols)
            target_to_source = cas_function(indptr, indices, clusters, target, clusters[source], cluster_vols)
            if combine_function=="and":
                weights[i] = source_to_target * target_to_source
            elif combine_function=="or":
                weights[i] = source_to_target + target_to_source - source_to_target * target_to_source
            elif combine_function=="min":
                weights[i] = np.minimum(source_to_target, target_to_source)
            elif combine_function=="mean":
                weights[i] = (source_to_target + target_to_source)/2
            i += 1
    
    if normalize:
        return weights / (np.max(weights) - np.min(weights) + eps)
    return weights


def ensemble_cas_edge_weights(G, cas_function=p_to_cluster, ens_size=16, combine_function="and", normalize=True, clustering_method="first_louvain", resolution=1.0):
    weights = np.zeros(len(G.data))

    if combine_function not in ["and", "or", "min", "mean"]:
        raise ValueError(f"combine_function expected one of and, or, min or mean. Got {combine_function}")

    if clustering_method == "leiden":
        method = sn.clustering.Leiden(resolution=resolution, shuffle_nodes=True)
    elif clustering_method == "first_leiden":
        method = sn.clustering.Leiden(resolution=resolution, n_aggregations=1, shuffle_nodes=True)
    elif clustering_method == "louvain":
        method = sn.clustering.Louvain(resolution=resolution, shuffle_nodes=True)
    elif clustering_method == "first_louvain":
        method = sn.clustering.Louvain(resolution=resolution, n_aggregations=1, shuffle_nodes=True)
    else:
        raise ValueError(f"clustering_method expected one of leiden, louvain, or first_louvain. Got {clustering_method}")

    for _ in range(ens_size):
        clustering = method.fit_predict(G)
        weights += compute_edge_weights(G.indptr, G.indices, cas_function, clustering, combine_function=combine_function, normalize=normalize)

    return weights/ens_size


def cluster_edges(G, edge_weights, min_weight=0.05, twocore=True, final="leiden", resolution=1.0, n_aggregations=2):
    if min_weight > 0:
        edge_weights = (1-min_weight)*edge_weights + min_weight
    
    WG = G.copy()
    WG.data = edge_weights
    if twocore:
        core = sn.topology.get_core_decomposition(G)
        for i, core_num in enumerate(core):
            if core_num != 1:
                continue
            WG.data[WG.indptr[i]:WG.indptr[i+1]] = min_weight

    if final == "leiden":
            method = sn.clustering.Leiden(resolution=resolution, shuffle_nodes=True, n_aggregations=n_aggregations)
    elif final == "louvain":
            method = sn.clustering.Louvain(resolution=resolution, shuffle_nodes=True, n_aggregations=n_aggregations)
    else:
            raise ValueError(f"final expected one of leiden or louvain. Got {final}")
    return method.fit_predict(WG)


#######################################
# Outlier scores and pruning methods #
######################################
@jit
def attatchment_scores(indptr, indices, edge_weights, clustering, eps=1e-6):
    # 1 - outlier scores from ECG. So that low attatchment are likely outliers to match with CAS scores.
    degs = np.array([indptr[i+1]-indptr[i] for i in range(len(indptr)-1)])
    good = np.zeros(len(degs))
    bad = np.zeros(len(degs))
    i = 0
    for source in range(len(indptr)-1):
        for target in indices[indptr[source]:indptr[source+1]]:
            if clustering[source] == clustering[target]:
                good[source] += edge_weights[i]
            else:
                bad[source] += edge_weights[i]
            i+=1
    
    overall = 1 - (degs - bad - good)/(degs + eps)
    community = 1 - bad/(bad + good + eps)
    return overall, community


@jit
def get_scores(indptr, indices, edge_weights, clusters, score):
    scores = np.empty(len(indptr)-1, dtype="float64")
    if score == "ief":
        cluster_vols = get_cluster_vols(indptr, clusters)
        for i in range(len(scores)):
            scores[i] = ief_to_cluster(indptr, indices, clusters, i, clusters[i], cluster_vols)
    elif score == "nief":
        cluster_vols = get_cluster_vols(indptr, clusters)
        for i in range(len(scores)):
            scores[i] = nief_to_cluster(indptr, indices, clusters, i, clusters[i], cluster_vols)
    elif score == "p":
        cluster_vols = get_cluster_vols(indptr, clusters)
        for i in range(len(scores)):
            scores[i] = p_to_cluster(indptr, indices, clusters, i, clusters[i], cluster_vols)

    elif score in ["oout", "cout"]:
        oout, cout = attatchment_scores(indptr, indices, edge_weights, clusters)
        if score == "oout":
            scores = oout
        elif score == "cout":
            scores = cout
    return scores


def prune(G, clusters, thresh, score="cout", max_per_round=100, recursive=True, edge_weights=None, eps=1e-6):
    if score not in ["ief", "nief", "p", "oout", "cout"]:
        raise ValueError(f"expected score to be one of: ief, nief, p, oout, cout. Got {score}.")
    # Prune recursively. The graph is not reclustered, and the CAS scores are computed with edge weights of the original graph.
    n = G.shape[0]
    pruned = np.zeros(n, dtype="bool")

    if score in ["oout", "cout"] and edge_weights is not None:
        G = G.copy()  # don't update original
        G.data = edge_weights
    elif score in ["oout", "cout"]:
        raise ValueError(f"edge_weights must be passed for score {score}")

    done = False
    round_counter = 1
    while not done:
        # Remove pruned nodes to make subgraph
        index_map = np.arange(n)[~pruned]
        subG = G[~pruned][:, ~pruned]
        subclusters = clusters[~pruned]

        scores = get_scores(subG.indptr, subG.indices, subG.data, subclusters, score)
        # prune below thresh
        argsort = np.argsort(scores)
        if len(scores) > max_per_round and scores[argsort[max_per_round]] < thresh:
            #prune max
            to_prune = argsort[:max_per_round]
            pruned[index_map[to_prune]] = True
        elif len(scores) > 1 and scores[argsort[0]] < thresh:
            # prune some
            pruned[index_map[argsort[scores[argsort] < thresh]]] = True  # TODO ew
        else:
            # nothing to prune
            done = True
    
        if not recursive:
            done = True
        round_counter += 1
        #print(f"{np.sum(pruned)} nodes pruned after round {round_counter}")
    return pruned