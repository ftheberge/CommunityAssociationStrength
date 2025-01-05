import numpy as np
import sknetwork as sn
import scipy.sparse as sp
from numba_stats import binom
from numba import jit

import sys
sys.path.append('../')
import CAS


@jit
def ief_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols):
    neighbors = indices[indptr[node]:indptr[node+1]]
    deg = len(neighbors)
    in_com_deg = np.sum(clusters[neighbors] == target_cluster)
    ief = in_com_deg/deg
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
    source_to_target = np.empty(len(indices), dtype="float64")
    target_to_source = np.empty(len(indices), dtype="float64")
    cluster_vols = get_cluster_vols(indptr, clusters)
    i = 0
    for source in range(len(indptr)-1):
        for target in indices[indptr[source]:indptr[source+1]]:
            source_to_target[i] = cas_function(indptr, indices, clusters, source, clusters[target], cluster_vols)
            target_to_source[i] = cas_function(indptr, indices, clusters, target, clusters[source], cluster_vols)
            i += 1

    if combine_function=="and":
        weights = source_to_target * target_to_source
    elif combine_function=="or":
        weights = source_to_target + target_to_source - source_to_target * target_to_source
    if combine_function=="min":
        weights = np.minimum(source_to_target, target_to_source)
    elif combine_function=="mean":
        weights = (source_to_target + target_to_source)/2
    else:
        raise ValueError(f"combine_function must be one of 'and', 'or', 'min', 'mean'. Got {combine_function}")
    
    if normalize:
        return weights / (np.max(weights) - np.min(weights) + eps)
    return weights


def ensemble_cas_edge_weights(G, cas_function=p_to_cluster, ens_size=16, combine_function="min", normalize=True, clustering_method="first_louvain", resolution=1.0):
    weights = np.zeros(len(G.data))

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


def cluster_edges(G, edge_weights, min_weight=0.05, twocore=True, final="leiden", resolution=1.0):
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
            method = sn.clustering.Leiden(resolution=resolution)
    elif final == "louvain":
            method = sn.clustering.Louvain(resolution=resolution)
    else:
            raise ValueError(f"final expected one of leiden or louvain. Got {final}")
    return method.fit_predict(WG)


def outlier_scores(G, edge_weights, clustering):
    degs = sn.utils.get_degrees(G)
    good = np.zeros(len(degs), dtype="float32")
    bad = np.zeros(len(degs), dtype="float32")
    i = 0
    for source in range(len(G.indptr)):
        for target in G.indices[source:source+1]:
            if clustering[source] == clustering[target]:
                good[source] += edge_weights[i]
                good[target] += edge_weights[i]
            else:
                bad[source] += edge_weights[i]
                bad[target] += edge_weights[i]
            i+=1
    
    overall = (degs - bad - good)/degs
    community = bad/(bad + good)
    return overall, community