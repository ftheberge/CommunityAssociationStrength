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
def ief_to_cluster(indptr, indices, clusters, node, target_cluster, cluster_vols):
    neighbors = indices[indptr[node]:indptr[node+1]]
    deg = len(neighbors)
    in_com_deg = np.sum(clusters[neighbors] == target_cluster)
    ief = in_com_deg/(deg + EPS)
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
    '''
    Input
    -----
    g: igraph.Graph object
    
    Optional
    --------
    ens_size: Number of clusterings to run in for the ensemble
    cas: Function to compute the association of a vertex to a community. Provided options are ief_to_cluster, nief_to_cluster, p_to_cluster, ecg_to_cluster.
    combine: Method to combine cas scores to create an edge weight. Options are 'and', 'or', 'min', 'mean'.
    method: Method to find a clustering during each run. Options are 'louvian', 'first_louvian', 'leiden', 'first_leiden'.
    resoltuion: Float that determines the resolution passed to the clustering method.

    Output
    ------
    weights: 1d numpy array of edge weights in the same order as g.es
    '''
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


def cluster_edges(g, edge_weights, min_weight=0.05, twocore=True, final="leiden", resolution=1.0):
    '''
    Input
    -----
    g: igraph.Graph object
    edge_weights: Numpy array of edge weights for the graph to cluster.

    Optional
    --------
    min_weight: minimum weight for edges in the grap to cluster
    twocore: Option to set all edges outside the two core to min_weight
    final: Clustering method. Options are 'leiden' or 'louvain'.
    resolution: Float to pass to the clustering method.
    
    Output
    ------
    igraph.VertexClustering: Clustering found by the method. Membership values are available via VertexClustering.membership.
    '''
    if min_weight > 0:
        edge_weights = (1-min_weight) * edge_weights + min_weight
    
    if twocore:
        core = g.shell_index()
        ecore = np.array([min(core[e.source],core[e.target]) for e in g.es])
        edge_weights[ecore == 1] = min_weight

    if final == "leiden":
        return g.community_leiden(weights=edge_weights, objective_function='modularity', resolution=resolution)
    elif final == "louvain":
        return g.community_multilevel(weights=edge_weights)
    else:
        raise ValueError(f"final expected one of leiden or louvain. Got {final}")


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
    if score in ["ief", "nief", "p"]:
        cluster_vols = get_cluster_vols(indptr, clusters)
        if score == "ief":
            cas = ief_to_cluster
        elif score == "nief":
            cas = nief_to_cluster
        elif score == "p":
            cas = p_to_cluster
        
        for i in range(len(scores)):
            neighbors = indices[indptr[i]:indptr[i+1]]
            adjacent_clusters = np.unique(clusters[neighbors])
            cluster_scores = np.empty_like(adjacent_clusters, dtype="float64")
            for j in range(len(adjacent_clusters)):
                cluster_scores[j] = cas(indptr, indices, clusters, i, adjacent_clusters[j], cluster_vols)
            scores[i] = np.max(cluster_scores)
    
    elif score in ["oout", "cout"]:
        oout, cout = attatchment_scores(indptr, indices, edge_weights, clusters)
        if score == "oout":
            scores = oout
        elif score == "cout":
            scores = cout
    return scores


def prune(g, clusters, thresh, score="cout", max_per_round=100, recursive=True, edge_weights=None):
    '''
    Input
    -----
    g: igraph.Graph object.
    clusters: numpy array of cluster memberships.
    thresh: Threshold in (0, 1) to prune to.

    Optional
    --------
    score: Method for scoring outlierness. Options are 'ief', 'nief', 'p', 'cout', 'oout'.
    max_per_round: Maximum number of vertices that can be pruned before recomputing scores.
    recursive: Prune rounds until all remaining scores are above the threshold.
    edge_weights: Required for cout and oout to calculate scores.
    
    Output
    ------
    pruned: Boolean numpy array with True if the vertex is classified as an outlier. The ordering is the same as g.vs.
    '''
    if score not in ["ief", "nief", "p", "oout", "cout"]:
        raise ValueError(f"expected score to be one of: ief, nief, p, oout, cout. Got {score}.")

    n = g.vcount()
    pruned = np.zeros(n, dtype="bool")
    ids = np.arange(n)

    if score in ["ief", "nief", "p"]:
        adj = g.get_adjacency_sparse()
    if score in ["oout", "cout"] and edge_weights is not None:
        g.es["w"] = edge_weights
        adj = g.get_adjacency_sparse(attribute="w")
    elif score in ["oout", "cout"]:
        raise ValueError(f"edge_weights must be passed for score {score}")

    done = False
    round_counter = 1
    while not done:
        # Remove pruned nodes to make subgraph
        index_map = ids[~pruned]
        subadj = adj[~pruned][:, ~pruned]
        subclusters = clusters[~pruned]

        scores = get_scores(subadj.indptr, subadj.indices, subadj.data, subclusters, score)
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

        if np.sum(pruned) == n:
            done = True
    
        if not recursive:
            done = True
        round_counter += 1
        #print(f"{np.sum(pruned)} nodes pruned after round {round_counter}")
    return pruned