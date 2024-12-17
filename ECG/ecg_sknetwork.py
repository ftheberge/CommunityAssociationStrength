import numpy as np
import sknetwork as sn
import scipy.sparse as sp

import sys
sys.path.append('../')
import CAS


# Setting shuffle_nodes = True does not work
# Need to manually shuffle
# Not confident we are keeping track correctly

# sknetwork clustering functions seem to always be similar, falling into similar small clusterings each time


def compute_edge_weights(G, cas, clusters, combine_function="min", normalize=True, eps=1e-06):
    source_to_target = np.empty(len(G.data))
    target_to_source = np.empty(len(G.data))
    i = 0
    for source in range(len(G.indptr)-1):
        for target in G.indices[source:source+1]:
            source_to_target[i] = cas[source, clusters[target]]
            target_to_source[i] = cas[target, clusters[source]]
            i += 1

    if combine_function=="and":
        weights = source_to_target * target_to_source
    elif combine_function=="or":
        weights = source_to_target + target_to_source - source_to_target*target_to_source
    if combine_function=="min":
        weights = np.minimum(source_to_target, target_to_source)
    elif combine_function=="mean":
        weights = (source_to_target + target_to_source)/2
    else:
        raise ValueError(f"combine_function must be one of 'and', 'or', 'min', 'mean'. Got {combine_function}")
    
    if normalize:
        return weights / (np.max(weights) - np.min(weights))
    return weights


def cas_edge_weights(G, clusters, combine_function="min", normalize=True):
    ief, beta, c, p, degs = CAS.CAS(G, CAS.partition2sparse(clusters))

    ief_weights = compute_edge_weights(G, ief, clusters, combine_function=combine_function, normalize=normalize)
    beta_weights = compute_edge_weights(G, beta, clusters, combine_function=combine_function, normalize=normalize)
    c_weights = compute_edge_weights(G, c, clusters, combine_function=combine_function, normalize=normalize)
    p_weights = compute_edge_weights(G, p, clusters, combine_function=combine_function, normalize=normalize)
    ecg_weights = compute_edge_weights(G, CAS.partition2sparse(clusters), clusters, combine_function=combine_function, normalize=False)
    
    return ief_weights, beta_weights, c_weights, p_weights, ecg_weights


def ensemble_cas_edge_weights(G, ens_size=16, combine_function="min", normalize=True, clustering_method="first_louvain", resolution=1.0):
    ief_weights = np.zeros(len(G.data))
    beta_weights = np.zeros(len(G.data))
    c_weights = np.zeros(len(G.data))
    p_weights = np.zeros(len(G.data))
    ecg_weights = np.zeros(len(G.data))

    if clustering_method == "leiden":
        method = sn.clustering.Leiden(resolution=resolution, shuffle_nodes=True)
    elif clustering_method == "first_leiden":
        method = sn.clustering.Louvain(resolution=resolution, n_aggregations=1, shuffle_nodes=True)
    elif clustering_method == "louvain":
        method = sn.clustering.Louvain(resolution=resolution, shuffle_nodes=True)
    elif clustering_method == "first_louvain":
        method = sn.clustering.Louvain(resolution=resolution, n_aggregations=1, shuffle_nodes=True)
    else:
        raise ValueError(f"clustering_method expected one of leiden, louvain, or first_louvain. Got {clustering_method}")

    for _ in range(ens_size):
        permutation = np.random.permutation(G.shape[0])
        temp = G[permutation][:, permutation]
        temp_clustering = method.fit_predict(temp)

        inverse_permutation = np.empty_like(permutation)
        inverse_permutation[permutation] = np.arange(len(permutation))
        clustering = temp_clustering[inverse_permutation]

        ief, beta, c, p, ecg = cas_edge_weights(G, clustering, combine_function=combine_function, normalize=normalize)

        ief_weights += ief
        beta_weights += beta
        c_weights += c
        p_weights += p
        ecg_weights += ecg

    return ief_weights/ens_size, beta_weights/ens_size, c_weights/ens_size, p_weights/ens_size, ecg_weights/ens_size


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