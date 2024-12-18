import numpy as np
import igraph as ig

import sys
sys.path.append('../')
import CAS


def compute_edge_weights(G, cas, clusters, combine_function="min", normalize=True):
    source_to_target = np.empty(G.ecount())
    target_to_source = np.empty(G.ecount())
    for i, e in enumerate(G.es):
        source_to_target[i] = cas[e.source, clusters[e.target]]
        target_to_source[i] = cas[e.target, clusters[e.source]]

    if combine_function=="and":
        weights = source_to_target * target_to_source
    elif combine_function=="or":
        weights = source_to_target + target_to_source - source_to_target*target_to_source
    elif combine_function=="min":
        weights = np.minimum(source_to_target, target_to_source)
    elif combine_function=="mean":
        weights = (source_to_target + target_to_source)/2
    else:
        raise ValueError(f"combine_function must be one of 'and', 'or', 'min', 'mean'. Got {combine_function}")
    
    if normalize:
        return weights / (np.max(weights) - np.min(weights))
    return weights


def cas_edge_weights(G, clusters, combine_function="min", normalize=True):
    graph_matrix = G.get_adjacency_sparse()
    cluster_matrix = CAS.partition2sparse(clusters)
    ief, beta, c, p, degs = CAS.CAS(graph_matrix, cluster_matrix)

    ief_weights = compute_edge_weights(G, ief, clusters, combine_function=combine_function, normalize=normalize)
    beta_weights = compute_edge_weights(G, beta, clusters, combine_function=combine_function, normalize=normalize)
    c_weights = compute_edge_weights(G, c, clusters, combine_function=combine_function, normalize=normalize)
    p_weights = compute_edge_weights(G, p, clusters, combine_function=combine_function, normalize=normalize)

    ecg_weights = compute_edge_weights(G, cluster_matrix, clusters, combine_function=combine_function, normalize=False)
    
    return ief_weights, beta_weights, c_weights, p_weights, ecg_weights


def ensemble_cas_edge_weights(G, ens_size=16, combine_function="min", normalize=True, clustering_method="first_louvain", resolution=0.1):
    ief_weights = np.zeros(G.ecount())
    beta_weights = np.zeros(G.ecount())
    c_weights = np.zeros(G.ecount())
    p_weights = np.zeros(G.ecount())
    ecg_weights = np.zeros(G.ecount())

    for _ in range(ens_size):
        permutation = np.random.permutation(G.vcount())
        g = G.permute_vertices(permutation)
        if clustering_method == "leiden":
            clustering = np.array(g.community_leiden(resolution=resolution).membership)
        elif clustering_method == "louvain":
            clustering = np.array(g.community_louvain(resolution=resolution).membership)
        elif clustering_method == "first_louvain":
            clustering = np.array(g.community_multilevel(return_levels=True)[0].membership)
        else:
            raise ValueError(f"clustering_method expected one of leiden, louvain, or first_louvain. Got {clustering_method}")
        
        ief, beta, c, p, ecg = cas_edge_weights(g, clustering, combine_function=combine_function, normalize=normalize)

        ief_weights += ief
        beta_weights += beta
        c_weights += c
        p_weights += p
        ecg_weights += ecg

    return ief_weights/ens_size, beta_weights/ens_size, c_weights/ens_size, p_weights/ens_size, ecg_weights/ens_size


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