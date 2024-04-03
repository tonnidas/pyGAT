import numpy as np
import scipy.sparse as sp
import torch

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalized_aggregated_features(features, adj):
    features_adj = normalize_adj(adj) @ features
    features_sim = normalize_adj(similarity_matrix(adj)) @ features
    features_triangles = normalize_adj(count_polygons(adj, 3)) @ features
    
    return sp.hstack((normalize_features(features), 
                          normalize_features(features_adj),
                          normalize_features(features_sim),
                          normalize_features(features_triangles)))

def count_paths(adj, k):
    """
    Count the number of paths of length k between two nodes.
    When k = 2, it signifies the number of common nodes between two nodes.
    """
    paths = adj
    for _ in range(k-1): paths = paths @ adj
    return paths

def similarity_matrix(adj):
    """sim[i,j] = |N(i) union N(j)| / min(|N(i)|, |N(j)|) where N(x) = set of neighbors of node x."""
    deg = np.sum(adj, axis=1).reshape((-1, 1))
    pairwise_min_deg = np.minimum(deg, deg.T)
    common_neighbors = count_paths(adj, 2)
    return common_neighbors / pairwise_min_deg

def count_polygons(adj, k):
    """Count the number of common polygons of size k between two nodes."""
    paths = count_paths(adj, k-1)
    paths.setdiag(0)
    return adj * paths

def similarity_based_adjacency(adj):
    """Set adj[i,j] = 1 if sim[i,j] > max similarity of node i with it's adjacent nodes."""
    sim = similarity_matrix(adj)
    sim.setdiag(0)

    max_adj_sim = np.max(adj * sim, axis=1).reshape((-1, 1))
    tmp = np.subtract(sim, max_adj_sim)
    tmp[tmp > 0], tmp[ tmp < 0] = 1, 0

    adj = adj + tmp
    adj[adj > 0], adj[adj < 0] = 1, 0

    return adj
