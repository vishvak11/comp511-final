from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel


def to_numpy(x):
    """Convert a tensor or other array-like to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "to_numpy"):
        return x.to_numpy()
    else:
        return np.asarray(x)


def calculate_ari(embeddings, labels, num_clusters):
    """Compute Adjusted Rand Index (ARI)."""
    kmeans = KMeans(n_clusters=num_clusters)
    pred_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, pred_labels)
    return ari


def calculate_nmi(embeddings, labels, num_clusters):
    """Compute Normalized Mutual Information (NMI)."""
    kmeans = KMeans(n_clusters=num_clusters)
    pred_labels = kmeans.fit_predict(embeddings)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    return nmi


def compute_w2_monge(source, T):
    """Compute 2-Wasserstein distance squared in Monge formulation

    Args:
        source (torch.tensor): source sample
        T (torch.tensor): transport map

    Returns:
        _type_: wasserstein distance
    """

    cost = 0.5 * torch.sum((source - T) * (source - T), dim=1)
    cost = cost.mean()
    return cost


# Computes unweighted MMD using the RBF kernel
def mmd_distance(x, y, gamma):
    x = to_numpy(x)
    y = to_numpy(y)

    xx = rbf_kernel(x, x, gamma)  # apply kernal function for x
    xy = rbf_kernel(x, y, gamma)  # apply kernal function between x and y
    yy = rbf_kernel(y, y, gamma)  # apply kernal function for y

    return xx.mean() + yy.mean() - 2 * xy.mean() # MMD formula 


# computes the weighted scalar MMD for multiple gamma values and returns their average
def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]  # Default gamma values

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    # Average weighted MMD over gammas
    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


