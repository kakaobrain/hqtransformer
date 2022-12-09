# ------------------------------------------------------------------------------------
# PRDC computation with torch + numpy.
# Modified from
# https://github.com/clovaai/generative-evaluation-prdc
# Copyright (c) 2020-present NAVER Corp.
# MIT license
# ------------------------------------------------------------------------------------

import numpy as np
import torch

__all__ = ['compute_prdc']


def compute_pairwise_distance_sklearn(data_x, data_y=None):
    """Legacy method to compute pairwise distance (as in original prdc package)
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    import sklearn.metrics

    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def batch_pairwise_distances(U, V):
    """Compute pairwise distances between two batches of feature vectors."""

    # Squared norms of each row in U and V.
    # norm_u as a column and norm_v as a row vectors.
    norm_u = U.pow(2.0).sum(1, keepdim=True)  # shape: (len(U), 1)
    norm_v = V.pow(2.0).sum(1, keepdim=True).transpose(0, 1)  # shape: (1, len(V))

    # Pairwise squared Euclidean distances.
    D = norm_u + norm_v - 2. * (U @ V.t())  # shape: (len(U), len(V))

    return D


def compute_pairwise_distance(data_x,
                              data_y=None,
                              row_batch_size=10000,
                              col_batch_size=10000,
                              ):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if data_y is None:
        data_y = data_x

    n_x = len(data_x)
    n_y = len(data_y)

    dists = np.zeros([n_x, n_y], dtype=np.float32)

    for begin1 in range(0, n_x, row_batch_size):
        end1 = min(begin1 + row_batch_size, n_x)
        row_batch = data_x[begin1:end1]
        row_batch = torch.from_numpy(row_batch).to(device)

        for begin2 in range(0, n_y, col_batch_size):
            end2 = min(begin2 + col_batch_size, n_y)
            col_batch = data_y[begin2:end2]
            col_batch = torch.from_numpy(col_batch).to(device)

            # Compute distances between batches.
            batch_dist = batch_pairwise_distances(row_batch, col_batch)
            dists[begin1:end1, begin2:end2] = batch_dist.cpu().numpy()

    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
        distance_real_fake <
        np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
        distance_real_fake <
        np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
        distance_real_fake <
        np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) <
        real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
