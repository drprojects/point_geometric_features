import numpy as np
from scipy.spatial import KDTree

import pgeof
from tests.helpers import random_nn


def test_knn():
    knn = 10
    xyz = np.random.rand(1000, 3)
    xyz = xyz.astype("float32")
    tree = KDTree(xyz)
    _, k_legacy = tree.query(xyz, k=knn, workers=-1)
    k_new, _ = pgeof.knn_search(xyz, xyz, knn)
    np.testing.assert_equal(k_legacy, k_new)

def test_radius_search():
    knn = 10
    radius = 0.2
    xyz = np.random.rand(1000, 3)
    xyz = xyz.astype("float32")
    tree = KDTree(xyz)
    _, k_legacy = tree.query(xyz, k=knn, distance_upper_bound=radius, workers=-1)
    k_legacy[k_legacy == xyz.shape[0]] = -1
    k_new, _ = pgeof.radius_search(xyz, xyz, radius, knn)
    np.testing.assert_equal(k_legacy, k_new)

def test_pgeof_multiscale():
    # Generate a random synthetic point cloud and NNs
    xyz, nn, nn_ptr = random_nn(10000, 50)

    # with pytest.raises(ValueError):
    # scales = np.array(
    #   [20, 50]
    # )  # scales in decreasing order in order to raise the exception
    # multi = pgeof.compute_features_multiscale(xyz, nn, nn_ptr, scales, False)
    scales = np.array(
        [50, 20]
    )  # scales in decreasing order in order to raise the exception
    multi = pgeof.compute_features_multiscale(xyz, nn, nn_ptr, np.flip(scales), False)
    simple = pgeof.compute_features(xyz, nn, nn_ptr, 50, False)
    multi_simple = pgeof.compute_features_multiscale(xyz, nn, nn_ptr, [20], False)
    np.testing.assert_allclose(multi[:, 0], multi_simple[:, 0], 1e-1, 1e-5)
    np.testing.assert_allclose(multi[:, 1], simple, 1e-1, 1e-5)