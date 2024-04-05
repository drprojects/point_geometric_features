import numpy as np
import pytest
import sys
from pgeof import pgeof
from scipy.spatial import KDTree

import pgeof2
from tests.helpers import random_nn


def test_knn():
    knn = 10
    xyz = np.random.rand(10000, 3)
    xyz = xyz.astype("float32")
    tree = KDTree(xyz)
    _, k_legacy = tree.query(xyz, k=knn, workers=-1)
    k_new, _ = pgeof2.knn_search(xyz, xyz, knn)
    np.testing.assert_equal(k_legacy, k_new)


def test_pgeof():
    # Generate a random synthetic point cloud and NNs
    xyz, nn, nn_ptr = random_nn(10000, 30)
    legacy = pgeof(xyz, nn, nn_ptr, 1, -1, 1, False)
    legacy = legacy[:, :11]
    new = pgeof2.compute_features(xyz, nn, nn_ptr, 1, False)
    np.testing.assert_allclose(new, legacy, 1e-1, 1e-5)


def test_pgeof2_multiscale():
    # Generate a random synthetic point cloud and NNs
    xyz, nn, nn_ptr = random_nn(10000, 50)

    # with pytest.raises(ValueError):
    # scales = np.array(
    #   [20, 50]
    # )  # scales in decreasing order in order to raise the exception
    # multi = pgeof2.compute_features_multiscale(xyz, nn, nn_ptr, scales, False)
    scales = np.array(
        [50, 20]
    )  # scales in decreasing order in order to raise the exception
    multi = pgeof2.compute_features_multiscale(xyz, nn, nn_ptr, np.flip(scales), False)
    simple = pgeof2.compute_features(xyz, nn, nn_ptr, 50, False)
    multi_simple = pgeof2.compute_features_multiscale(xyz, nn, nn_ptr, [20], False)
    np.testing.assert_allclose(multi[:, 0], multi_simple[:, 0], 1e-1, 1e-5)
    np.testing.assert_allclose(multi[:, 1], simple, 1e-1, 1e-5)

@pytest.mark.xfail(sys.platform == "linux", reason="UB in pgeof package")
def test_pgeof_optimal():
    # Generate a random synthetic point cloud and NNs
    xyz, nn, nn_ptr = random_nn(10000, 50)
    old_pgeof = pgeof(xyz, nn, nn_ptr, 25, 3, 20, False)
    new_pgeof = pgeof2.compute_features_optimal(xyz, nn, nn_ptr, 25, 3, 20, False)
    np.testing.assert_allclose(old_pgeof, new_pgeof, 1e-1, 1e-5)
