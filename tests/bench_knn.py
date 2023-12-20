import numpy as np
import pytest
from pgeof import pgeof
from scipy.spatial import KDTree

import pgeof2


@pytest.fixture
def random_point_cloud():
    return np.random.rand(100000, 3).astype("float32")


@pytest.mark.benchmark(group="knn", disable_gc=True, warmup=True)
def test_knn_scipy(benchmark, random_point_cloud):
    knn = 50

    def _to_bench():
        tree = KDTree(random_point_cloud)
        _ = tree.query(random_point_cloud, k=knn, workers=-1)

    benchmark(_to_bench)


@pytest.mark.benchmark(group="knn", disable_gc=True, warmup=True)
def test_knn_pgeof2(benchmark, random_point_cloud):
    knn = 50

    def _to_bench():
        _ = pgeof2.knn_search(random_point_cloud, random_point_cloud, knn)

    benchmark(_to_bench)
