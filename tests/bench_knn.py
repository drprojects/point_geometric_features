import numpy as np
import pytest
from scipy.spatial import KDTree

import pgeof


@pytest.fixture
def random_point_cloud():
    return np.random.rand(10000000, 3).astype("float32")


@pytest.mark.benchmark(group="knn", disable_gc=True, warmup=True)
def test_knn_scipy(benchmark, random_point_cloud):
    knn = 50

    def _to_bench():
        tree = KDTree(random_point_cloud)
        _ = tree.query(random_point_cloud, k=knn, workers=-1)

    benchmark(_to_bench)


@pytest.mark.benchmark(group="knn", disable_gc=True, warmup=True)
def test_knn_pgeof(benchmark, random_point_cloud):
    knn = 50

    def _to_bench():
        _ = pgeof.knn_search(random_point_cloud, random_point_cloud, knn)

    benchmark(_to_bench)


@pytest.mark.benchmark(group="radius-search", disable_gc=True, warmup=True)
def test_radius_scipy(benchmark, random_point_cloud):
    max_knn = 30
    radius = 0.2

    def _to_bench():
        tree = KDTree(random_point_cloud)
        _ = tree.query(random_point_cloud, k=max_knn, distance_upper_bound=radius, workers=-1)

    benchmark(_to_bench)


@pytest.mark.benchmark(group="radius-search", disable_gc=True, warmup=True)
def test_radius_pgeof(benchmark, random_point_cloud):
    max_knn = 30
    radius = 0.2

    def _to_bench():
        _ = pgeof.radius_search(random_point_cloud, random_point_cloud, radius, max_knn)

    benchmark(_to_bench)
