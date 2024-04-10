import numpy as np
import pytest

import pgeof
from pgeof import EFeatureID

# Skip if jakteristics import fail
# it should fail on darwin (macOS) systems
jakteristics = pytest.importorskip("jakteristics")


@pytest.fixture
def random_point_cloud():
    return np.random.rand(1000000, 3) * 100


@pytest.mark.benchmark(group="feature-computation-jak", disable_gc=True, warmup=True)
def test_bench_jak(benchmark, random_point_cloud):
    knn = 50
    dist = 5.0

    def _to_bench_feat():
        _ = jakteristics.compute_features(
            random_point_cloud,
            dist,
            kdtree=None,
            num_threads=-1,
            max_k_neighbors=knn,
            feature_names=["verticality"],
        )

    benchmark(_to_bench_feat)


@pytest.mark.benchmark(group="feature-computation-jak", disable_gc=True, warmup=True)
def test_pgeof(benchmark, random_point_cloud):
    knn = 50
    dist = 5.0

    def _to_bench_feat():
        _ = pgeof.compute_features_selected(
            random_point_cloud, dist, knn, [EFeatureID.Verticality]
        )

    benchmark(_to_bench_feat)
