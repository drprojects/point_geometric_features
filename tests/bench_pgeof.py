import pytest
from pgeof import pgeof

import pgeof2
from tests.helpers import random_nn


@pytest.fixture
def nn_data():
    return random_nn(1000000, 50)


@pytest.mark.benchmark(group="feature-computation", disable_gc=True, warmup=True)
def test_pgeof(benchmark, nn_data):
    # Generate a random synthetic point cloud and NNs
    xyz, nn, nn_ptr = nn_data

    def _to_bench_feat():
        _ = pgeof(xyz, nn, nn_ptr, 1, -1, 1, False)

    benchmark(_to_bench_feat)


@pytest.mark.benchmark(group="feature-computation", disable_gc=True, warmup=True)
def test_pgeof2(benchmark, nn_data):
    xyz, nn, nn_ptr = nn_data

    def _to_bench_feat():
        _ = pgeof2.compute_features(xyz, nn, nn_ptr, 1, False)

    benchmark(_to_bench_feat)


@pytest.mark.benchmark(group="feature-computation-optimal", disable_gc=True, warmup=True)
def test_pgeof_optimal(benchmark, nn_data):
    # Generate a random synthetic point cloud and NNs
    xyz, nn, nn_ptr = nn_data

    def _to_bench_feat():
        _ = pgeof(xyz, nn, nn_ptr, 25, 3, 20, False)

    benchmark(_to_bench_feat)


@pytest.mark.benchmark(group="feature-computation-optimal", disable_gc=True, warmup=True)
def test_pgeof2_optimal(benchmark, nn_data):
    xyz, nn, nn_ptr = nn_data

    def _to_bench_feat():
        _ = pgeof2.compute_features_optimal(xyz, nn, nn_ptr, 25, 3, 20, False)

    benchmark(_to_bench_feat)