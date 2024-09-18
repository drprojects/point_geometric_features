#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <Eigen/Dense>
#include <iostream>
#include <nanoflann.hpp>
#include <taskflow/algorithm/for_each.hpp>

#include "pca.hpp"
namespace nb = nanobind;

namespace pgeof
{

/**
 * Given two point clouds, compute for each point present in one of the point cloud
 * the N closest points in the other point cloud
 *
 * It should be faster than scipy.spatial.KDTree for this task.
 *
 * @param data the reference point cloud.
 * @param query the point cloud used for the queries.
 * @param knn the number of neighbors to take into account for each point.
 * @return a pair of nd::array, both of size (n_points x knn), the first one contains the indices of each neighbor, the
 * second one the square distances between the query point and each of its neighbors.
 */
template <typename real_t>
static std::pair<nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>, nb::ndarray<nb::numpy, real_t, nb::ndim<2>>>
    nanoflann_knn_search(RefCloud<real_t> data, RefCloud<real_t> query, const uint32_t knn)
{
    using kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<RefCloud<real_t>, 3, nanoflann::metric_L2_Simple>;

    if (knn > data.rows()) { throw std::invalid_argument("knn size is greater than the data point cloud size"); }

    kd_tree_t          kd_tree(3, data, 10, 0);
    const Eigen::Index n_points = query.rows();
    uint32_t*          indices  = new uint32_t[knn * n_points];
    nb::capsule        owner_indices(indices, [](void* p) noexcept { delete[] (uint32_t*)p; });

    real_t*     sqr_dist = new real_t[knn * n_points];
    nb::capsule owner_dist(sqr_dist, [](void* p) noexcept { delete[] (real_t*)p; });

    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.for_each_index(
        Eigen::Index(0), n_points, Eigen::Index(1),
        [&](Eigen::Index point_id)
        {
            nanoflann::KNNResultSet<real_t, uint32_t, uint32_t> result_set(knn);

            const size_t id = point_id * knn;
            result_set.init(&indices[id], &sqr_dist[id]);
            kd_tree.index_->findNeighbors(result_set, query.row(point_id).data());
        }),
        tf::StaticPartitioner(0);

    executor.run(taskflow).get();

    const size_t shape[2] = {static_cast<size_t>(n_points), static_cast<size_t>(knn)};
    return {
        nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>(indices, 2, shape, owner_indices),
        nb::ndarray<nb::numpy, real_t, nb::ndim<2>>(sqr_dist, 2, shape, owner_dist)};
};

/**
 * Search for the points within a specified sphere in a point cloud.
 *
 * It could be a fallback replacement for FRNN into SuperPointTransformer code base.
 * It should be faster than scipy.spatial.KDTree for this task.
 *
 * @param data the reference point cloud.
 * @param query the point cloud used for the queries (sphere centers)
 * @param search_radius the search radius.
 * @param max_knn the maximum number of neighbors to fetch inside the radius. (Fixing a
 * reasonable max number of neighbors prevents running OOM for large radius/dense point clouds.)
 * @return a pair of nd::array, both of size (n_points x knn), the first one contains the 'indices' of each neighbor,
 * the second one the 'square_distances' between the query point and each neighbor. Point having a number of neighbors <
 * 'max_knn' inside the 'search_radius' will have their 'indices' and and 'square_distances' filled respectively with
 * '-1' and 'O' for any missing neighbor.
 */
template <typename real_t>
static std::pair<nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>, nb::ndarray<nb::numpy, real_t, nb::ndim<2>>>
    nanoflann_radius_search(
        RefCloud<real_t> data, RefCloud<real_t> query, const real_t search_radius, const uint32_t max_knn)
{
    using kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<RefCloud<real_t>, 3, nanoflann::metric_L2_Simple>;

    if (max_knn > data.rows())
    {
        throw std::invalid_argument("max knn size is greater than the data point cloud size");
    }

    kd_tree_t    kd_tree(3, data, 10, 0);
    const real_t sq_search_radius = search_radius * search_radius;

    const Eigen::Index n_points = query.rows();

    int32_t*    indices = new int32_t[max_knn * n_points];
    nb::capsule owner_indices(indices, [](void* p) noexcept { delete[] (int32_t*)p; });
    std::fill(indices, indices + (max_knn * n_points), -1);

    real_t*     sqr_dist = new real_t[max_knn * n_points];
    nb::capsule owner_dist(sqr_dist, [](void* p) noexcept { delete[] (real_t*)p; });
    std::fill(sqr_dist, sqr_dist + (max_knn * n_points), real_t(0.0));

    tf::Executor executor;
    tf::Taskflow taskflow;

    taskflow.for_each_index(
        Eigen::Index(0), n_points, Eigen::Index(1),
        [&](Eigen::Index point_id)
        {
            nanoflann::RKNNResultSet<real_t, int32_t, uint32_t> result_set(max_knn, sq_search_radius);

            const size_t id = point_id * max_knn;

            result_set.init(&indices[id], &sqr_dist[id]);
            kd_tree.index_->findNeighbors(result_set, query.row(point_id).data());
        },
        tf::StaticPartitioner(0));

    executor.run(taskflow).get();

    const size_t shape[2] = {static_cast<size_t>(n_points), static_cast<size_t>(max_knn)};
    return {
        nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>(indices, 2, shape, owner_indices),
        nb::ndarray<nb::numpy, real_t, nb::ndim<2>>(sqr_dist, 2, shape, owner_dist)};
};

}  // namespace pgeof
