#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>
#include <vector>

#include "pca.hpp"

namespace nb = nanobind;

namespace pgeof
{

namespace log
{
/**
 * Log a progress into the std::cout. It has multiple quirks (std::cout usage, race condition)
 * but it is mostly used for debugging purposes.
 *
 * @param progress_count a reference to the current count. it is incremented by one at each call of this function
 * @param progress_total the total expected count of iteration
 */
static inline void progress(size_t& progress_count, const size_t progress_total)
{
    ++progress_count;
    // Print progress
    // NB: when in parallel progress_count behavior is undefined, but
    // gives a good indication of progress
    if (progress_count % 10000 == 0)
    {
        std::cout << progress_count << "% done          \r" << std::flush;
        std::cout << std::ceil(progress_count * 100 / progress_total) << "% done          \r" << std::flush;
    }
};

/**
 * flush the logger
 */
static inline void flush() { std::cout << std::endl; };
}  // namespace log

/**
 * Compute a set of geometric features for a point cloud from a precomputed list of neighbors.
 *
 *  * The following features are computed:
 * - linearity
 * - planarity
 * - scattering
 * - verticality
 * - normal vector (oriented towards positive z-coordinates)
 * - length
 * - surface
 * - volume
 * - curvature
 *
 * @param xyz The point cloud.
 * @param nn Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
 * '-1' indices will either crash or silently compute incorrect features.
 * @param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
 * are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'.
 * @param k_min Minimum number of neighbors to consider for features computation. If a point has less,
 * it features will be a set of '0' values.
 * @param verbose Whether computation progress should be printed out
 * @return the geometric features associated with each point's neighborhood in a (num_points, features_count) nd::array.
 */
template <typename real_t = float, const size_t feature_count = 11>
static nb::ndarray<nb::numpy, real_t, nb::shape<-1, static_cast<nb::ssize_t>(feature_count)>>
    compute_geometric_features(
        RefCloud<real_t> xyz, nb::ndarray<const uint32_t, nb::ndim<1>> nn,
        nb::ndarray<const uint32_t, nb::ndim<1>> nn_ptr, const size_t k_min, const bool verbose)
{
    if (k_min < 1) { throw std::invalid_argument("k_min should be > 1"); }
    // Each point can be treated in parallel
    const size_t    n_points    = nn_ptr.size() - 1;  // number of points is not determined by xyz
    size_t          s_point     = 0;
    const uint32_t* nn_data     = nn.data();
    const uint32_t* nn_ptr_data = nn_ptr.data();

    real_t*     features = (real_t*)calloc(n_points * feature_count, sizeof(real_t));
    nb::capsule owner_features(features, [](void* f) noexcept { delete[] (real_t*)f; });

    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.for_each_index(
        size_t(0), size_t(n_points), size_t(1),
        [&](size_t i_point)
        {
            if (verbose) log::progress(s_point, n_points);

            // Recover the points' total number of neighbors
            const size_t k_nn = static_cast<size_t>(nn_ptr_data[i_point + 1] - nn_ptr_data[i_point]);

            // If the cloud has less than k_min point, continue
            if (k_nn >= k_min)
            {
                const PCAResult<real_t> pca = pca_from_neighborhood(xyz, nn_data, nn_ptr_data, i_point, k_nn);
                compute_features(pca, &features[i_point * feature_count]);
            }
        },
        tf::StaticPartitioner(0));
    executor.run(taskflow).get();

    // Final print to start on a new line
    if (verbose) log::flush();
    const size_t shape[2] = {n_points, feature_count};
    return nb::ndarray<nb::numpy, real_t, nb::shape<-1, static_cast<nb::ssize_t>(feature_count)>>(
        features, 2, shape, owner_features);
}
/**
 * Convenience function that check that scales are well ordered in increasing order.
 *
 * @param k_scales the list of scale size (number of neighbors).
 */
static bool check_scales(const std::vector<uint32_t>& k_scales)
{
    uint32_t previous_scale = 1;  // minimal admissible k_min value is 1
    for (const auto& current_scale : k_scales)
    {
        if (current_scale < previous_scale) { return false; }
        previous_scale = current_scale;
    }
    return true;
}

/**
 * Compute a set of geometric features for a point cloud in a multiscale fashion.
 *
 * The following features are computed:
 * - linearity
 * - planarity
 * - scattering
 * - verticality
 * - normal vector (oriented towards positive z-coordinates)
 * - length
 * - surface
 * - volume
 * - curvature
 *
 * @param xyz The point cloud
 * @param nn Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
 *  '-1' indices will either crash or silently compute incorrect features.
 * @param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
 *  are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'.
 * @param k_scale Array of number of neighbors to consider for features computation. If a at a given scale, a point has
 * less features will be a set of '0' values.
 * @param verbose Whether computation progress should be printed out
 * @return Geometric features associated with each point's neighborhood in a (num_points, features_count, n_scales)
 * nd::array
 */
template <typename real_t, const size_t feature_count = 11>
static nb::ndarray<nb::numpy, real_t, nb::shape<-1, -1, static_cast<nb::ssize_t>(feature_count)>>
    compute_geometric_features_multiscale(
        RefCloud<real_t> xyz, nb::ndarray<const uint32_t, nb::ndim<1>> nn,
        nb::ndarray<const uint32_t, nb::ndim<1>> nn_ptr, const std::vector<uint32_t>& k_scales, const bool verbose)
{
    if (!check_scales(k_scales))
    {
        throw std::invalid_argument("k_scales should be > 1 and sorted in ascending order");
    }
    const size_t    n_points    = nn_ptr.size() - 1;  // number of points is not determined by xyz
    const size_t    n_scales    = k_scales.size();
    size_t          s_point     = 0;
    const uint32_t* nn_data     = nn.data();
    const uint32_t* nn_ptr_data = nn_ptr.data();

    real_t*     features = (real_t*)calloc(n_points * n_scales * feature_count, sizeof(real_t));
    nb::capsule owner_features(features, [](void* f) noexcept { delete[] (real_t*)f; });

    // Each point can be treated in parallel
    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.for_each_index(
        size_t(0), size_t(n_points), size_t(1),
        [&](size_t i_point)
        {
            if (verbose) log::progress(s_point, n_points);
            // Recover the points' total number of neighbors
            const size_t k_nn = static_cast<size_t>(nn_ptr_data[i_point + 1] - nn_ptr_data[i_point]);

            for (size_t i_scale = 0; i_scale < n_scales; ++i_scale)
            {
                const size_t knn_scale = static_cast<size_t>(k_scales[i_scale]);

                if (k_nn < knn_scale)
                    break;  // we assume scales are stored in increasing order,
                            // so we could do an early break in case of k_nn <
                            // knn_scale
                const PCAResult<real_t> pca = pca_from_neighborhood(xyz, nn_data, nn_ptr_data, i_point, knn_scale);
                compute_features(pca, &features[(i_point * n_scales + i_scale) * feature_count]);
            }
        },
        tf::StaticPartitioner(0));

    executor.run(taskflow).get();

    // Final print to start on a new line
    if (verbose) log::flush();

    const size_t shape[3] = {n_points, n_scales, feature_count};
    return nb::ndarray<nb::numpy, real_t, nb::shape<-1, -1, static_cast<nb::ssize_t>(feature_count)>>(
        features, 3, shape, owner_features);
}

/**
 * Compute a set of geometric features for a point cloud using the optimal neighborhood selection described in
 * http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf
 *
 *  * The following features are computed:
 * - linearity
 * - planarity
 * - scattering
 * - verticality
 * - normal vector (oriented towards positive z-coordinates)
 * - length
 * - surface
 * - volume
 * - curvature
 * - optimal_nn
 *
 * @param xyz The point cloud
 * @param nn Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
 *  '-1' indices will either crash or silently compute incorrect features.
 * @param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
 *  are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'.
 * @param k_min Minimum number of neighbors to consider for features computation. If a point has less,
 * its features will be a set of '0' values.
 * @param k_step Step size to take when searching for the optimal neighborhood, size for each point following
 * Weinmann, 2015
 * @param k_min_search Minimum neighborhood size at which to start when searching for the optimal neighborhood size for
 each point. It is advised to use a value of 10 or higher, for geometric features robustness.
 * @param verbose Whether computation progress should be printed out
 * @return Geometric features associated with each point's neighborhood in a (num_points, features_count) nd::array
 */
template <typename real_t, const size_t feature_count = 12>
static nb::ndarray<nb::numpy, real_t, nb::shape<-1, static_cast<nb::ssize_t>(feature_count)>>
    compute_geometric_features_optimal(
        RefCloud<real_t> xyz, nb::ndarray<const uint32_t, nb::ndim<1>> nn,
        nb::ndarray<const uint32_t, nb::ndim<1>> nn_ptr, const uint32_t k_min, const uint32_t k_step,
        const uint32_t k_min_search, const bool verbose)
{
    if (k_min < 1 && k_min_search < 1) { throw std::invalid_argument("k_min and k_min_search should be > 1"); }
    // Each point can be treated in parallel
    const size_t    n_points    = nn_ptr.size() - 1;  // number of points is not determined by xyz
    size_t          s_point     = 0;
    const uint32_t* nn_data     = nn.data();
    const uint32_t* nn_ptr_data = nn_ptr.data();

    real_t*     features = (real_t*)calloc(n_points * feature_count, sizeof(real_t));
    nb::capsule owner_features(features, [](void* f) noexcept { delete[] (real_t*)f; });

    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.for_each_index(
        size_t(0), size_t(n_points), size_t(1),
        [&](size_t i_point)
        {
            if (verbose) log::progress(s_point, n_points);

            // Recover the points' total number of neighbors
            const size_t k_nn = static_cast<size_t>(nn_ptr_data[i_point + 1] - nn_ptr_data[i_point]);

            // Process only if the cloud has the required number of point
            if (k_nn >= k_min && k_nn >= k_min_search)
            {
                size_t k0 = std::min(std::max(static_cast<size_t>(k_min), static_cast<size_t>(k_min_search)), k_nn);

                PCAResult<real_t> pca_optimal;
                real_t            eigenentropy_optimal = real_t(1.0);
                size_t            k_optimal            = k_nn;
                for (size_t k = k0; k <= k_nn; ++k)
                {
                    // Only evaluate the neighborhood's PCA every 'k_step'
                    // and at the boundary values: k0 and k_nn
                    if ((k > k0) && (k % k_step != 0) && (k != k_nn)) { continue; }

                    const PCAResult<real_t> pca          = pca_from_neighborhood(xyz, nn_data, nn_ptr_data, i_point, k);
                    const real_t            eigenentropy = compute_eigentropy(pca);
                    // Keep track of the optimal neighborhood size with the
                    // lowest eigenentropy
                    if ((k == k0) || (eigenentropy < eigenentropy_optimal))
                    {
                        eigenentropy_optimal = eigenentropy;
                        k_optimal            = k;
                        pca_optimal          = pca;
                    }
                }
                compute_features(pca_optimal, &features[i_point * feature_count]);
                // Add best nn
                features[i_point * feature_count + 11] = real_t(k_optimal);
            }
        },
        tf::StaticPartitioner(0));

    executor.run(taskflow).get();

    if (verbose) log::flush();

    const size_t shape[2] = {n_points, feature_count};
    return nb::ndarray<nb::numpy, real_t, nb::shape<-1, static_cast<nb::ssize_t>(feature_count)>>(
        features, 2, shape, owner_features);
}

/**
 * Compute a selected set of geometric features for a point cloud via radius search.
 *
 * This function aims to mimic the behavior of jakteristics and provide an efficient way
 * to compute a limited set of features.
 *
 * @param xyz The point cloud
 * @param search_radius the search radius.
 * @param max_knn the maximum number of neighbors to fetch inside the radius. The central point is included. Fixing a
 * reasonable max number of neighbors prevents running OOM for large radius/dense point clouds.
 * @param selected_features the list of selected features. See pgeof::EFeatureID
 * @return Geometric features associated with each point's neighborhood in a (num_points, features_count) nd::array
 */
template <typename real_t>
static nb::ndarray<nb::numpy, real_t, nb::shape<-1, -1>> compute_geometric_features_selected(
    RefCloud<real_t> xyz, const real_t search_radius, const uint32_t max_knn,
    const std::vector<EFeatureID>& selected_features)
{
    using kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<RefCloud<real_t>, 3, nanoflann::metric_L2_Simple>;
    // TODO: where knn < num of points

    kd_tree_t          kd_tree(3, xyz, 10, 0);
    const size_t       feature_count    = selected_features.size();
    const Eigen::Index n_points         = xyz.rows();
    real_t             sq_search_radius = search_radius * search_radius;

    real_t*     features = (real_t*)calloc(n_points * feature_count, sizeof(real_t));
    nb::capsule owner_features(features, [](void* f) noexcept { delete[] (real_t*)f; });

    tf::Executor executor;
    tf::Taskflow taskflow;

    taskflow.for_each_index(
        Eigen::Index(0), n_points, Eigen::Index(1),
        [&](Eigen::Index point_id)
        {
            std::vector<nanoflann::ResultItem<Eigen::Index, real_t>> result_set;

            nanoflann::RadiusResultSet<real_t, Eigen::Index> radius_result_set(sq_search_radius, result_set);
            const auto                                       num_found =
                kd_tree.index_->radiusSearchCustomCallback(xyz.row(point_id).data(), radius_result_set);

            // not enough point, no feature computation
            if (num_found < 2) return;

            // partial sort for max_knn
            if (num_found > max_knn)
            {
                std::partial_sort(
                    result_set.begin(), result_set.begin() + max_knn, result_set.end(), nanoflann::IndexDist_Sorter());
            }

            const size_t num_nn = std::min(static_cast<uint32_t>(num_found), max_knn);

            PointCloud<real_t> cloud(num_nn, 3);
            for (size_t id = 0; id < num_nn; ++id) { cloud.row(id) = xyz.row(result_set[id].first); }
            const PCAResult<real_t> pca = pca_from_pointcloud(cloud);
            compute_selected_features(pca, selected_features, &features[point_id * feature_count]);
        });
    executor.run(taskflow).get();

    return nb::ndarray<nb::numpy, real_t, nb::shape<-1, -1>>(
        features, {static_cast<size_t>(n_points), feature_count}, owner_features);
}
}  // namespace pgeof
