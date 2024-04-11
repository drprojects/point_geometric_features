
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "nn_search.hpp"
#include "pgeof.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pgeof_ext, m)
{
    m.doc() =
        "Compute, for each point in a 3D point cloud, local geometric "
        "features "
        "in parallel on CPU";
    nb::enum_<pgeof::EFeatureID>(m, "EFeatureID")
        .value("Linearity", pgeof::EFeatureID::Linearity)
        .value("Planarity", pgeof::EFeatureID::Planarity)
        .value("Scattering", pgeof::EFeatureID::Scattering)
        .value("VerticalityPGEOF", pgeof::EFeatureID::VerticalityPGEOF)
        .value("Normal_x", pgeof::EFeatureID::Normal_x)
        .value("Normal_y", pgeof::EFeatureID::Normal_y)
        .value("Normal_z", pgeof::EFeatureID::Normal_z)
        .value("Length", pgeof::EFeatureID::Length)
        .value("Surface", pgeof::EFeatureID::Surface)
        .value("Volume", pgeof::EFeatureID::Volume)
        .value("Curvature", pgeof::EFeatureID::Curvature)
        .value("K_optimal", pgeof::EFeatureID::K_optimal)  // TODO: remove or handle
        .value("Verticality", pgeof::EFeatureID::Verticality)
        .value("Eigentropy", pgeof::EFeatureID::Eigentropy)
        .export_values();
    m.def(
        "compute_features", &pgeof::compute_geometric_features<float>, "xyz"_a.noconvert(), "nn"_a.noconvert(),
        "nn_ptr"_a.noconvert(), "k_min"_a = 1, "verbose"_a = false, R"(
            Compute a set of geometric features for a point cloud from a precomputed list of neighbors.

            * The following features are computed:
            - linearity
            - planarity
            - scattering
            - verticality
            - normal vector (oriented towards positive z-coordinates)
            - length
            - surface
            - volume
            - curvature
            :param xyz: The point cloud. A numpy array of shape (n, 3).
            :param nn: Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
            '-1' indices will either crash or silently compute incorrect features.
            :param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
            are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'.
            :param k_min: Minimum number of neighbors to consider for features computation. If a point has less,
            its features will be a set of '0' values.
            :param verbose: Whether computation progress should be printed out
            :return: the geometric features associated with each point's neighborhood in a (num_points, features_count) numpy array.
        )");
    m.def(
        "compute_features_multiscale", &pgeof::compute_geometric_features_multiscale<float>, "xyz"_a.noconvert(),
        "nn"_a.noconvert(), "nn_ptr"_a.noconvert(), "k_scales"_a, "verbose"_a = false, R"(
            Compute a set of geometric features for a point cloud in a multiscale fashion.
            
            * The following features are computed:
            - linearity
            - planarity
            - scattering
            - verticality
            - normal vector (oriented towards positive z-coordinates)
            - length
            - surface
            - volume
            - curvature
            
            :param xyz: The point cloud. A numpy array of shape (n, 3).
            :param nn: Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
            '-1' indices will either crash or silently compute incorrect features.
            :param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
            are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'.
            :param k_scale: Array of number of neighbors to consider for features computation. If a at a given scale, a point has
            less features will be a set of '0' values.
            :param verbose: Whether computation progress should be printed out
            :return: Geometric features associated with each point's neighborhood in a (num_points, features_count, n_scales)
            numpy array.
        )");
    m.def(
        "compute_features_optimal", &pgeof::compute_geometric_features_optimal<float>, "xyz"_a.noconvert(),
        "nn"_a.noconvert(), "nn_ptr"_a.noconvert(), "k_min"_a = 1, "k_step"_a = 1, "k_min_search"_a = 1,
        "verbose"_a = false, R"(
            Compute a set of geometric features for a point cloud using the optimal neighborhood selection described in
            http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf

            * The following features are computed:
            - linearity
            - planarity
            - scattering
            - verticality
            - normal vector (oriented towards positive z-coordinates)
            - length
            - surface
            - volume
            - curvature
            - optimal_nn
            :param xyz: the point cloud
            :param nn: Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
            '-1' indices will either crash or silently compute incorrect features.
            :param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
            are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'.
            :param k_min: Minimum number of neighbors to consider for features computation. If a point has less,
            its features will be a set of '0' values.
            :param k_step: Step size to take when searching for the optimal neighborhood, size for each point following
            Weinmann, 2015
            :param k_min_search: Minimum neighborhood size at which to start when searching for the optimal neighborhood size for
            each point. It is advised to use a value of 10 or higher, for geometric features robustness.
            :param verbose: Whether computation progress should be printed out
            :return: Geometric features associated with each point's neighborhood in a (num_points, features_count) numpy array.
        )");
    m.def("knn_search", &pgeof::nanoflann_knn_search<float>, "data"_a.noconvert(), "query"_a.noconvert(), "knn"_a, R"(
        Given two point clouds, compute for each point present in one of the point cloud 
        the N closest points in the other point cloud

        It should be faster than scipy.spatial.KDTree for this task.
        
        :param data: the reference point cloud. A numpy array of shape (n, 3).
        :param query: the point cloud used for the queries. A numpy array of shape (n, 3).
        :param knn: the number of neighbors to take into account for each point.
        :return: a pair of arrays, both of size (n_points x knn), the first one contains the indices of each neighbor, the
        second one the square distances between the query point and each of its neighbors.
    )");
    m.def(
        "radius_search", &pgeof::nanoflann_radius_search<float>, "data"_a.noconvert(), "query"_a.noconvert(),
        "search_radius"_a, "knn"_a = 200, R"(
            Search for the points within a specified sphere in a point cloud.
            
            It could be a fallback replacement for FRNN into SuperPointTransformer code base.
            It should be faster than scipy.spatial.KDTree for this task.
            
            :param data: the reference point cloud. A numpy array of shape (n, 3).
            :param query: the point cloud used for the queries (sphere centers). A numpy array of shape (n, 3).
            :param search_radius: the search radius.
            :param max_knn: the maximum number of neighbors to fetch inside the radius. The central point is included. Fixing a
            reasonable max number of neighbors prevents running OOM for large radius/dense point clouds.
            :return: a pair of arrays, both of size (n_points x knn), the first one contains the 'indices' of each neighbor,
            the second one the 'square_distances' between the query point and each neighbor. Point having a number of neighbors <
            'max_knn' inside the 'search_radius' will have their 'indices' and and 'square_distances' filled respectively with
            '-1' and 'O' for any missing neighbor.
        )");
    m.def(
        "compute_features_selected", &pgeof::compute_geometric_features_selected<double>, "xyz"_a.noconvert(),
        "search_radius"_a, "knn"_a, "selected_features"_a, R"(
            Compute a selected set of geometric features for a point cloud via radius search.

            This function aims to mimick the behavior of jakteristics and provide an efficient way
            to compute a limited set of features (double precision version).
            
            :param xyz: the point cloud. A numpy array of shape (n, 3).
            :param search_radius: the search radius. A numpy array of shape (n, 3).
            :param max_knn: the maximum number of neighbors to fetch inside the sphere. The central point is included. Fixing a
            reasonable max number of neighbors prevents running OOM for large radius/dense point clouds.
            :param selected_features: List of selected features. See EFeatureID
            :return: Geometric features associated with each point's neighborhood in a (num_points, features_count) numpy array.
        )");
    m.def(
        "compute_features_selected", &pgeof::compute_geometric_features_selected<float>, "xyz"_a.noconvert(),
        "search_radius"_a, "knn"_a, "selected_features"_a, R"(
            Compute a selected set of geometric features for a point cloud via radius search.

            This function aims to mimic the behavior of jakteristics and provide an efficient way
            to compute a limited set of features (float precision version).
            
            :param xyz: the point cloud
            :param search_radius: the search radius.
            :param max_knn: the maximum number of neighbors to fetch inside the sphere. The central point is included. Fixing a
            reasonable max number of neighbors prevents running OOM for large radius/dense point clouds.
            :param selected_features: List of selected features. See EFeatureID
            :return: Geometric features associated with each point's neighborhood in a (num_points, features_count) numpy array.
        )");
}