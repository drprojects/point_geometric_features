#pragma once

#include <nanobind/eigen/dense.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace nb = nanobind;

namespace pgeof
{

// Type definitions
template <typename real_t>
using PointCloud = Eigen::Matrix<real_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <typename real_t>
using RefCloud = Eigen::Ref<const PointCloud<real_t>>;

template <typename real_t>
using Vec3 = Eigen::RowVector<real_t, 3>;

template <typename real_t>
using MatrixCloud = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename real_t>
using DRefMatrixCloud = nb::DRef<const MatrixCloud<real_t>>;

// epsilon definition, for now same for float an double
// the eps is meant to stabilize the division when the cloud's 3rd eigenvalue is near 0
template <typename real_t> constexpr real_t epsilon;
template <> constexpr float  epsilon<float>  = 1e-3f;
template <> constexpr double epsilon<double> = 1e-3;

template <typename real_t>
struct PCAResult
{
    Vec3<real_t> val;
    Vec3<real_t> v0;
    Vec3<real_t> v1;
    Vec3<real_t> v2;
};

// enum of features
typedef enum EFeatureID
{
    Linearity = 0,
    Planarity,
    Scattering,
    VerticalityPGEOF,  // Formula is different from the "classical" formula
    Normal_x,  // Normal as the third eigenvector
    Normal_y,
    Normal_z,
    Length,
    Surface,
    Volume,
    Curvature,
    K_optimal,
    Verticality,  // this is the "classical" verticality
    Eigentropy
} EFeatureID;

/**
 * Given A point cloud compute a PCAResult
 *
 * @param cloud the point cloud
 * @returns A PCAResult
 */
template <typename real_t>
static inline PCAResult<real_t> pca_from_pointcloud(const PointCloud<real_t>& cloud)
{
    // Compute the (3, 3) covariance matrix
    const PointCloud<real_t>          centered_cloud = cloud.rowwise() - cloud.colwise().mean();
    const Eigen::Matrix<real_t, 3, 3> cov = (centered_cloud.adjoint() * centered_cloud) / real_t(cloud.rows());

    // Compute the eigenvalues and eigenvectors of the covariance
    Eigen::EigenSolver<Eigen::Matrix<real_t, 3, 3>> es(cov);

    // Sort the values and vectors in order of increasing eigenvalue
    const auto ev = es.eigenvalues().real();

    std::array<Eigen::Index, 3> indices = {0, 1, 2};

    std::sort(
        std::begin(indices), std::end(indices), [&](Eigen::Index i1, Eigen::Index i2) { return ev(i1) > ev(i2); });

    Vec3<real_t> val = {
        (std::max(ev(indices[0]), real_t(0.))), (std::max(ev(indices[1]), real_t(0.))),
        (std::max(ev(indices[2]), real_t(0.)))};
    Vec3<real_t> v0 = es.eigenvectors().col(indices[0]).real();
    Vec3<real_t> v1 = es.eigenvectors().col(indices[1]).real();
    Vec3<real_t> v2 = es.eigenvectors().col(indices[2]).real();

    // To standardize the orientation of eigenvectors, we choose to enforce all eigenvectors
    // to be expressed in the Z+ half-space.
    // Only the third eigenvector (v2) needs to be reoriented because it is the
    // only one used in further computations.
    // TODO: In case we want to orient normal, this should be improved
    if (v2(2) < real_t(0.)) { v2 = real_t(-1.) * v2; }
    return {val, v0, v1, v2};
};

/**
 * Given A point cloud and a CSR definition of the neighboring information for each point, compute a PCAResult
 *
 * @param xyz the point cloud
 * @param nn Integer 1D array. Flattened neighbor indices. Make sure those are all positive,
 *  '-1' indices will either crash or silently compute incorrect
 * @param nn_ptr: [n_points+1] Integer 1D array. Pointers wrt 'nn'. More specifically, the neighbors of point 'i'
 *  are 'nn[nn_ptr[i]:nn_ptr[i + 1]]'
 * @param i_point the index of the 'central point' or  point
 * @param k_nnn the number of neighbors to take into account to compute the PCA. It's the caller responsibility
 * to ensure k_nn won't overflow nn_ptr array.
 * @returns A PCAResult
 */
template <typename real_t, typename index_t>
static PCAResult<real_t> pca_from_neighborhood(
    RefCloud<real_t> xyz, const index_t* nn, const index_t* nn_ptr, const size_t i_point, const size_t k_nn)
{
    // Initialize the cloud (n_neighbors, 3) matrix holding the
    // points' neighbors XYZ coordinates
    PointCloud<real_t> cloud(k_nn, 3);
    // Recover the neighbors' XYZ coordinates using nn and xyz
    for (size_t i_nei = 0; i_nei < k_nn; i_nei++)
    {
        // Recover the neighbor's position in the xyz vector
        const Eigen::Index idx_nei = static_cast<Eigen::Index>(nn[nn_ptr[i_point] + i_nei]);
        // Recover the corresponding xyz coordinates
        cloud.row(i_nei) = xyz.row(idx_nei);
    }
    return pca_from_pointcloud(cloud);
};

/**
 * Given a PCA result compute the eigentropy
 *
 * This Eigentropy is used for neighborhood size selection in some function
 * and could be used a an individual feature as well
 *
 * @param pca PCAResult
 * @return the eigentropy
 */
template <typename real_t>
static inline real_t compute_eigentropy(const PCAResult<real_t>& pca)
{
    // Compute the eigentropy as defined in:
    // http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf
    const real_t       val_sum = pca.val.sum() + epsilon<real_t>;
    const Vec3<real_t> e       = pca.val / val_sum;
    return (-e(0) * std::log(e(0) + epsilon<real_t>) - e(1) * std::log(e(1) + epsilon<real_t>) - e(2) * std::log(e(2) + epsilon<real_t>));
};

/**
 * Given a PCA result compute a full set of feature (the initial set of feature r)
 *
 * This function intends on mimicking the behavior of original PGEOF feature computation
 *
 * @param[in] pca PCAResult
 * @param[out] feature_results the array of resulting features.
 */
template <typename real_t>
static void compute_features(const PCAResult<real_t>& pca, real_t* features)
{
    constexpr real_t sq_eps    = real_t(1e-6);
    constexpr real_t cub_eps   = real_t(1e-9);
    constexpr real_t one_third = real_t(1.) / real_t(3.);

    // Compute the dimensionality features. The eps term is meant
    // to stabilize the division when the cloud's 3rd eigenvalue is
    // near 0 (points lie in 1D or 2D). Note we take the sqrt of the
    // eigenvalues since the PCA eigenvalues are homogeneous to m²
    const real_t val0      = std::sqrt(pca.val(0));
    const real_t val1      = std::sqrt(pca.val(1));
    const real_t val2      = std::sqrt(pca.val(2));
    const real_t val0_fact = real_t(1.0) / (val0 + epsilon<real_t>);

    features[EFeatureID::Normal_x]   = pca.v2(0);
    features[EFeatureID::Normal_y]   = pca.v2(1);
    features[EFeatureID::Normal_z]   = pca.v2(2);
    features[EFeatureID::Linearity]  = (val0 - val1) * val0_fact;
    features[EFeatureID::Planarity]  = (val1 - val2) * val0_fact;
    features[EFeatureID::Scattering] = val2 * val0_fact;
    features[EFeatureID::Length]     = val0;
    features[EFeatureID::Surface]    = std::sqrt(val0 * val1 + sq_eps);
    features[EFeatureID::Volume]     = std::pow(val0 * val1 * val2 + cub_eps, one_third);
    features[EFeatureID::Curvature]  = val2 / (val0 + val1 + val2 + epsilon<real_t>);

    // Compute the verticality. NB we account for the edge case
    // where all features are 0
    if (val0 > real_t(0.))
    {
        const Vec3<real_t> unary_vector = {
            pca.val(0) * std::abs(pca.v0(0)) + pca.val(1) * std::abs(pca.v1(0)) + pca.val(2) * std::abs(pca.v2(0)),
            pca.val(0) * std::abs(pca.v0(1)) + pca.val(1) * std::abs(pca.v1(1)) + pca.val(2) * std::abs(pca.v2(1)),
            // pca.v2 is already absolute value (positive). but we keep the operation for now
            // since we can come with our own normal orientation or any other external normal orientation routine
            pca.val(0) * std::abs(pca.v0(2)) + pca.val(1) * std::abs(pca.v1(2)) + pca.val(2) * std::abs(pca.v2(2))};

        features[EFeatureID::VerticalityPGEOF] = unary_vector(2) / unary_vector.norm();
    }
};

/**
 * Given a PCA result compute only a subset of features.
 *
 * This function intends on mimicking the behavior of Jakteristics
 *
 * @param[in] pca PCAResult
 * @param[in] selected_feature a vector of the type of features to compute
 * @param[out] feature_result the array of resulting features. Result are inserted sequentially the order is defined
 * by the selected_feature array.
 */
template <typename real_t>
void compute_selected_features(
    const PCAResult<real_t>& pca, const std::vector<EFeatureID>& selected_feature, real_t* feature_results)
{
    // Compute the dimensionality features. The 1e-3 term is meant
    // to stabilize the division when the cloud's 3rd eigenvalue is
    // near 0 (points lie in 1D or 2D). Note we take the sqrt of the
    // eigenvalues since the PCA eigenvalues are homogeneous to m²
    const real_t val0      = std::sqrt(pca.val(0));
    const real_t val1      = std::sqrt(pca.val(1));
    const real_t val2      = std::sqrt(pca.val(2));
    const real_t val0_fact = real_t(1.0) / (val0 + epsilon<real_t>);

    const auto compute_feature = [val0, val1, val2, val0_fact, &pca](
                                     const EFeatureID feature_id, const size_t output_id, auto* feature_results)
    {
        switch (feature_id)
        {
            case EFeatureID::Normal_x:
                feature_results[output_id] = pca.v2(0);
                break;
            case EFeatureID::Normal_y:
                feature_results[output_id] = pca.v2(1);
                break;
            case EFeatureID::Normal_z:
                feature_results[output_id] = pca.v2(2);
                break;
            case EFeatureID::Linearity:
                feature_results[output_id] = (val0 - val1) * val0_fact;
                break;
            case EFeatureID::Planarity:
                feature_results[output_id] = (val1 - val2) * val0_fact;
                break;
            case EFeatureID::Scattering:
                feature_results[output_id] = val2 * val0_fact;
                break;
            case EFeatureID::Length:
                feature_results[output_id] = val0;
                break;
            case EFeatureID::Surface:
                feature_results[output_id] = std::sqrt(val0 * val1 + 1e-6f);
                break;
            case EFeatureID::Volume:
                feature_results[output_id] = std::pow(
                    val0 * val1 * val2 + real_t(1e-8),
                    real_t(1.) / real_t(3.));  // 1e-9 eps is a too small value for float32 so we fallback to 1e-8
                break;
            case EFeatureID::Curvature:
                feature_results[output_id] = val2 / (val0 + val1 + val2 + epsilon<real_t>);
                break;
            case EFeatureID::VerticalityPGEOF:
                // the verticality as defined in PGEOF
                if (val0 > real_t(0.))
                {
                    const Vec3<real_t> unary_vector = {
                        pca.val(0) * std::abs(pca.v0(0)) + pca.val(1) * std::abs(pca.v1(0)) +
                            pca.val(2) * std::abs(pca.v2(0)),
                        pca.val(0) * std::abs(pca.v0(1)) + pca.val(1) * std::abs(pca.v1(1)) +
                            pca.val(2) * std::abs(pca.v2(1)),
                        // pca.v2 is already absolute value (positive). But we keep the operation for now
                        // since we can come with our own normal orientation or any other external normal orientation
                        // routine
                        pca.val(0) * std::abs(pca.v0(2)) + pca.val(1) * std::abs(pca.v1(2)) +
                            pca.val(2) * std::abs(pca.v2(2))};

                    feature_results[output_id] = unary_vector(2) / unary_vector.norm();
                    // TODO: Jakteristics compute this as feature_results[output_id] = real_t(1.0) -
                    // std::abs(pca.v2(2));
                    // It seems to be the most common formula for the verticality in the literature
                }
                break;
            case EFeatureID::Verticality:
                // The verticality as defined in most of the papers
                // http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf
                feature_results[output_id] = real_t(1.0) - std::abs(pca.v2(2));
            case EFeatureID::Eigentropy:
                feature_results[output_id] = compute_eigentropy(pca);
            default:
                break;
        }
    };

    for (size_t i = 0; i < selected_feature.size(); ++i) { compute_feature(selected_feature[i], i, feature_results); }
}

}  // namespace pgeof
