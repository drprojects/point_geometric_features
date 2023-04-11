#include <iostream>
#include <cstdio>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <numpy/ndarrayobject.h>
#include <limits>
#include <map>

namespace ei = Eigen;

typedef ei::Matrix<float, 3, 3> Matrix3f;
typedef ei::Matrix<float, 3, 1> Vector3f;


void compute_geometric_features(
    float *xyz, uint32_t *nn, const uint32_t *nn_ptr, int k_min, int n_points, float *features, bool verbose)
{

    // Each point can be treated in parallel
    std::size_t s_point = 0;
    #pragma omp parallel for schedule(static)
    for (std::size_t i_point = 0; i_point < n_points; i_point++)
    {
        // Compute the cloud (n_neighbors + 1, 3) matrix holding the
        // points' neighbors XYZ coordinates
        std::size_t k_nn = nn_ptr[i_point + 1] - nn_ptr[i_point];
        ei::MatrixXf cloud(k_nn, 3);

        // If the cloud has only one point, populate the final feature
        // vector with zeros and continue
        if (k_nn < k_min or k_nn <= 0)
        {
            std::cout << i_point <<" "<< k_nn<< std::endl;
            features[i_point * 11 + 0]  = 0;
            features[i_point * 11 + 1]  = 0;
            features[i_point * 11 + 2]  = 0;
            features[i_point * 11 + 3]  = 0;
            features[i_point * 11 + 4]  = 0;
            features[i_point * 11 + 5]  = 0;
            features[i_point * 11 + 6]  = 0;
            features[i_point * 11 + 7]  = 0;
            features[i_point * 11 + 8]  = 0;
            features[i_point * 11 + 9]  = 0;
            features[i_point * 11 + 10] = 0;
            continue;
        }

        // Recover the neighbors' XYZ coordinates using nn and xyz
        std::size_t idx_nei;
        for (std::size_t i_nei = 0; i_nei < k_nn; i_nei++)
        {
            // Recover the neighbor's position in the xyz vector
            idx_nei = nn[nn_ptr[i_point] + i_nei];

            // Recover the corresponding xyz coordinates
            cloud(i_nei, 0) = xyz[3 * idx_nei];
            cloud(i_nei, 1) = xyz[3 * idx_nei + 1];
            cloud(i_nei, 2) = xyz[3 * idx_nei + 2];
        }

        // Compute the (3, 3) covariance matrix
        ei::MatrixXf centered_cloud = cloud.rowwise() - cloud.colwise().mean();
        ei::Matrix3f cov =
            (centered_cloud.adjoint() * centered_cloud) / float(k_nn);

        // Compute the eigenvalues and eigenvectors of the covariance
        ei::EigenSolver<Matrix3f> es(cov);

        // Sort the values and vectors in order of increasing eigenvalue
        std::vector<float> ev = {
            es.eigenvalues()[0].real(),
            es.eigenvalues()[1].real(),
            es.eigenvalues()[2].real()};
        std::vector<int> indices(3);
        std::size_t n(0);
        std::generate(
            std::begin(indices),
            std::end(indices),
            [&]{ return n++; });
        std::sort(
            std::begin(indices),
            std::end(indices),
            [&](int i1, int i2) { return ev[i1] > ev[i2]; } );
        std::vector<float> val = {
            (std::max(ev[indices[0]],0.f)),
            (std::max(ev[indices[1]],0.f)),
            (std::max(ev[indices[2]],0.f))};
        std::vector<float> v0 = {
            es.eigenvectors().col(indices[0])(0).real(),
            es.eigenvectors().col(indices[0])(1).real(),
            es.eigenvectors().col(indices[0])(2).real()};
        std::vector<float> v1 = {
            es.eigenvectors().col(indices[1])(0).real(),
            es.eigenvectors().col(indices[1])(1).real(),
            es.eigenvectors().col(indices[1])(2).real()};
        std::vector<float> v2 = {
            es.eigenvectors().col(indices[2])(0).real(),
            es.eigenvectors().col(indices[2])(1).real(),
            es.eigenvectors().col(indices[2])(2).real()};

        // To standardize the orientation of eigenvectors, we choose to
        // enforce all eigenvectors to be expressed in the Z+ half-space
        if (v0[2] < 0)
        {
            v0[0] = -v0[0];
            v0[1] = -v0[1];
            v0[2] = -v0[2];
        }

        if (v1[2] < 0)
        {
            v1[0] = -v1[0];
            v1[1] = -v1[1];
            v1[2] = -v1[2];
        }

        if (v2[2] < 0)
        {
            v2[0] = -v2[0];
            v2[1] = -v2[1];
            v2[2] = -v2[2];
        }

        // Compute the dimensionality features. The 1e-3 term is meant
        // to stabilize the division when the cloud's 3rd eigenvalue is
        // near 0 (points lie in 1D or 2D). Note we take the sqrt of the
        // eigenvalues since the PCA eigenvaluess are homogeneous to m²
        float val0       = sqrtf(val[0]);
        float val1       = sqrtf(val[1]);
        float val2       = sqrtf(val[2]);
        float linearity  = (val0 - val1) / (val0 + 1e-3);
        float planarity  = (val1 - val2) / (val0 + 1e-3);
        float scattering = val2 / (val0 + 1e-3);
        float length     = val0;
        float surface    = sqrtf(val0 * val1 + 1e-6);
        float volume     = powf(val0 * val1 * val2 + 1e-9, 1 / 3.);
        float curvature  = val2 / (val0 + val1 + val2 + 1e-3);

        // Compute the verticality. NB we account for the edge case
        // where all features are 0
        float verticality = 0;
        if (val0 > 0)
        {
            std::vector<float> unary_vector = {
                val[0] * fabsf(v0[0]) + val[1] * fabsf(v1[0]) + val[2] * fabsf(v2[0]),
                val[0] * fabsf(v0[1]) + val[1] * fabsf(v1[1]) + val[2] * fabsf(v2[1]),
                val[0] * fabsf(v0[2]) + val[1] * fabsf(v1[2]) + val[2] * fabsf(v2[2])};
            float norm = sqrt(
                unary_vector[0] * unary_vector[0]
                + unary_vector[1] * unary_vector[1]
                + unary_vector[2] * unary_vector[2]);
            verticality = unary_vector[2] / norm;
        }

        // Populate the final feature vector
        features[i_point * 11 + 0]  = linearity;
        features[i_point * 11 + 1]  = planarity;
        features[i_point * 11 + 2]  = scattering;
        features[i_point * 11 + 3]  = verticality;
        features[i_point * 11 + 4]  = v2[0];
        features[i_point * 11 + 5]  = v2[1];
        features[i_point * 11 + 6]  = v2[2];
        features[i_point * 11 + 7]  = length;
        features[i_point * 11 + 8]  = surface;
        features[i_point * 11 + 9]  = volume;
        features[i_point * 11 + 10] = curvature;

        // Print progress
        // NB: when in parallel s_point behavior is undefined, but gives
        // a good indication of progress
        s_point++;
        if (s_point % 10000 == 0 && verbose)
        {
            std::cout << s_point << "% done          \r" << std::flush;
            std::cout << ceil(s_point * 100 / n_points) << "% done          \r" << std::flush;
        }
    }

    // Final print to start on a new line
    if (verbose)
    {
        std::cout << std::endl;
    }

    return;
}
