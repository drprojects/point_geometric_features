#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
#include <boost/tuple/tuple_comparison.hpp>
#include <limits>
#include <map>

namespace bp = boost::python;
namespace ei = Eigen;
namespace bpn = boost::python::numpy;

typedef ei::Matrix<float, 3, 3> Matrix3f;
typedef ei::Matrix<float, 3, 1> Vector3f;

typedef boost::tuple< std::vector< std::vector<float> >, std::vector< std::vector<uint8_t> >, std::vector< std::vector<uint32_t> >, std::vector<std::vector<uint32_t> > > Custom_tuple;
typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t> > Components_tuple;
typedef boost::tuple< std::vector<uint8_t>, std::vector<uint8_t> > Subgraph_tuple;

typedef boost::tuple< uint32_t, uint32_t, uint32_t > Space_tuple;

struct VecToArray
{//converts a vector<uint8_t> to a numpy array
    static PyObject * convert(const std::vector<uint8_t> & vec) {
    npy_intp dims = vec.size();
    PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT8);
    void * arr_data = PyArray_DATA((PyArrayObject*)obj);
    memcpy(arr_data, &vec[0], dims * sizeof(uint8_t));
    return obj;
    }
};

template <class T>
struct VecvecToArray
{//converts a vector< vector<uint32_t> > to a numpy 2d array
    static PyObject * convert(const std::vector< std::vector<T> > & vecvec)
    {
        npy_intp dims[2];
        dims[0] = vecvec.size();
        dims[1] = vecvec[0].size();
        PyObject * obj;
        if (typeid(T) == typeid(uint8_t))
            obj = PyArray_SimpleNew(2, dims, NPY_UINT8);
        else if (typeid(T) == typeid(float))
            obj = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        else if (typeid(T) == typeid(uint32_t))
            obj = PyArray_SimpleNew(2, dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        std::size_t cell_size = sizeof(T);
        for (std::size_t i = 0; i < dims[0]; i++)
        {
            memcpy(arr_data + i * dims[1] * cell_size, &(vecvec[i][0]), dims[1] * cell_size);
        }
        return obj;
    }
};

struct VecToArray32
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};


template<class T>
struct VecvecToList
{//converts a vector< vector<T> > to a list
        static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};

struct to_py_tuple
{//converts to a python tuple
    static PyObject* convert(const Custom_tuple & c_tuple){
        bp::list values;

        PyObject * pyo1 = VecvecToArray<float>::convert(c_tuple.get<0>());
        PyObject * pyo2 = VecvecToArray<uint8_t>::convert(c_tuple.get<1>());
        PyObject * pyo3 = VecvecToArray<uint32_t>::convert(c_tuple.get<2>());
        PyObject * pyo4 = VecvecToArray<uint32_t>::convert(c_tuple.get<3>());

        values.append(bp::handle<>(bp::borrowed(pyo1)));
        values.append(bp::handle<>(bp::borrowed(pyo2)));
        values.append(bp::handle<>(bp::borrowed(pyo3)));
        values.append(bp::handle<>(bp::borrowed(pyo4)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

struct to_py_tuple_components
{//converts output to a python tuple
    static PyObject* convert(const Components_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<uint32_t>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray32::convert(c_tuple.get<1>());

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

struct to_py_tuple_subgraph
{//converts output to a python tuple
    static PyObject* convert(const Subgraph_tuple& s_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vec_pyo1 = VecToArray::convert(s_tuple.get<0>());
        PyObject * vec_pyo2 = VecToArray::convert(s_tuple.get<1>());

        values.append(bp::handle<>(bp::borrowed(vec_pyo1)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo2)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

class AttributeGrid {
//voxelization of the space, allows to accumulate the position, the color and labels
    std::map<Space_tuple, uint64_t> space_tuple_to_index;//associate eeach non-empty voxel to an index
    uint64_t index;
    std::vector<uint32_t> bin_count;//count the number of point in each non-empty voxel
    std::vector< std::vector<float> > acc_xyz;//accumulate the position of the points
    std::vector< std::vector<uint32_t> > acc_rgb;//accumulate the color of the points
    std::vector< std::vector<uint32_t> > acc_labels;//accumulate the label of the points
    std::vector< std::vector<uint32_t> > acc_objects;//accumulate the object indices of the points
  public:
    AttributeGrid():
       index(0)
    {}
    //---methods for the occurence grid---
    uint64_t n_nonempty_voxels()
    {
        return this->index;
    }
    uint32_t get_index(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin)
    {
        return space_tuple_to_index.at(Space_tuple(x_bin, y_bin, z_bin));
    }
    bool add_occurence(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin)
    {
        Space_tuple st(x_bin, y_bin, z_bin);
        auto inserted = space_tuple_to_index.insert(std::pair<Space_tuple, uint64_t>(st, index));
        if (inserted.second)
        {
            this->index++;
            return true;
        }
        else
        {
            return false;
        }
    }
    std::map<Space_tuple,uint64_t>::iterator begin()
    {
        return this->space_tuple_to_index.begin();
    }
    std::map<Space_tuple,uint64_t>::iterator end()
    {
        return this->space_tuple_to_index.end();
    }
    //---methods for accumulating atributes---
    void initialize(uint8_t n_labels, int n_objects)
    {//must be run once space_tuple_to_index is complete and the number of non-empty voxels is known
        bin_count  = std::vector<uint32_t>(this->index, 0);
        acc_xyz    = std::vector< std::vector<float> >(this->index, std::vector <float>(3,0));
        acc_rgb    = std::vector< std::vector<uint32_t> >(this->index, std::vector <uint32_t>(3,0));
        acc_labels = std::vector< std::vector<uint32_t> >(this->index, std::vector <uint32_t>(n_labels+1,0));
        acc_objects = std::vector< std::vector<uint32_t> >(this->index, std::vector <uint32_t>(n_objects+1,0));
    }
    uint32_t get_count(uint64_t voxel_index)
    {
        return bin_count.at(voxel_index);
    }
    std::vector<float> get_pos(uint64_t voxel_index)
    {
        return acc_xyz.at(voxel_index);
    }
    std::vector<uint32_t> get_rgb(uint64_t voxel_index)
    {
        return acc_rgb.at(voxel_index);
    }
    std::vector<uint32_t> get_acc_labels(uint64_t voxel_index)
    {
        return acc_labels.at(voxel_index);
    }
    std::vector<uint32_t> get_acc_objects(uint64_t voxel_index)
    {
        return acc_objects.at(voxel_index);
    }
    uint8_t get_label(uint64_t voxel_index)
    {//return the majority label from this voxel
     //ignore the unlabeled points (0), unless all points are unlabeled
        std::vector<uint32_t> label_hist = acc_labels.at(voxel_index);
        std::vector<uint32_t>::iterator chosen_label = std::max_element(label_hist.begin() + 1, label_hist.end());
        if (*chosen_label == 0)
        {
            return 0;
        }
        else
        {
            return (uint8_t)std::distance(label_hist.begin(), chosen_label);
        }
    }
    uint32_t get_object(uint64_t voxel_index)
    {//return the majority object from this voxel
     //ignore the unattributed points (0), unless all points are unattributed
        std::vector<uint32_t> object_hist = acc_objects.at(voxel_index);
        std::vector<uint32_t>::iterator chosen_object = std::max_element(object_hist.begin() + 1, object_hist.end());
        if (*chosen_object == 0)
        {
            return 0;
        }
        else
        {
            return (uint32_t)std::distance(object_hist.begin(), chosen_object);
        }
    }
    void add_attribute(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b)
    {//add a point x y z in voxel x_bin y_bin z_bin
        uint64_t bin = get_index(x_bin, y_bin, z_bin);
        bin_count.at(bin) = bin_count.at(bin) + 1;
        acc_xyz.at(bin).at(0) = acc_xyz.at(bin).at(0) + x;
        acc_xyz.at(bin).at(1) = acc_xyz.at(bin).at(1) + y;
        acc_xyz.at(bin).at(2) = acc_xyz.at(bin).at(2) + z;
        acc_rgb.at(bin).at(0) = acc_rgb.at(bin).at(0) + r;
        acc_rgb.at(bin).at(1) = acc_rgb.at(bin).at(1) + g;
        acc_rgb.at(bin).at(2) = acc_rgb.at(bin).at(2) + b;
    }
    void add_attribute(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint8_t label)
    {//add a point x y z in voxel x_bin y_bin z_bin - with label
        uint32_t bin =get_index(x_bin, y_bin, z_bin);
        bin_count.at(bin) = bin_count.at(bin) + 1;
        acc_xyz.at(bin).at(0) = acc_xyz.at(bin).at(0) + x;
        acc_xyz.at(bin).at(1) = acc_xyz.at(bin).at(1) + y;
        acc_xyz.at(bin).at(2) = acc_xyz.at(bin).at(2) + z;
        acc_rgb.at(bin).at(0) = acc_rgb.at(bin).at(0) + r;
        acc_rgb.at(bin).at(1) = acc_rgb.at(bin).at(1) + g;
        acc_rgb.at(bin).at(2) = acc_rgb.at(bin).at(2) + b;
        acc_labels.at(bin).at(label) = acc_labels.at(bin).at(label) + 1;
    }
    void add_attribute(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint8_t label, uint32_t object)
    {//add a point x y z in voxel x_bin y_bin z_bin - with label
        uint32_t bin =get_index(x_bin, y_bin, z_bin);
        bin_count.at(bin) = bin_count.at(bin) + 1;
        acc_xyz.at(bin).at(0) = acc_xyz.at(bin).at(0) + x;
        acc_xyz.at(bin).at(1) = acc_xyz.at(bin).at(1) + y;
        acc_xyz.at(bin).at(2) = acc_xyz.at(bin).at(2) + z;
        acc_rgb.at(bin).at(0) = acc_rgb.at(bin).at(0) + r;
        acc_rgb.at(bin).at(1) = acc_rgb.at(bin).at(1) + g;
        acc_rgb.at(bin).at(2) = acc_rgb.at(bin).at(2) + b;
        acc_labels.at(bin).at(label) = acc_labels.at(bin).at(label) + 1;
        acc_objects.at(bin).at(object) = acc_objects.at(bin).at(object) + 1;
    }
};

PyObject *  prune(const bpn::ndarray & xyz ,float voxel_size, const bpn::ndarray & rgb, const bpn::ndarray & labels, const bpn::ndarray & objects, const int n_labels, const int n_objects)
{//prune the point cloud xyz with a regular voxel grid
//    std::cout << "=========================" << std::endl;
//    std::cout << "======== pruning ========" << std::endl;
//    std::cout << "=========================" << std::endl;
    uint64_t n_ver = bp::len(xyz);
    bool have_labels = n_labels>0;
    bool have_objects = n_objects>0;
    //---read the numpy arrays data---
    const float * xyz_data = reinterpret_cast<float*>(xyz.get_data());
    const uint8_t * rgb_data = reinterpret_cast<uint8_t*>(rgb.get_data());
    const uint8_t * label_data;
    if (have_labels)
        label_data = reinterpret_cast<uint8_t*>(labels.get_data());
    const uint32_t * object_data;
    if (have_objects)
        object_data = reinterpret_cast<uint32_t*>(objects.get_data());
    //---find min max of xyz----
    float x_max = std::numeric_limits<float>::lowest(), x_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest(), y_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest(), z_min = std::numeric_limits<float>::max();

    #pragma omp parallel for reduction(max : x_max, y_max, z_max), reduction(min : x_min, y_min, z_min)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        if (x_max < xyz_data[3 * i_ver]){           x_max = xyz_data[3 * i_ver];}
        if (y_max < xyz_data[3 * i_ver + 1]){       y_max = xyz_data[3 * i_ver + 1];}
        if (z_max < xyz_data[3 * i_ver + 2]){       z_max = xyz_data[3 * i_ver + 2];}
        if (x_min > xyz_data[3 * i_ver]){           x_min = xyz_data[3 * i_ver];}
        if (y_min > xyz_data[3 * i_ver + 1]){       y_min = xyz_data[3 * i_ver + 1];}
        if (z_min > xyz_data[3 * i_ver + 2 ]){      z_min = xyz_data[3 * i_ver + 2];}
    }
    //---compute the voxel grid size---
    uint32_t n_bin_x = std::ceil((x_max - x_min) / voxel_size);
    uint32_t n_bin_y = std::ceil((y_max - y_min) / voxel_size);
    uint32_t n_bin_z = std::ceil((z_max - z_min) / voxel_size);
//    std::cout << "Voxelization into " << n_bin_x << " x " << n_bin_y << " x " << n_bin_z << " grid" << std::endl;
    //---detect non-empty voxels----
    AttributeGrid vox_grid;
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        uint32_t bin_x = std::floor((xyz_data[3 * i_ver] - x_min) / voxel_size);
        uint32_t bin_y = std::floor((xyz_data[3 * i_ver + 1] - y_min) / voxel_size);
        uint32_t bin_z = std::floor((xyz_data[3 * i_ver + 2] - z_min) / voxel_size);
        vox_grid.add_occurence(bin_x, bin_y, bin_z);
    }
//    std::cout << "Reduced from " << n_ver << " to " << vox_grid.n_nonempty_voxels() << " points ("
//              << std::ceil(10000 * vox_grid.n_nonempty_voxels() / n_ver)/100 << "%)" << std::endl;
    vox_grid.initialize(n_labels, n_objects);
    //---accumulate points in the voxel map----
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        uint32_t bin_x = std::floor((xyz_data[3 * i_ver    ] - x_min) / voxel_size);
        uint32_t bin_y = std::floor((xyz_data[3 * i_ver + 1] - y_min) / voxel_size);
        uint32_t bin_z = std::floor((xyz_data[3 * i_ver + 2] - z_min) / voxel_size);
        if (have_labels&&!have_objects)
            vox_grid.add_attribute(bin_x, bin_y, bin_z
                    , xyz_data[3 * i_ver], xyz_data[3 * i_ver + 1], xyz_data[3 * i_ver + 2]
                    , rgb_data[3 * i_ver], rgb_data[3 * i_ver + 1], rgb_data[3 * i_ver + 2], label_data[i_ver]);
        else if(have_labels&&have_objects)
            vox_grid.add_attribute(bin_x, bin_y, bin_z
                    , xyz_data[3 * i_ver], xyz_data[3 * i_ver + 1], xyz_data[3 * i_ver + 2]
                    , rgb_data[3 * i_ver], rgb_data[3 * i_ver + 1], rgb_data[3 * i_ver + 2], label_data[i_ver], object_data[i_ver]);
        else
            vox_grid.add_attribute(bin_x, bin_y, bin_z
                    , xyz_data[3 * i_ver], xyz_data[3 * i_ver + 1], xyz_data[3 * i_ver + 2]
                    , rgb_data[3 * i_ver], rgb_data[3 * i_ver + 1], rgb_data[3 * i_ver + 2]);
    }
    //---compute pruned cloud----
    std::vector< std::vector< float > > pruned_xyz(vox_grid.n_nonempty_voxels(), std::vector< float >(3, 0.f));
    std::vector< std::vector< uint8_t > > pruned_rgb(vox_grid.n_nonempty_voxels(), std::vector< uint8_t >(3, 0));
    std::vector< std::vector< uint32_t > > pruned_labels(vox_grid.n_nonempty_voxels(), std::vector< uint32_t >(n_labels + 1, 0));
    std::vector< std::vector< uint32_t > > pruned_objects(vox_grid.n_nonempty_voxels(), std::vector< uint32_t >(n_objects + 1, 0));
    for (std::map<Space_tuple,uint64_t>::iterator it_vox=vox_grid.begin(); it_vox!=vox_grid.end(); ++it_vox)
    {//loop over the non-empty voxels and compute the average posiition/color + majority label
        uint64_t voxel_index = it_vox->second; //
        float count = (float)vox_grid.get_count(voxel_index);
        std::vector<float> pos = vox_grid.get_pos(voxel_index);
        pos.at(0) = pos.at(0) / count;
        pos.at(1) = pos.at(1) / count;
        pos.at(2) = pos.at(2) / count;
        pruned_xyz.at(voxel_index) = pos;
        std::vector<uint32_t> col = vox_grid.get_rgb(voxel_index);
        std::vector<uint8_t> col_uint8_t(3);
        col_uint8_t.at(0) = (uint8_t)((float) col.at(0) / count);
        col_uint8_t.at(1) = (uint8_t)((float) col.at(1) / count);
        col_uint8_t.at(2) = (uint8_t)((float) col.at(2) / count);
        pruned_rgb.at(voxel_index) = col_uint8_t;
        pruned_labels.at(voxel_index) = vox_grid.get_acc_labels(voxel_index);
        pruned_objects.at(voxel_index) = vox_grid.get_acc_objects(voxel_index);
    }
    return to_py_tuple::convert(Custom_tuple(pruned_xyz,pruned_rgb, pruned_labels, pruned_objects));
}



PyObject * compute_geometric_features(
    const bpn::ndarray & xyz_boost, const bpn::ndarray & nn_boost,
    const bpn::ndarray & nn_ptr_boost, int k_min, bool verbose)
{
    /*
    Compute the geometric features associated with each point's
    neighborhood. The following features are computed:
     - linearity
     - planarity
     - scattering
     - verticality
     - normal vector
     - length
     - surface
     - volume

    Parameters
    ----------
    xyz_boost : bpn::ndarray
        Array of size (n_points, 3) holding the XYZ coordinates for N
        points
    nn_boost : bpn::ndarray
        Array of size (n_neighbors) holding the points' neighbor indices
        flattened for CSR format
    nn_ptr_boost : bpn::ndarray
        Array of size (n_points + 1) indicating the start and end
        indices of each point's neighbors in nn_boost
    k_min: int
        Minimum number of neighbors to consider for features
        computation. If less, the point set will be given 0 features
    verbose: bool
        Whether computation progress should be printed out
    */

    // Initialize the features
    std::size_t n_points = bp::len(nn_ptr_boost) - 1;
    std::vector<std::vector<float>> features(n_points, std::vector<float>(11, 0));

    // Read numpy array data
    const float * xyz = reinterpret_cast<float*>(xyz_boost.get_data());
    const uint32_t * nn = reinterpret_cast<uint32_t*>(nn_boost.get_data());
    const uint32_t * nn_ptr = reinterpret_cast<uint32_t*>(nn_ptr_boost.get_data());

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
            features[i_point][0]  = 0;
            features[i_point][1]  = 0;
            features[i_point][2]  = 0;
            features[i_point][3]  = 0;
            features[i_point][4]  = 0;
            features[i_point][5]  = 0;
            features[i_point][6]  = 0;
            features[i_point][7]  = 0;
            features[i_point][8]  = 0;
            features[i_point][9]  = 0;
            features[i_point][10] = 0;
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
        features[i_point][0]  = linearity;
        features[i_point][1]  = planarity;
        features[i_point][2]  = scattering;
        features[i_point][3]  = verticality;
        features[i_point][4]  = v2[0];
        features[i_point][5]  = v2[1];
        features[i_point][6]  = v2[2];
        features[i_point][7]  = length;
        features[i_point][8]  = surface;
        features[i_point][9]  = volume;
        features[i_point][10] = curvature;

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

    return VecvecToArray<float>::convert(features);
}


using namespace boost::python;
BOOST_PYTHON_MODULE(libpgeof)
{
    _import_array();
    bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
    bp::to_python_converter< Custom_tuple, to_py_tuple>();
    Py_Initialize();
    bpn::initialize();
    def("compute_geometric_features", compute_geometric_features);
    def("prune", prune);
}
