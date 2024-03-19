#include <cstdint>
#include <cstdio>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "pgeof.hpp"

/* template for handling several index types in compute_geometric_features */
static PyObject* pgeof(
    PyArrayObject *py_xyz, PyArrayObject * py_nn, PyArrayObject * py_nn_ptr,
    int k_min, int k_step, int k_min_search, bool verbose)
{
    //convert from python to arrays
    float * c_xyz = (float*) PyArray_DATA(py_xyz);
    uint32_t * c_nn =  (uint32_t*) PyArray_DATA(py_nn);
    uint32_t * c_nn_ptr= (uint32_t*) PyArray_DATA(py_nn_ptr);
    int n_points = PyArray_DIMS(py_nn_ptr)[0] - 1;
    int num_features = 12;

    //prepare output
    npy_intp size_of_feature[] = {n_points, num_features};
    PyArrayObject* py_features = (PyArrayObject*) PyArray_Zeros(
        2, size_of_feature, PyArray_DescrFromType(NPY_FLOAT32), 0);
    float *features = (float*) PyArray_DATA(py_features);

    compute_geometric_features(
        c_xyz, c_nn, c_nn_ptr, n_points, features, k_min, k_step, k_min_search, verbose);

    return Py_BuildValue("N", py_features);
}


/* actual interface*/
static PyObject* pgeof_cpy(PyObject* self, PyObject *args, PyObject *kwargs)
{   (void) self; // suppress unused parameter warning

    /* inputs  */
    PyArrayObject *xyz;
    PyArrayObject *nn;
    PyArrayObject *nn_ptr;
    int k_min = 1;
    int k_step = -1;
    int k_min_search = 10;
    bool verbose = false;

    // Build variable names used for input args + kwargs parsing
    static char *keywords[] = {
        (char*)"xyz",
        (char*)"nn",
        (char*)"nn_ptr",
        (char*)"k_min",
        (char*)"k_step",
        (char*)"k_min_search",
        (char*)"verbose",
        NULL};

    /* parse the input, from Python Object to C PyArray */
//    if(!PyArg_ParseTuple(args, "OOOiiii", &xyz, &nn, &nn_ptr, &k_min, &k_step, &k_min_search, &verbose)){
//        return NULL;
//    }
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "OOO|iiip",
        keywords,
        &xyz,
        &nn,
        &nn_ptr,
        &k_min,
        &k_step,
        &k_min_search,
        &verbose))
    {
        return NULL;
    }

    PyObject* PyReturn = pgeof(xyz, nn, nn_ptr, k_min, k_step, k_min_search, verbose);
    return PyReturn;
    }

static const char* pgeof_doc =
    "Compute the geometric features associated with each point's\n"
    "neighborhood. The following features are computed:\n"
    "  - linearity\n"
    "  - planarity\n"
    "  - scattering\n"
    "  - verticality\n"
    "  - normal vector (oriented towards positive z-coordinates)\n"
    "  - length\n"
    "  - surface\n"
    "  - volume\n"
    "  - curvature\n"
    "  - optimal neighborhood size\n\n"
    "⚠️ Input neighbors are expected in CSR format. See demo.py for \n"
    "examples of how to build such structure from typical scikit-learn\n"
    " K-NN or radius-NN search outputs.\n\n"
    "Parameters\n"
    "----------\n"
    ":param xyz: [n_points, 3] float32 2D array\n"
    "    3D point coordinates\n"
    ":param nn: [num_neighborhoods] uint32 1D array\n"
    "    Flattened neighbor indices. Make sure those are all positive,\n"
    "    '-1' indices will either crash or silently compute incorrect\n"
    "    features\n"
    ":param nn_ptr: [n_points+1] uint32 1D array\n"
    "    Pointers wrt `nn`. More specifically, the neighbors of point `i`\n"
    "    are `nn[nn_ptr[i]:nn_ptr[i + 1]]`\n"
    ":param k_min: (optional, default=1) int\n"
    "    Minimum number of neighbors to consider for features\n"
    "    computation. If a point has less, it will be given 0 features\n"
    ":param k_step: (optional, default=-1) int\n"
    "    Step size to take when searching for the optimal neighborhood\n"
    "    size for each point, following:\n"
    "    http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf\n"
    "    If k_step < 1, pgeof will not search for the optimal\n"
    "    neighborhood and features will be computed based on the all\n"
    "    available neighbors for each point\n"
    ":param k_min_search: (optional, default=10) int\n"
    "    Minimum neighborhood size at which to start when searching for\n"
    "    the optimal neighborhood size for each point. It is advised to\n"
    "    use a value of 10 or higher, for geometric features robustness\n"
    ":param verbose: (optional, default=False) bool\n"
    "    Whether computation progress should be printed out\n";

static PyMethodDef pgeof_methods[] = {
    {"pgeof", (PyCFunction) pgeof_cpy, METH_VARARGS | METH_KEYWORDS, pgeof_doc},
    {NULL, NULL, 0, NULL}
};

/* module initialization */

static struct PyModuleDef pgeof_module = {
    PyModuleDef_HEAD_INIT,
    "pgeof", /* name of module */
    "Pointwise geometric feature from point cloud", /* module documentation, may be null */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    pgeof_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_pgeof(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */

    PyObject* m;

    /* create the module */
    m = PyModule_Create(&pgeof_module);
    if (!m){ return NULL; }

    return m;
}
