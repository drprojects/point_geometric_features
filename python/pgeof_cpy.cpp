#include <cstdint>
#include <limits>
#include <cstdio>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/pgeof.cpp"


/* template for handling several index types in edge_list_to_forward_star */
static PyObject* pgeof(PyArrayObject *py_xyz, PyArrayObject * py_nn, PyArrayObject * py_nn_ptr,
    int k_min, int verbose)
{
    //convert from python to arrays
    float * xyz = (float*) PyArray_DATA(py_xyz);
    uint32_t * nn =  (uint32_t*) PyArray_DATA(py_nn);
    uint32_t * nn_ptr= (uint32_t*) PyArray_DATA(py_nn_ptr);
    int n_points = PyArray_DIMS(py_xyz)[0];

    //prepare output
    npy_intp size_of_feature[] = {n_points, 11};
    PyArrayObject* py_features = (PyArrayObject*) PyArray_Zeros(2,
        size_of_feature, PyArray_DescrFromType(NPY_FLOAT32), 0);
    float *features = (float*) PyArray_DATA(py_features);

    compute_geometric_features(xyz, nn, nn_ptr, k_min, n_points, features, verbose);

    return Py_BuildValue("O", py_features);
}

/* actual interface*/
static PyObject* pgeof_cpy(PyObject* self, PyObject* args)
{   (void) self; // suppress unused parameter warning

    /* inputs  */
    int k_min, verbose;
    PyArrayObject *py_xyz, *py_nn, *py_nn_ptr;

    /* parse the input, from Python Object to C PyArray */
    if(!PyArg_ParseTuple(args, "OOOii", &py_xyz, &py_nn, &py_nn_ptr, &k_min, &verbose)){
        return NULL;
    }

    PyObject* PyReturn = pgeof(py_xyz, py_nn, py_nn_ptr, k_min, verbose);
        return PyReturn;
    }

static const char* pgeof_doc =
"pgeof doc\n"
"TODO\n";

static PyMethodDef pgeof_methods[] = {
    {"pgeof", pgeof_cpy, METH_VARARGS, pgeof_doc},
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
