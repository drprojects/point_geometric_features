<div align="center">

# Point Geometric Features

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](#)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#)


</div>


## 📌 Description

The `pgeof` library provides utilities for fast, parallelized computing ⚡ of **local geometric 
features for 3D point clouds** ☁️ **on CPU** .

<details>
<summary><b>️List of available features ️👇</b></summary>

- linearity
- planarity
- scattering
- verticality (two formulations)
- normal_x
- normal_y
- normal_z
- length
- surface
- volume
- curvature
- optimal neighborhood size
</details>

`pgeof` allows computing features in multiple fashions: **on-the-fly subset of features** 
_a la_ [jakteristics](https://jakteristics.readthedocs.io), **array of features**, or 
**multiscale features**. Moreover, `pgeof` also offers functions for fast **K-NN** or 
**radius-NN** searches 🔍. 

Behind the scenes, the library is a Python wrapper around C++ utilities.
The overall code is not intended to be DRY nor generic, it aims at providing efficient as 
possible implementations for some limited scopes and usages.

## 🧱 Installation

### From binaries

```bash
python -m pip install pgeof 
```

or 

```bash
python -m pip install git+https://github.com/drprojects/point_geometric_features
```

### Building from sources

`pgeof` depends on [Eigen library](https://eigen.tuxfamily.org/), [Taskflow](https://github.com/taskflow/taskflow), [nanoflann](https://github.com/jlblancoc/nanoflann) and [nanobind](https://github.com/wjakob/nanobind).
The library adheres to [PEP 517](https://peps.python.org/pep-0517/) and uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) as build backend. 
Build dependencies (`nanobind`, `scikit-build-core`, ...) are fetched at build time.
C++ third party libraries are embedded as submodules.


```bash
# Clone project
git clone --recurse-submodules https://github.com/drprojects/point_geometric_features.git
cd point_geometric_features

# Build and install the package
python -m pip install .
```

## 🚀 Using Point Geometric Features

Here we summarize the very basics of `pgeof` usage. 
Users are invited to use `help(pgeof)` for further details on parameters.

At its core `pgeof` provides three functions to compute a set of features given a 3D point cloud and
some precomputed neighborhoods.

```python
import pgeof

# Compute a set of 11 predefined features per points
pgeof.compute_features(
    xyz, # The point cloud. A numpy array of shape (n, 3)
    nn, # CSR data structure see below
    nn_ptr, # CSR data structure see below
    k_min = 1 # Minimum number of neighbors to consider for features computation
    verbose = false # Basic verbose output, for debug purposes
)
```

```python
# Sequence of n scales feature computation
pgeof.compute_features_multiscale(
    ...
    k_scale # array of neighborhood size
)
```

```python
# Feature computation with optimal neighborhood selection as exposed in Weinmann et al., 2015
# return a set of 12 features per points (11 + the optimal neighborhood size)
pgeof.compute_features_optimal(
    ...
    k_min = 1, # Minimum number of neighbors to consider for features computation
    k_step = 1, # Step size to take when searching for the optimal neighborhood
    k_min_search = 1, # Starting size for searching the optimal neighborhood size. Should be >= k_min 
)
```

⚠️ Please note that for theses three functions the **neighbors are expected in CSR format**. 
This allows expressing neighborhoods of varying sizes with dense arrays (e.g. the output of a 
radius search).

We provide very tiny and specialized **k-NN** and **radius-NN** search routines. 
They rely on `nanoflann` C++ library and should be **faster and lighter than `scipy` and 
`sklearn` alternatives**.

Here are some examples of how to easily compute and convert typical k-NN or radius-NN neighborhoods to CSR format (`nn` and `nn_ptr` are two flat `uint32` arrays):

```python
import pgeof
import numpy as np

# Generate a random synthetic point cloud and k-nearest neighbors
num_points = 10000
k = 20
xyz = np.random.rand(num_points, 3).astype("float32")
knn, _ = pgeof.knn_search(xyz, xyz, k)

# Converting k-nearest neighbors to CSR format
nn_ptr = np.arange(num_points + 1) * k
nn = knn.flatten()

# You may need to convert nn/nn_ptr to uint32 arrays
nn_ptr = nn_ptr.astype("uint32")
nn = nn.astype("uint32")

features = pgeof.compute_features(xyz, nn, nn_ptr)
```

```python
import pgeof
import numpy as np

# Generate a random synthetic point cloud and k-nearest neighbors
num_points = 10000
radius = 0.2
k = 20
xyz = np.random.rand(num_points, 3).astype("float32")
knn, _ = pgeof.radius_search(xyz, xyz, radius, k)

# Converting radius neighbors to CSR format
nn_ptr = np.r_[0, (knn >= 0).sum(axis=1).cumsum()]
nn = knn[knn >= 0]

# You may need to convert nn/nn_ptr to uint32 arrays
nn_ptr = nn_ptr.astype("uint32")
nn = nn.astype("uint32")

features = pgeof.compute_features(xyz, nn, nn_ptr)
```

At last, and as a by-product, we also provide a function to **compute a subset of features on the fly**. 
It is inspired by the [jakteristics](https://jakteristics.readthedocs.io) python package (while 
being less complete but faster).
The list of features to compute is given as an array of `EFeatureID`.

```python
import pgeof
from pgeof import EFeatureID
import numpy as np

# Generate a random synthetic point cloud and k-nearest neighbors
num_points = 10000
radius = 0.2
k = 20
xyz = np.random.rand(num_points, 3)

# Compute verticality and curvature
features = pgeof.compute_features_selected(xyz, radius, k, [EFeatureID.Verticality, EFeatureID.Curvature])
```

## Known limitations

Some functions only accept `float` scalar types and `uint32` index types, and we avoid implicit
cast / conversions.
This could be a limitation in some situations (e.g. point clouds with `double` coordinates or 
involving very large big integer indices).
Some C++ functions could be templated / to accept other types without conversion.
For now, this feature is not enabled everywhere, to reduce compilation time and enhance code 
readability. 
Please let us know if you need this feature !

By convention, our normal vectors are forced to be oriented towards positive Z values. 
We make this design choice in order to return consistently-oriented normals. 

## Testing

Some basic tests and benchmarks are provided in the `tests` directory.
Tests can be run in a clean and reproducible environments via `tox` (`tox run` and
`tox run -e bench`).

## 💳 Credits
This implementation was largely inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph). The main modifications here allow: 
- parallel computation on all points' local neighborhoods, with neighborhoods of varying sizes
- more geometric features
- optimal neighborhood search from this [paper](http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf)
- some corrections on geometric features computation

Some heavy refactoring (port to nanobind, test, benchmarks), packaging, speed optimization, feature addition (NN search, on the fly feature computation...) were funded by:

Centre of Wildfire Research of Swansea University (UK) in collaboration with the Research Institute of Biodiversity (CSIC, Spain) and the Department of Mining Exploitation of the University of Oviedo (Spain).

Funding provided by the UK NERC project (NE/T001194/1):

'Advancing 3D Fuel Mapping for Wildfire Behaviour and Risk Mitigation Modelling'

and by the Spanish Knowledge Generation project (PID2021-126790NB-I00):

‘Advancing carbon emission estimations from wildfires applying artificial intelligence to 3D terrestrial point clouds’.

## License

Point Geometric Features is licensed under the MIT License. 
