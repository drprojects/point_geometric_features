<div align="center">

# Point Geometric Features

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)


</div>


## 📌 Description

Python wrapper around C++ helper to compute, for each point in a 3D point cloud, local geometric features in parallel on CPU:

<details>
<summary><b>️List of computed features️</b></summary>

- linearity
- planarity
- scattering
- verticality
- normal_x
- normal_y
- normal_z
- length
- surface
- volume
- curvature
- optimal neighborhood size
</details>


## 🧱 Installation

Pgeof will __soon__ be available as pre compiled package on PyPI for both Linux and Windows OSes.

```bash
python -m pip install pgeof 
```

### building from sources

Pgeof depends on [Eigen library](https://eigen.tuxfamily.org/) and numpy headers at build time.
The good version of numpy will be fetched from PyPI automatically by the build system but your are responsible for providing
the path to the Eigen library you want to use (for example py using `CXXFLAGS` variable on Linux or setting `EIGEN_LIB_PATH`)

```bash
# clone project
git clone https://github.com/drprojects/point_geometric_features.git
cd point_geometric_features

# set the EIGEN_LIB_PATH if needed
export EIGEN_LIB_PATH="path_to_eigen_root_dir"
# build and install the package
python -m pip install .
```

### conda 

The following will install the project in a new `pgeof` conda environment.

```bash
# clone project
git clone https://github.com/drprojects/point_geometric_features.git
cd point_geometric_features

# Installation in a new dedicated `pgeof` conda environment
bash install.sh
```

You can easily adapt `install.sh` to install the project in an already-existing 
environment.

## 🚀 Using Point Geometric Features

The `pgeof` function should be used as follows:

```python
from pgeof import pgeof

pgeof(
    xyz,              # [n_points, 3] float32 2D array - 3D point coordinates
    nn,               # [num_neighborhoods] uint32 1D array - Flattened neighbor indices. Make sure those are all positive, '-1' indices will either crash or silently compute incorrect features
    nn_ptr,           # [n_points+1] uint32 1D array - Pointers wrt `nn`. More specifically, the neighbors of point `i` are `nn[nn_ptr[i]:nn_ptr[i + 1]]`
    k_min=1,          # (optional, default=1) int - Minimum number of neighbors to consider for features computation. If a point has less, it will be given 0 features
    k_step=-1,        # (optional, default=-1) int - Step size to take when searching for the optimal neighborhood size for each point, following: http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf. If k_step < 1, pgeof will not search for the optimal neighborhood and features will be computed based on the all available neighbors for each point 
    k_min_search=10,  # (optional, default=10) int - Minimum neighborhood size at which to start when searching for the optimal neighborhood size for each point. It is advised to use a value of 10 or higher, for geometric features robustness
    verbose=False)    # (optional, default=False) bool - Whether computation progress should be printed out

# Print details on how pgeof works and expected input parameters
print(help(pgeof))
```

👇 You may check out the provided `demo.py` script to get started.

```bash
python demo.py
```

⚠️ Please note the **neighbors are expected in CSR format**. This allows 
expressing neighborhoods of varying sizes with dense arrays (eg the output of a 
radius search). Here are examples of how to easily convert typical k-NN or 
radius-NN neighborhoods to CSR format.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Generate a random synthetic point cloud and k-nearest neighbors
num_points = 10000
k = 20
xyz = np.random.rand(num_points, 3)
kneigh = NearestNeighbors(n_neighbors=k).fit(xyz).kneighbors(xyz)

# Converting k-nearest neighbors to CSR format
nn_ptr = np.arange(num_points + 1) * k
nn = kneigh[1].flatten()
```

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Generate a random synthetic point cloud and radius neighbors
num_points = 10000
radius = 0.1
xyz = np.random.rand(num_points, 3)
rneigh = NearestNeighbors(radius=radius).fit(xyz).radius_neighbors(xyz)

# Converting radius neighbors to CSR format
nn_ptr = np.r_[0, np.array([x.shape[0] for x in rneigh[1]]).cumsum()]
nn = np.concatenate(rneigh[1])
```


## 💳 Credits
This implementation was largely inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph). The main modifications here allow: 
- parallel computation on all points' local neighborhoods, with neighborhoods of varying sizes
- more geometric features
- optimal neighborhood search from this [paper](http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf)
- some corrections on geometric features computation


## License

Point Geometric Features is licensed under the MIT License. 
