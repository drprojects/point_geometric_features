<div align="center">

# Point Geometric Features

[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
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

The `pgeof` function of the `pgeof` module should be used as follows:

```python
import sys
import os.path as osp
sys.path.append(osp.join(osp.realpath(osp.dirname(__file__)), "python/bin"))
from pgeof import pgeof

pgeof(
    xyz,              # ndarray - Array of size (n_points, 3) holding the XYZ coordinates for N points
    nn,               # ndarray - Array of size (n_neighbors) holding the points' neighbor indices flattened for CSR format
    nn_ptr,           # ndarray - Array of size (n_points + 1) indicating the start and end indices of each point's neighbors in nn
    k_min=1,          # int - Minimum number of neighbors to consider for features computation. If less, the point set will be given 0 features
    k_step=-1,        # int - Step size to take when searching for the optimal neighborhood size, following: http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf. If k_step < 1, the optimal neighborhood will be computed based on all the neighbors available for each point 
    k_min_search=10,  # int - Minimum neighborhood size used when searching the optimal neighborhood size. It is advised to use a value of 10 or higher
    verbose=False)    # bool - Whether computation progress should be printed out
```

You may check out the provided demonstration script to get started 👇

```bash
python demo.py
```

⚠️ Please note the **neighbors are expected in CSR format**. This allows 
expressing neighborhoods of varying sizes with dense arrays (eg the output of a 
radius search).


## 💳 Credits
This implementation was largely inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph). The main modifications here allow: 
- parallel computation on all points' local neighborhoods, with neighborhoods of varying sizes
- more geometric features
- optimal neighborhood search from this [paper](http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf)
- some corrections on geometric features computation


## License

Point Geometric Features is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```