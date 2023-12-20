<div align="center">

# Point Geometric Features

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](#)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#)


</div>


## 📌 Description

Python wrapper around C++ helper to compute, for each point in a 3D point cloud, local geometric features in parallel on CPU:

<details>
<summary><b>️List of computed features️</b></summary>

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

The wrapper allows to compute feature in multiple fashions (on the fly subset of features _a la_ jakteritics or an array of features, multiscale features). Moreover, it offers basic interfaces to compute fast K-NN or Radius search on point clouds. 
The overall code is not intended to be DRY nor generic, it aims at providing efficient as possible implementations for some limited scopes and usages.

## 🧱 Installation

```bash
python -m pip install pgeof2 
```

### building from sources

Pgeof depends on [Eigen library](https://eigen.tuxfamily.org/), [Taskflow](https://github.com/taskflow/taskflow), 
[nanoflann](https://github.com/jlblancoc/nanoflann) and [nanobind](https://github.com/wjakob/nanobind).


Pgeof adhere to [PEP 517](https://peps.python.org/pep-0517/) and use [scikit-build-core](https://github.com/scikit-build/scikit-build-core) as build backend. Build dependencies (nanobind, scikit-build-core...) are fetched at build time. C++ third party libraries are embedded as submodules.


```bash
# clone project
git clone --recurse-submodules https://github.com/drprojects/point_geometric_features.git
cd point_geometric_features
# build and install the package
python -m pip install .
```

## 🚀 Using Point Geometric Features

👇 You may check out the provided `test_pgeof.py` script to get started.

⚠️ Please note the **neighbors are expected in CSR format**. This allows 
expressing neighborhoods of varying sizes with dense arrays (eg the output of a 
radius search). Here are examples of how to easily convert typical k-NN or 
radius-NN neighborhoods to CSR format.


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
