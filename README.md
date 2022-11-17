<div align="center">

# Point Geometric Features

[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)


</div>


## 📌 Description

Python wrapper around C++ helper to compute the following local geometric features in parallel on CPU:
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


## 🧱 Installation

Tested

```bash
# clone project
https://github.com/drprojects/point_geometric_features.git
cd point_geometric_features

# Installation in a dedicated conda environment
bash install.sh
```

## 🚀 Demo

```bash
python demo.py
```


## 💳 Credits
This implementation was largely inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph). The main modifications here allow: 
- parallel computation on all points' local neighborhoods, with neighborhoods of varying sizes
- more geometric features
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