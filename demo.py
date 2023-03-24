import numpy as np
import sys, os
import matplotlib.lines as mlines # compat. with Python 2
from matplotlib import cm
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                              "./python/bin"))

from pgeof import pgeof

import numpy as np

# Parameters
num_points = 100
k_min = 1
verbose = True

# Generate synthetic data
# xyz:    [N, 3] array
#     3D point coordinates
# nn: [num_neighborhoods] array
#     Flattened neighbor indices. Make sure those are all positive, '-1' indices
#     will either crash or silently compute incorrect features
# nn_ptr: [N+1] array
#     Pointers wrt `nn`. More specifically, the neighbors of point `i` are
#     `nn[nn_ptr[i]:nn_ptr[i + 1]]`
xyz = np.random.rand(num_points, 3)
nn_ptr = np.r_[0, np.random.randint(low=0, high=10, size=num_points).cumsum()]
nn = np.random.randint(low=0, high=num_points, size=nn_ptr[-1])

# Make sure xyz are float32 and nn and nn_ptr are uint32
xyz = xyz.astype('float32')
nn_ptr = nn_ptr.astype('uint32')
nn = nn.astype('uint32')

# Make sure arrays are contiguous (C-order) and not Fortran-order
xyz = np.ascontiguousarray(xyz)
nn_ptr = np.ascontiguousarray(nn_ptr)
nn = np.ascontiguousarray(nn)

# Features have shape [N, 11]:
#   0 - linearity
#   1 - planarity
#   2 - scattering
#   3 - verticality
#   4 - normal_x
#   5 - normal_y
#   6 - normal_z
#   7 - length
#   8 - surface
#   9 - volume
#  10 - curvature
geof = pgeof(xyz, nn, nn_ptr, k_min, verbose)

# WARNING: we can trust the direction of the eigenvectors but their senses might
# fluctuate. So you may want to define a standard sense for your normals (eg
# normals all expressed with positive z-coordinates)
