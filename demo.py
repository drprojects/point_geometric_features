import sys
import os.path as osp
sys.path.append(osp.join(osp.realpath(osp.dirname(__file__)), "python/bin"))
from pgeof import pgeof
import numpy as np

# Parameters
num_points = 10000
k_min = 10
k_step = 1
k_min_search = 10
verbose = True

# Generate synthetic data
# xyz: [N, 3] 2D array
#     3D point coordinates
# nn: [num_neighborhoods] 1D array
#     Flattened neighbor indices. Make sure those are all positive, '-1'
#     indices will either crash or silently compute incorrect features
# nn_ptr: [N+1] 1D array
#     Pointers wrt `nn`. More specifically, the neighbors of point `i`
#     are `nn[nn_ptr[i]:nn_ptr[i + 1]]`

# xyz = np.random.rand(num_points, 3)
# nn_ptr = np.r_[0, np.random.randint(low=0, high=30, size=num_points).cumsum()]
# nn = np.random.randint(low=0, high=num_points, size=nn_ptr[-1])

k = 30
xyz = np.load('cube_in_cube.npy')
xyz = np.ascontiguousarray(xyz)
num_points = xyz.shape[0]

dist = ((xyz.reshape(1, num_points, 3) - xyz.reshape(num_points, 1, 3))**2).sum(axis=-1)
nn = dist.argsort(axis=1)[:, :k]

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(xyz)
nn = neigh.kneighbors(xyz, k, return_distance=False)


nn = nn.flatten()
nn_ptr = np.r_[0, np.full(num_points, k).cumsum()]

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
geof = pgeof(
    xyz, nn, nn_ptr, k_min=k_min, k_step=k_step, k_min_search=k_min_search,
    verbose=verbose)

np.save("/home/ign.fr/drobert-admin/projects/point_geometric_features/cube_in_cube_geof.npy", geof)

# WARNING: we can trust the orientation of the eigenvectors but their
# senses might fluctuate. So you may want to define a standard sense for
# your normals (eg normals all expressed with positive z-coordinates)
