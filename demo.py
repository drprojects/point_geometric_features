import pgeof
import numpy as np

# Generate a random synthetic point cloud
num_points = 10000
xyz = np.random.rand(num_points, 3)

# Manually generating random neighbors in CSR format
nn_ptr = np.r_[0, np.random.randint(low=0, high=30, size=num_points).cumsum()]
nn = np.random.randint(low=0, high=num_points, size=nn_ptr[-1])

# Converting k-nearest neighbors to CSR format
from sklearn.neighbors import NearestNeighbors
k = 20
kneigh = NearestNeighbors(n_neighbors=k).fit(xyz).kneighbors(xyz)
nn_ptr = np.arange(num_points + 1) * k
nn = kneigh[1].flatten()

# Converting radius neighbors to CSR format
from sklearn.neighbors import NearestNeighbors
radius = 0.1
rneigh = NearestNeighbors(radius=radius).fit(xyz).radius_neighbors(xyz)
nn_ptr = np.r_[0, np.array([x.shape[0] for x in rneigh[1]]).cumsum()]
nn = np.concatenate(rneigh[1])

# Make sure xyz are float32 and nn and nn_ptr are uint32
xyz = xyz.astype('float32')
nn_ptr = nn_ptr.astype('uint32')
nn = nn.astype('uint32')

# Make sure arrays are contiguous (C-order) and not Fortran-order
xyz = np.ascontiguousarray(xyz)
nn_ptr = np.ascontiguousarray(nn_ptr)
nn = np.ascontiguousarray(nn)

# Print details on how pgeof works and expected input parameters
print(help(pgeof))

# Features have shape [N, 12]:
#   0 - linearity
#   1 - planarity
#   2 - scattering
#   3 - verticality
#   4 - normal_x (oriented towards positive z-coordinates)
#   5 - normal_y (oriented towards positive z-coordinates)
#   6 - normal_z (oriented towards positive z-coordinates)
#   7 - length
#   8 - surface
#   9 - volume
#  10 - curvature
#  11 - optimal neighborhood size (if 'k_step' argument is used)
geof = pgeof(
    xyz, nn, nn_ptr, k_min=10, k_step=1, k_min_search=15,
    verbose=True)
