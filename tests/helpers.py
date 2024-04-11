import numpy as np
from scipy.spatial import KDTree


def random_nn(num_points, k):
    # Generate a random synthetic point cloud
    xyz = np.random.rand(num_points, 3)

    # Converting k-nearest neighbors to CSR format
    kneigh = KDTree(xyz).query(xyz, k=k, workers=-1)
    nn_ptr = np.arange(num_points + 1) * k
    nn = kneigh[1].flatten()

    # Make sure xyz are float32 and nn and nn_ptr are uint32
    xyz = xyz.astype("float32")
    nn_ptr = nn_ptr.astype("uint32")
    nn = nn.astype("uint32")

    # Make sure arrays are contiguous (C-order) and not Fortran-order
    xyz = np.ascontiguousarray(xyz)
    nn_ptr = np.ascontiguousarray(nn_ptr)
    nn = np.ascontiguousarray(nn)
    return xyz, nn, nn_ptr