# distutils: language=c++

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
cimport cython
from math import sqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport acos as cacos
from libc.math cimport sqrt as csqrt

import os
from yaml import load
from matplotlib.pyplot import imread

cdef cranges_to_xy(np.float32_t[:] ranges, np.float32_t[:] angles,
                 np.float32_t[:] x, np.float32_t[:] y):
    for i in range(ranges.shape[0]):
        x[i] = ranges[i] * ccos(angles[i])
        y[i] = ranges[i] * csin(angles[i])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def euclidean_clustering(np.float32_t[:] ranges, np.float32_t[:] angles, thresh=0.5):
    # assume angles from 0 to 2pi
    cdef int n_points = ranges.shape[0]
    cdef np.float32_t[:] x = np.zeros((n_points), dtype=np.float32)
    cdef np.float32_t[:] y = np.zeros((n_points), dtype=np.float32)
    # get points as x y coordinates
    cranges_to_xy(ranges, angles, x, y)

    cdef np.float32_t THRESH = thresh
    cdef np.float32_t THRESH_SQ = THRESH * THRESH

    clusters = []
    cdef bool create_new_cluster = True
    cdef np.float32_t xi
    cdef np.float32_t yi
    cdef np.float32_t dx
    cdef np.float32_t dy
    # for each point:
    # find a cluster within THRESH of point, add point to that cluster.
    # if no cluster is found within THRESH of point, create new cluster with point in it
    for i in range(n_points):
        create_new_cluster = True
        xi = x[i]
        yi = y[i]
        for j in range(len(clusters)):
            c = clusters[j]
            for k in range(len(c)):
                p = c[k]
                dx = xi - x[p]
                dy = yi - y[p]
                if ( dx * dx + dy * dy ) < THRESH_SQ:
                    c.append(i)
                    create_new_cluster = False
                    break
            if not create_new_cluster:
                break
        if create_new_cluster:
            clusters.append([i])

    return clusters, x, y

def cluster_ids(n_points, clusters):
    """ returns for each point, its cluster id """
    max_id = np.max(np.max(clusters))
    if max_id > n_points:
        raise ValueError("n_points ({}) smaller than cluster point ids ({}) for clusters {}".format(
            n_points, max_id, clusters))
    cdef np.int64_t[:] cluster_ids = np.zeros((n_points), dtype=np.int64)

    for i in range(len(clusters)):
        c = clusters[i]
        for p in c:
            cluster_ids[int(p)] = i

    return cluster_ids

def cluster_sizes(n_points, clusters):
    """ returns for each point, its cluster size """
    max_id = np.max(np.max(clusters))
    if max_id > n_points:
        raise ValueError("n_points ({}) smaller than cluster point ids ({}) for clusters {}".format(
            n_points, max_id, clusters))
    cluster_sizes = np.zeros((n_points), dtype=np.int64)

    for i in range(len(clusters)):
        c = clusters[i]
        for p in c:
            cluster_sizes[int(p)] = len(c)

    return cluster_sizes
