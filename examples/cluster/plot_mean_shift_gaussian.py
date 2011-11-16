"""
=============================================
A demo of the mean-shift clustering algorithm
=============================================

Reference:

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.

"""
print __doc__

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds
from sklearn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample data
centers1 = [[0., 0.25],[-1, 0.20], [-1, 1], [0, 1], [1, 1]]
centers2 = [[-3.0, -2], [0, -2], [3.0,-2]]
X1, _ = make_blobs(n_samples=5000, centers=centers1, cluster_std=0.25)
X2, _ = make_blobs(n_samples=5000, centers=centers2, cluster_std=0.75)
X = np.concatenate((X1, X2), axis=0)


# Set bandwidths manually (these have to be tuned!)
gauss_bandwidth = 0.33
flat_bandwidth = 0.5

### Get seeds
seeds = get_bin_seeds(X, 0.25, min_bin_freq=10)

###############################################################################
# Compute clustering with gaussian MeanShift

ms_gauss = MeanShift(bandwidth=gauss_bandwidth, kernel="gaussian", seeds=seeds)
ms_gauss.fit(X)
labels_gauss = ms_gauss.labels_
cluster_centers_gauss = ms_gauss.cluster_centers_
labels_unique_gauss = np.unique(labels_gauss)
n_clusters_gauss = len(labels_unique_gauss)

# Compute clustering with flat MeanShift

ms_flat = MeanShift(bandwidth=flat_bandwidth, kernel="flat", seeds=seeds)
ms_flat.fit(X)
labels_flat = ms_flat.labels_
cluster_centers_flat = ms_flat.cluster_centers_
labels_unique_flat = np.unique(labels_flat)
n_clusters_flat = len(labels_unique_flat)



###############################################################################
# Plot result
import pylab as pl
from itertools import cycle

fig = pl.figure(figsize=(3.,5.))
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

gauss_axes = fig.add_subplot(211)
for k, col in zip(range(n_clusters_gauss), colors):
    gauss_members = labels_gauss == k
    cluster_center = cluster_centers_gauss[k]
    gauss_axes.plot(X[gauss_members, 0], X[gauss_members, 1], col + '.')
    gauss_axes.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=14)
gauss_axes.set_ylabel("Gauss - bandwidth %0.2f" % gauss_bandwidth)

flat_axes = fig.add_subplot(212)
for k, col in zip(range(n_clusters_flat), colors):
    flat_members = labels_flat == k
    cluster_center = cluster_centers_flat[k]
    flat_axes.plot(X[flat_members, 0], X[flat_members, 1], col + '.')
    flat_axes.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=14)
flat_axes.set_ylabel("Flat - bandwidth %0.2f" % flat_bandwidth)

pl.show()
