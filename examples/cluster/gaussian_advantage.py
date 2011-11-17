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
import pylab
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds
from sklearn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample data
centers = [[0., 1.5],[0, -1.5]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=0.75)
print len(X)
# Set bandwidths manually (these have to be tuned!)
gauss_bandwidth = 0.9
flat_bandwidth = 2.7

### Get seeds
seeds = get_bin_seeds(X, 0.25, min_bin_freq=3)

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
    for rad, alpha in [(gauss_bandwidth*1, 0.3), (gauss_bandwidth*2, 0.15), (gauss_bandwidth*3, 0.1)]:
        circle = pylab.Circle(cluster_center,radius=rad, alpha=alpha, color=col)
        gauss_axes.add_patch(circle)

gauss_axes.set_ylabel("Gaussian kernel")
for line in gauss_axes.get_xticklines() + gauss_axes.get_yticklines() + gauss_axes.get_xticklabels() + gauss_axes.get_yticklabels(): 
    line.set_visible(False) 

flat_axes = fig.add_subplot(212)
for k, col in zip(range(n_clusters_flat), colors):
    flat_members = labels_flat == k
    cluster_center = cluster_centers_flat[k]
    flat_axes.plot(X[flat_members, 0], X[flat_members, 1], col + '.')
    flat_axes.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=14)

    circle = pylab.Circle(cluster_center,radius=flat_bandwidth, alpha=0.3, color=col)
    flat_axes.add_patch(circle)
    
    
flat_axes.set_ylabel("Flat kernel")
for line in flat_axes.get_xticklines() + flat_axes.get_yticklines() + flat_axes.get_xticklabels() + flat_axes.get_yticklabels(): 
    line.set_visible(False) 

pl.show()
