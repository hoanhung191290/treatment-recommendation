import csv
from numpy import genfromtxt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
from scipy.cluster.hierarchy import fclusterdata
from sklearn.neighbors import BallTree
import sklearn.metrics.pairwise
import scipy.spatial.distance
import sklearn.utils as skl
# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
pars = {}
with open('configClustering.txt') as f1:

    for line in f1:
        searchObj = re.search(r'(.*) = (.*)', line, re.M | re.I)
        pars[searchObj.group(1)]=searchObj.group(2)

# read flash.dat to a list of lists

# read flash.dat to a list of lists


wIdxClusters = open('./idxClusters.txt','w')


max_d = 50
idxs = []
nCluster = int(pars['noCluster'])
rand_state = int(pars['random_state'])
linkage_type = pars['linkage']
dist_metric = pars['distance_metric']
testsize = int(pars['testsize'])


with open('data/derived/patientIdx.txt') as f:
    idxs = f.read().split()

def cluster_indices(cluster_assignments):
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    return indices

# write it as a new CSV file
i=0
low = i * testsize
up = (i + 1) * testsize
probsMat = genfromtxt('data/derived/mixedOutStat.csv', delimiter=',')
#probsMat = pd.read_csv('data/derived/mixedOutStat.csv')
indices = range(probsMat.shape[0])
idxes = skl.shuffle(range(len(probsMat)), random_state=0)
testIdxes = idxes[low:up]

trainIdxes = list(set(idxes) - set(testIdxes))
trainIdxes = [int(t) for t  in trainIdxes]
#print(probsMat)
train=probsMat[trainIdxes,:]
test = probsMat[testIdxes,:]

#train, test ,indexTrain, indexTest = train_test_split(probsMat, indices, random_state= rand_state, test_size=testsize)

#cA_hamming = distance.squareform(pairwise_distances(train,))
Y = scipy.spatial.distance.pdist(train, dist_metric)
Z = linkage(Y,linkage_type)
#print(my_data)
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('patient index  or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
fancy_dendrogram(
    Z,
    #truncate_mode='lastp',  # show only the last p merged clusters
    p=19,  # show only the last p merged clusters
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=12.,  # font size for the x axis labels
    show_leaf_counts=True,
    show_contracted = True,
    annotate_above=10,
    max_d = 0.4,
)
plt.show()


last = Z[-10:, 10]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)


cutoff = nCluster
cluster_assignments = fclusterdata(train,cutoff, criterion='maxclust',metric=dist_metric,method=linkage_type,depth=1)
num_clusters = cluster_assignments.max()
print("%d clusters" % num_clusters)
indices = cluster_indices(cluster_assignments)
for ind in indices:
    print(len(ind))
for k, ind in enumerate(indices):
    print("cluster", k + 1, "is", len(ind))
    wIdxClusters.write(",".join(str(idxs[indexTrain[x]]) for x in ind))
    wIdxClusters.write('\n')
wIdxClusters.write(",".join(str(idxs[x]) for x in indexTest) + '\n')

ballt = BallTree(probsMat, leaf_size = 30, metric = dist_metric)
distances, neighbors = ballt.query(probsMat, k=2)
neighDf = pd.DataFrame(neighbors)
neighDf.columns = ['idx','neighbor1']
neighDf.to_csv('neighbor.csv',index=False)
