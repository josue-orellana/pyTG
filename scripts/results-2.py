import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import loadmat
from pyTG import (compNodePairsNodes, phaseLockingValue)

#load Data
dataMat = loadmat('../data/anglebeta.mat')
X = dataMat['anglebeta']
# explicitly set positions
posNodes = {0: (21, 5), 1: (22, 4.7), 2: (23, 5), 3: (24, 4.7), 4: (25, 5), \
      5: (2, 6), 6: (3, 5.7), 7: (4, 6), 8: (5, 5.7), 9: (6, 6), 10: (7, 5.7), 11: (8, 6), 12: (9, 5.7), \
      13: (-2, 4), 14: (-1, 3.7), 15: (0, 4),\
      16: (12, 3), 17: (13, 2.7), 18: (14, 3), 19: (15, 2.7), 20: (16, 3), 21: (17, 2.7), 22: (18, 3), 23: (19, 2.7)}

# %%  PLV, 24dim network graph, figures S8
num = {'nodes': X.shape[0]}
nodePairs = {'nodes':compNodePairsNodes(num['nodes'])}
plv, plvPvals = phaseLockingValue(X, nodePairs)

plvPairsElecActive = nodePairs['nodes'][plvPvals <= 0.0005, :]

plvGraph = nx.Graph()
plvGraph.add_nodes_from(range(num['nodes']))
plvGraph.add_edges_from(plvPairsElecActive)

fig, ax= plt.subplots(1,2)
plt.axes(ax[0])
nx.draw_networkx(plvGraph, posNodes)
plt.text(23,5.5,'CA3', fontsize = 13)
plt.text(3.5,6.5,'DG', fontsize = 13)
plt.text(-10,4.3,'Sub', fontsize = 13)
plt.text(12,2.1,'PFCv', fontsize = 13)
ax[0].set_aspect(1/ax[0].get_data_ratio())
plt.axis("off")
plt.axes(ax[1])
plt.imshow(nx.to_numpy_matrix(plvGraph))
plt.show()