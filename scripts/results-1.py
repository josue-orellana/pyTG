import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import loadmat
from pyTG import torusGraphs

#load Data
dataMat = loadmat('../data/anglebeta.mat')
X = dataMat['anglebeta']
# explicitly set positions
posNodes = {0: (21, 5), 1: (22, 4.7), 2: (23, 5), 3: (24, 4.7), 4: (25, 5), \
      5: (2, 6), 6: (3, 5.7), 7: (4, 6), 8: (5, 5.7), 9: (6, 6), 10: (7, 5.7), 11: (8, 6), 12: (9, 5.7), \
      13: (-2, 4), 14: (-1, 3.7), 15: (0, 4),\
      16: (12, 3), 17: (13, 2.7), 18: (14, 3), 19: (15, 2.7), 20: (16, 3), 21: (17, 2.7), 22: (18, 3), 23: (19, 2.7)}
posNodeGroup = {'CA3': (21, 5), 'DG': (2, 6), 'Sub': (-2, 4), 'PFCv': (12, 3)}

nodeGroupLabels= [x[0].tolist().pop() for x in dataMat['subregions']]
groupLabels = ('CA3', 'DG', 'Sub', 'PFCv')
selMode = (True, True, False)
nodeGraph, groupNodeGraph, num, nodePairs, groupNodePairs, phiHat, covPhiHat \
   = torusGraphs(X, edgesAlphaLevel=0.05, selMode=selMode, groupLabels = groupLabels, nodeGroupLabels = nodeGroupLabels, groupEdgesAlphaLevel=0.001/6)

fig, ax= plt.subplots(1,2)
plt.axes(ax[0])
nx.draw_networkx(nodeGraph, posNodes)
ax[0].margins(0.20)
plt.text(23,5.5,'CA3', fontsize = 13)
plt.text(3.5,6.5,'DG', fontsize = 13)
plt.text(-10,4.3,'Sub', fontsize = 13)
plt.text(12,2.1,'PFCv', fontsize = 13)
ax[0].set_aspect(1/ax[0].get_data_ratio())
plt.axis("off")
plt.axes(ax[1])
plt.imshow(nx.to_numpy_matrix(nodeGraph))
plt.show()

options = {
   "font_size": 20,
   "node_size": 5000,
   "node_color": "white",
   "edgecolors": "black",
   "linewidths": 5,
   "width": 5,
}

nx.draw_networkx(groupNodeGraph, posNodeGroup, **options)
ax = plt.gca()
ax.margins(0.40)
plt.axis("off")
plt.show()