""" Python module for Torus Graphs for multivariate phase coupling analysis.

This module includes function for fitting and sampling from a torus graphs model
as described in Klein*Orellana*. 

Reference:
Klein N*, Orellana J*, Brincat SL, Miller EK, Kass RE. 
Torus graphs for multivariate phase coupling analysis.
Annals of Applied Statistics. 2020;14(2):635–660.

  Typical usage example: Four conditionally independent nodes

  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> import networkx as nx
  >>> # Generate some random numbers to illustrate data input
  >>> rng = np.random.default_rng(seed=42)
  >>> X = rng.vonmises(0, 0, (4, 1000)) # 4 nodes, 1000 trials
  >>> # Torus graph with default settings:
  >>> elecEdgesGraph, *otherOutputs = torusGraphs(X)
  >>> # explicitly set positions
  >>> posElec = {0: (0, 5), 1: (0, 0), 2: (5, 5), 3: (5, 0)}
  >>> nx.draw_networkx(elecEdgesGraph, posElec)
  >>> ax = plt.gca()
  >>> axLabel = ax.set_xlabel('Four conditionally independent nodes')
  >>> ax.margins(0.20)
  >>> plt.show()

"""

import numpy as np
import pandas as pd

from numpy import pi
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2
from scipy.special import iv as besseli
import networkx as nx

def compNodePairsNodes(numNodes):
    """ Returns electrode pairs.
    
    Args:
        numNodes: int 
            Number of nodes that you wish to pair.
    
    Outputs: 
        nodePairsNodes: numpy array (numNodePairs, 2)
            Each row of ints is a (j, k) index pair, with j starting at 0 
            and such that j<k<numNodes.
    
    >>> a = compNodePairsNodes(numNodes = 4)
    >>> a[0,:] # first pair of nodes
    array([0, 1])
    >>> b = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
    >>> np.array_equal(a, b)
    True

    """
    numNodePairs = int(numNodes*(numNodes-1)/2)
    nodePairsNodes = np.zeros((numNodePairs, 2), dtype=int)
    inc = 0
    for i in range(numNodes):
        for j in range(i+1,numNodes):
            nodePairsNodes[inc,:] = [i, j]
            inc +=1

    return nodePairsNodes

def compNodePairsParamInds(selMode,num):
    """ References the parameters associated with each node pair.

    The indices are tied to selMode, and the concatenation of the 
    sufficient statistics blocks (see :meth:`funsTG.compSufStat`).

    Args:
        described in :meth:`funsTG.torusGraphs`.
    
    Outputs:
        nodePairsParamInds: list of lists with num['nodePairs'] elements.

    >>> selMode = (True, True, True)
    >>> num = {'nodes':3, 'nodePairs':3, 'param': 18}
    >>> a = compNodePairsParamInds(selMode, num) 
    >>> b = [[6,9,12,15], [7,10,13,16], [8,11,14,17]]
    >>> a == b
    True

    >>> selMode = (False, True, False)
    >>> num = {'nodes':4, 'nodePairs':6, 'param': 12}
    >>> c = compNodePairsParamInds(selMode, num) 
    >>> d = [[0,6], [1,7], [2,8], [3,9], [4,10], [5,11]]
    >>> c == d
    True

    """

    nodePairsParamInds = [[] for _ in range(num['nodePairs']) ]

    if selMode[0]:
        startInd = 2*num['nodes']
    else:
        startInd = 0

    endInd = num['param']
    step4Ind = num['nodePairs']

    for r in range(num['nodePairs']):
        nodePairsParamInds[r] = list(range(startInd, endInd, step4Ind))
        startInd += 1
    # nodePairsParamInds = tupleAListOfLists(nodePairsParamInds)

    return nodePairsParamInds

def compGroupNodeInds(num, groupLabels, nodeGroupLabels):
    """ Computes node indices associated with each group label.

    Args:
        Described in :meth:`funsTG.torusGraphs`.
        groupLabels must be a List or Tuple. 

    Outputs:
        groupNodeInds: list of lists with num['groupNodes'] elements.

    >>> num = {'groupNodes': 3}
    >>> groupLabels = [0, 1, 2]
    >>> nodeGroupLabels = [1, 0, 2, 1, 2]
    >>> a = compGroupNodeInds(num, groupLabels, nodeGroupLabels)
    >>> b = [[1], [0, 3], [2, 4]]
    >>> a == b
    True

    >>> groupLabels = ['G0', 'G1', 'G2']
    >>> nodeGroupLabels = ['G1', 'G0', 'G2', 'G1', 'G2']
    >>> c = compGroupNodeInds(num, groupLabels, nodeGroupLabels)
    >>> d = [[1], [0, 3], [2, 4]]
    >>> c == d
    True

    """

    groupNodeInds = [[] for _ in range(num['groupNodes']) ]
    for iNode,iGroupLabel in enumerate(nodeGroupLabels):
        iLabelIndex = groupLabels.index(iGroupLabel)
        groupNodeInds[iLabelIndex].append(iNode)
    
    return groupNodeInds

def compGroupNodePairsNodePairInds(num, groupLabels, nodeGroupLabels, nodePairs, groupNodePairs):
    """ References group-node-pairs with node-pair indices.

    Args:
        Described in :meth:`funsTG.torusGraphs`.

    Outputs: 
        groupNodePairsNodePairInds: list of lists with num['groupNodePairs'] elements. 
        Contains the corresponding indices of the node-pairs associated with each group-pair.
    
    The example below has 4 nodes and three groups, then there are three group edges.
    The first group-node-pair is G0-G1 and the corresponding node pairs are (0,1) and (0,3);
    their respective indices in nodePairs['nodes'] are pairs 0 and 2.  
    The second group-node-pair is G0-G2 and the corresponding node pair is (0,2)
    with nodePairs['nodes'] index 1.
    The third group-node-pair is G1-G2 and the corresponding node pairs are (1,2) and (2,3)
    with nodePairs['nodes'] indices 3 and 5. 

    Note that node-pair index (1,3) -- nodePairs['nodes'] index 4 -- is within group G1 and therefore it is not assigned
    to any of the group-node pairs. 

    >>> num = {'nodePairs':6, 'groupNodePairs':3}
    >>> groupLabels = ['G0', 'G1', 'G2']
    >>> nodeGroupLabels = ['G0', 'G1', 'G2', 'G1']
    >>> nodePairs = {'nodes': np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])}
    >>> groupNodePairs = {'nodes': np.array([[0,1], [0,2], [1,2]])}
    >>> a = compGroupNodePairsNodePairInds(num, groupLabels, nodeGroupLabels, nodePairs, groupNodePairs)
    >>> b = [[0, 2], [1], [3, 5]]
    >>> a == b
    True

    """
    
    groupNodePairsNodePairInds = [[] for _ in range(num['groupNodePairs']) ]
    # Step through all node-pairs
    for r in range(num['nodePairs']):
        rGroupLabelA = nodeGroupLabels[ nodePairs['nodes'][r,0] ]
        rGroupLabelB = nodeGroupLabels[ nodePairs['nodes'][r,1] ]
        # group-node pairs are made out of distinct group-nodes
        if (rGroupLabelA != rGroupLabelB): 
            rGroupIndA = groupLabels.index(rGroupLabelA)
            rGroupIndB = groupLabels.index(rGroupLabelB)

            rGroupIndA, rGroupIndB = min(rGroupIndA, rGroupIndB), max(rGroupIndA, rGroupIndB)
            
            # finding index of this group-pair
            rIndPairAB = list(np.logical_and(
                        groupNodePairs['nodes'][:,0] == rGroupIndA, 
                        groupNodePairs['nodes'][:,1] == rGroupIndB ))
            # append node-pair index to the corresponding group-pair
            groupNodePairsNodePairInds[rIndPairAB.index(True)].append(r) 

    return groupNodePairsNodePairInds

def compGroupNodePairsParamInds(num, nodePairs, groupNodePairs):
    """ References the parameters associated with each group-node pair.

    The indices are tied to nodePairs['paramInds']   
    (see :meth:`compNodePairsParamInds`).

    Args:
        described in :meth:`funsTG.torusGraphs`.
    
    Outputs:
        nodePairsParamInds: list of lists with num['groupNodePairs'] elements.
    
    >>> selMode = (False, True, False)
    >>> num = {'groupNodePairs': 3, 'nodes':4, 'nodePairs':6, 'param': 12}
    >>> groupNodePairs = {'nodePairInds': [[0, 2], [1], [3, 5]]}
    >>> nodePairs = {'paramInds': compNodePairsParamInds(selMode, num)}
    >>> # nodePairs['paramInds'] = [[0,6], [1,7], [2,8], [3,9], [4,10], [5,11]]
    >>> a = compGroupNodePairsParamInds(num, nodePairs, groupNodePairs)
    >>> b = [[0,6, 2,8], [1,7], [3,9, 5,11]]
    >>> a == b
    True

    """

    groupNodePairsParamInds = [[] for _ in range(num['groupNodePairs']) ]
    # Step through all group-node pairs
    for r in range(num['groupNodePairs']):
        rNodePairs = np.array(groupNodePairs['nodePairInds'][r])
        rNodePairsParamInds = np.array(nodePairs['paramInds'])[rNodePairs].flatten()
        groupNodePairsParamInds[r].extend(rNodePairsParamInds)

    return groupNodePairsParamInds

def compSufStat(X, selMode, nodePairs):
    """ Computes the sufficient statistics of the torus graph model.

    The sufficient statistics are functions of the data X;
    selMode determines the types of blocks that are included.  

    Let

    ``Xi = X[nodePairs['nodes'][:,0],:]``

    ``Xj = X[nodePairs['nodes'][:,1],:]``

    ``Xdif = Xi - Xj``

    ``Xsum = Xi + Xj``

    Args:
        described in :meth:`funsTG.torusGraphs`.
    
    Outputs:
        H : numpy array (num['param'], num['trials']).
            H is the Laplacian, with respect to the data, of the suficient statistics. 
            The sufficient statistics concatenate sC, sS, sAlpha, sBeta, sGamma, and sDelta.
            After taking the laplacian (second derivative and sum), H is equal to concatenating  
            sC, sS, 2*sAlpha, 2*sBeta, 2*sGamma, 2*sDelta.
            However, whether these blocks are included depends on selMode.

        sC : cos(X)

        sS : sin(X)

        sAlpha : cos(Xdif)

        sBeta : sin(Xdif)

        sGamma : cos(Xsum)

        sDelta = sin(Xsum)

    >>> selMode = (True, True, True) 
    >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ]) }
    >>> X = np.array([[0, pi], [pi, pi/2], [pi/2, pi/3] ]) # 3 nodes, 2 trials
    >>> Xi = X[nodePairs['nodes'][:,0],:]
    >>> Xj = X[nodePairs['nodes'][:,1],:]
    >>> H, sC, sS, sAlpha, sBeta, sGamma, sDelta = compSufStat(X, selMode, nodePairs)
    >>> h = np.concatenate((np.cos(X), np.sin(X), 2*np.cos(Xi-Xj), 2*np.sin(Xi-Xj), 2*np.cos(Xi+Xj), 2*np.sin(Xi+Xj) ), axis = 0)
    >>> np.all(abs(H - h) < 1e-12)
    True

    """

    # Sufficient Statistics
    Xi = X[nodePairs['nodes'][:,0],:]
    Xj = X[nodePairs['nodes'][:,1],:]
    H = []
    if selMode[0]:
        sC = np.cos(X)
        sS = np.sin(X)
        H.extend(sC)
        H.extend(sS)
    else:
        sC = None
        sS = None

    if selMode[1]:
        Xdif = Xi - Xj
        sAlpha = np.cos(Xdif)
        sBeta = np.sin(Xdif)
        H.extend(2*sAlpha)
        H.extend(2*sBeta)
    else:
        sAlpha = None
        sBeta = None

    if selMode[2]:
        Xsum = Xi + Xj
        sGamma = np.cos(Xsum)
        sDelta = np.sin(Xsum)
        H.extend(2*sGamma)
        H.extend(2*sDelta)
    else:
        sGamma = None
        sDelta = None

    H = np.array(H)

    return (H, sC, sS, sAlpha, sBeta, sGamma, sDelta)

def compGammaXt(t, num, selMode, nodePairs, 
                sC, sS, sAlpha, sBeta, sGamma, sDelta ):
    """ Computes a function Gamma for a single trial from X. 

    Let Dt (num['param'], num[''nodes]) be the Jacobian of the suficient statistics; 
    by Jacobian we mean the first derivative with respect to the data, then

    ``gammaXt = Dt @ Dt.T # np.matmul(Dt,Dt.T)``

    Args:
        described in :meth:`funsTG.torusGraphs` and :meth:`funsTG.compSufStat`.
    
    Outputs:
        gammaXt : numpy array (num['param'], num['param']).

    >>> X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
    >>> selMode = (True, True, True)
    >>> num = {'nodes': 3, 'trials': 6, 'nodePairs': 3, 'param': 18}
    >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ])}
    >>> H, sC, sS, sAlpha, sBeta, sGamma, sDelta = compSufStat(X, selMode, nodePairs)
    >>> gammaXt = compGammaXt(1, num, selMode, nodePairs, sC, sS, sAlpha, sBeta, sGamma, sDelta)
    >>> sum(gammaXt.flatten())
    15.5466459624921

    """
    # define gammaXt (t-th trial)
    Dt = []

    if selMode[0]:
        dC = -np.diag(sS[:,t])
        dS =  np.diag(sC[:,t])
        Dt.extend(dC)
        Dt.extend(dS)
    if selMode[1]:
        dAlpha = np.zeros((num['nodePairs'],num['nodes']))
        dBeta =  np.zeros((num['nodePairs'],num['nodes']))
    if selMode[2]:
        dGamma = np.zeros((num['nodePairs'],num['nodes']))
        dDelta = np.zeros((num['nodePairs'],num['nodes']))

    for r in range(num['nodePairs']):
        colInds = [ nodePairs['nodes'][r,0], \
                    nodePairs['nodes'][r,1] ]
        if selMode[1]:
            dAlpha[r, colInds] = np.array([-1,  1]) * sBeta[r, t]
            dBeta[r, colInds]  = np.array([ 1, -1]) * sAlpha[r, t]
        if selMode[2]:
            dGamma[r, colInds] = np.array([-1, -1]) * sDelta[r, t]
            dDelta[r, colInds] = np.array([ 1,  1]) * sGamma[r, t]
    if selMode[1]:
        Dt.extend(dAlpha)
        Dt.extend(dBeta)
    if selMode[2]:
        Dt.extend(dGamma)
        Dt.extend(dDelta)
    Dt = np.array(Dt)

    gammaXt = Dt @ Dt.T # np.matmul(Dt,Dt.T)

    return gammaXt

def compPhiHatAndCovPhiHat(X, num, selMode, nodePairs):
    """ Computes the parameter estimates and their covariance. 
    
    Args:
        described in :meth:`funsTG.torusGraphs`.
    
    Outputs: 
        phiHat : numpy array with num['param'] elements.
            Model parameters estimates.
        
        covPhiHat : numpy matrix (num['param'], num['param']) 
            Covariance for the model parameter estimates.

    >>> X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
    >>> selMode = (True, True, True)
    >>> num = {'nodes': 3, 'trials': 6, 'nodePairs': 3, 'param': 18}
    >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ])}
    >>> phiHat, covPhiHat = compPhiHatAndCovPhiHat(X, num, selMode, nodePairs)
    >>> phiHat
    array([-39.14836993,   7.87810064,  39.42105888, -16.63875025,
            64.03174047, -47.82433269,  25.35498747, -31.41571978,
            58.81343556, -50.01232377,  -5.12549116,  11.0160441 ,
           -18.56673785,  -1.31352256,  16.1610135 , -23.23622016,
           -31.69057573, -34.39433503])
    >>> sum(covPhiHat.flatten())
    23556.69806086992

    """
    # Sufficient Statistics
    H, sC, sS, sAlpha, sBeta, sGamma, sDelta = \
        compSufStat(X, selMode, nodePairs)

    # define compPhiHatAndCovPhiHat
    HhatX = np.mean(H,1)

    # gammaHatX: average of gammaXt over trials
    gammaHatX = np.zeros((num['param'],num['param']))
    for t in range(num['trials']):
        gammaXt = compGammaXt(t, num, selMode, nodePairs,
                            sC, sS, sAlpha, sBeta, sGamma, sDelta)
        gammaHatX += gammaXt
    gammaHatX /= num['trials']   

    # Estimating phiHat
    phiHat = np.linalg.inv(gammaHatX) @ HhatX

    # Estimating the covariance of phiHat: covPhiHat
    vHatX = np.zeros((num['param'],num['param']))
    for t in range(num['trials']):
        gammaXt = compGammaXt(t, num, selMode, nodePairs, 
                            sC, sS, sAlpha, sBeta, sGamma, sDelta)
        vVec = (gammaXt @ phiHat) - H[:,t]
        vVec = np.reshape(vVec,(-1,1))
        vHatX += vVec @ vVec.T
    vHatX /= num['trials']  
    invGammaHatX = np.linalg.inv(gammaHatX)
    covPhiHat = (invGammaHatX @ vHatX @ invGammaHatX)/num['trials']

    return (phiHat, covPhiHat)

def phiToParamGroups(phi, num, selMode=(True,True,True)):
    """ Categorizes the groups of parameters in phi.

    Args:
        described in :meth:`funsTG.torusGraphs`.
    
    Outputs: 
        paramG : dictionary.
            The keys are ['cosMu', 'sinMu', 'cosMuDif', 'sinMuDif', 'cosMuSum', 'sinMuSum'].
            The corresponding fields are numpy arrays, which can be filled with zeros 
            depending on selMode. 
    
    Raises:
        ValueError: when len(phi) doesn't agree with selMode
    
    >>> num = {'nodes': 4, 'nodePairs': 6, 'param':12}
    >>> selMode = (False, True, False)
    >>> phi = np.arange(num['param'])
    >>> paramG = phiToParamGroups(phi, num, selMode)
    >>> all(paramG['cosMu'] == 0)
    True
    >>> all(paramG['sinMu'] == 0)
    True
    >>> len(paramG['cosMuDif']) == num['nodePairs']
    True
    >>> a = np.concatenate((paramG['cosMuDif'], paramG['sinMuDif']), axis = 0)
    >>> all(a == phi)
    True
    >>> all(paramG['cosMuSum'] == 0)
    True
    >>> all(paramG['sinMuSum'] == 0)
    True
    >>> paramG = phiToParamGroups(phi, num, selMode=(True,True,True))
    Traceback (most recent call last):
        ...
    ValueError: The number of parameters in phi doesn't agree with the provided selMode, which expects 32 parameters.

    """

    phi = np.copy(phi) # to avoid changing input phi
    
    numParam = 0
    paramG = dict()
    if selMode[0]:
        paramG['cosMu'], phi  = phi[:num['nodes']], phi[num['nodes']:]
        paramG['sinMu'], phi  = phi[:num['nodes']], phi[num['nodes']:] 
        numParam += 2*num['nodes']
    else:
        paramG['cosMu'] = np.zeros(num['nodes'])
        paramG['sinMu'] = np.zeros(num['nodes'])

    if selMode[1]:
        paramG['cosMuDif'], phi = phi[:num['nodePairs']], phi[num['nodePairs']:]
        paramG['sinMuDif'], phi = phi[:num['nodePairs']], phi[num['nodePairs']:]
        numParam += 2*num['nodePairs']
    else:
        paramG['cosMuDif'] = np.zeros(num['nodePairs'])
        paramG['sinMuDif'] = np.zeros(num['nodePairs'])

    if selMode[2]:
        paramG['cosMuSum'], phi = phi[:num['nodePairs']], phi[num['nodePairs']:]
        paramG['sinMuSum'], phi = phi[:num['nodePairs']], phi[num['nodePairs']:]
        numParam += 2*num['nodePairs']
    else:
        paramG['cosMuSum'] = np.zeros(num['nodePairs'])
        paramG['sinMuSum'] = np.zeros(num['nodePairs'])
    
    if num['param'] != numParam :
        raise ValueError(
        "The number of parameters in phi doesn't agree with the provided selMode, which expects " \
        + str(numParam) + " parameters.")

    return paramG

def phiParamGroupsToMats(num, paramG, nodePairs):
    """ Converts the `paramG` parameter groups to matrices. 

    Args:
        num and nodePairs are described in :meth:`funsTG.torusGraphs`,
        paramG is the output of :meth:`funsTG.phiToParamGroups`
    
    Outputs: 
        matParamG : dictionary.
            The keys are ['cosMu', 'sinMu', 'cosMuDif', 'sinMuDif', 'cosMuSum', 'sinMuSum'].
            These are identical to paramG, except that the pairwise components 
            ('cosMuDif', 'sinMuDif', 'cosMuSum', 'sinMuSum')
            are in matrix form, numpy (num['nodes'], num['nodes']). 
    
    >>> num = {'nodes': 4, 'nodePairs': 6, 'param':12}
    >>> nodePairs = {'nodes': np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])}
    >>> selMode = (False, True, False)
    >>> phi = np.arange(num['param'])
    >>> paramG = phiToParamGroups(phi, num, selMode)
    >>> matParamG = phiParamGroupsToMats(num, paramG, nodePairs)
    >>> a = np.array([ [0,0,1,2],[0,0,3,4],[1,3,0,5],[2,4,5,0] ])
    >>> np.all(abs(matParamG['cosMuDif'] - a) < 1e-12)
    True

    """

    # Only the pairwise components become matrices
    matParamG = {'cosMu': paramG['cosMu'],
                 'sinMu': paramG['sinMu'],
                 'cosMuDif': np.zeros((num['nodes'], num['nodes'])),
                 'sinMuDif': np.zeros((num['nodes'], num['nodes'])),
                 'cosMuSum': np.zeros((num['nodes'], num['nodes'])),
                 'sinMuSum': np.zeros((num['nodes'], num['nodes']))}

    # Step through all node-pairs
    for r in range(num['nodePairs']):
        rPair = (nodePairs['nodes'][r,0], nodePairs['nodes'][r,1])
        matParamG['cosMuDif'][rPair] = paramG['cosMuDif'][r]
        matParamG['sinMuDif'][rPair] = paramG['sinMuDif'][r]
        matParamG['cosMuSum'][rPair] = paramG['cosMuSum'][r]
        matParamG['sinMuSum'][rPair] = paramG['sinMuSum'][r]

    # make matrices symmetric
    matParamG['cosMuDif'] += matParamG['cosMuDif'].T
    matParamG['sinMuDif'] += matParamG['sinMuDif'].T
    matParamG['cosMuSum'] += matParamG['cosMuSum'].T
    matParamG['sinMuSum'] += matParamG['sinMuSum'].T

    return matParamG

def compPairsTestStat(phiHat, covPhiHat, pairsParamInds):
    """ Computes the test statistic for each given pair of nodes or group-nodes.
    
    Args:
        phiHat, covPhiHat are described in :meth:`funsTG.torusGraphs`,
        pairsParamInds can be either nodePairs['paramInds'] or groupNodePairs['paramInds'].
    
    Outputs: 
        tStatPairs: numpy array with either num['nodePairs'] or num['groupNodePairs'] elements. 
            The null distribution for this test statistic is chi-squared with either
            nodePairs['DoFs'] or groupNodePairs['DoFs']. 

    >>> X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
    >>> selMode = (True, True, True)
    >>> num = {'nodes': 3, 'trials': 6, 'nodePairs': 3, 'param': 18}
    >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ])}
    >>> phiHat, covPhiHat = compPhiHatAndCovPhiHat(X, num, selMode, nodePairs)
    >>> nodePairsParamInds = compNodePairsParamInds(selMode, num) 
    >>> tStatPairs = compPairsTestStat(phiHat, covPhiHat, nodePairsParamInds)
    >>> tStatPairs
    array([ 2.76461924, 24.74041927,  8.48667236])

    """
    numPairs = len(pairsParamInds)
    tStatPairs = np.zeros(numPairs)

    # Step through all pairs from pairsParamInds.
    for r in range(numPairs):
        rPhiHat = phiHat[pairsParamInds[r]]
        rCovPhiHat = covPhiHat[ pairsParamInds[r] ]\
                            [:, pairsParamInds[r] ]
        invrCovPhiHat = np.linalg.inv(rCovPhiHat)
        tStatPairs[r] = rPhiHat.T @ invrCovPhiHat @ rPhiHat

    return tStatPairs

def torusGraphs(X, edgesAlphaLevel = 0.05, selMode = (True, True, True), 
                groupLabels = (), nodeGroupLabels = (), groupEdgesAlphaLevel = 0.05):
    """Fits a torus graph to multivariate observations of angles.

    The Torus Graphs probability density function was developed by analogy
    to the multivariate Gaussian distribution, and its purpose is to model
    the dependence between multiple angle-valued variables. 

    Torus graphs is a regular full exponential family with sufficient statistics that 
    may include: 

    * 1 -- first circular moments (marginal concentrations).
    * 2 -- rotational dependence (pairwise angle differences).
    * 3 -- reflectional dependence (pairwise angle sums). 
    
    The rotational and reflectional components make the second circular moments.
    The first circular moment has 2 real-valued parameters which represent the components of a single complex number per
    random variable Xj. Similarly, the dependence between Xj and Xk has 4
    real-valued parameters which represent the complex numbers associated with 
    rotational and reflectional dependence respectively.  
    [See Theorem 4.2.1 in Klein*Orellana*]

    The Conditional independence property of torus graphs allows us to build a 
    network graph where each node represents a random variable, 
    and the absence of an edge between a pair of nodes indicates that they are 
    conditionally independent given all other random variables in the model. 
    In brief, the random variables Xj and Xk are conditionally independent 
    given all other variables if and only if the pairwise interaction terms 
    involving Xj and Xk vanish. [See Corollary 4.0.1 in Klein*Orellana*]
    Specifically, we use a chi-squared test on the parameter estimates where: 
    
    * H0 : (no-edge) conditional-independence between Xj and Xk; zero vector for the corresponding dependence parameters in the multivariate model.
    * HA : (edge) conditional-dependence between Xj and Xk.

    When group-labels are available, we can obtain a network graph where each 
    node represents a group of random variables, and edges between pairs of 
    nodes represent conditional dependence relationships. 

    Nodes (electrodes) refers to the individual sensors associated with each random 
    variables. 

    Args:
        X : numpy array (Num nodes, Num trials (repeated observations) ).
            Angle vector of repeated observations

        edgesAlphaLevel : number between 0 and 1.
            Probability of falsely placing an edge between pairs of electrode nodes.

        selMode : array_like containing 3 booleans.
            Indicates whether to fit first circular moments (selMode[0]), 
            pairwise angle differences (selMode[1]), and pairwise angle sums
            (selMode[2]). 

        groupLabels : List or Tuple with num['groupNodes'] elements. 
            Indexed array of group labels.
            For example:

            ``groupLabels = [0, 1, 2]``

            ``groupLabels = ['A', 'B', 'C']``

        nodeGroupLabels : array_like with num['nodes'] elements.
            If available, labels denoting exclusive belonging of each electrode node to a group.
            The labels must be present in groupLabels.
            In the following two examples, the first group is composed of node index 1, 
            the second group is composed of node indices 0 and 3, 
            and the third group is composed of node indices 2 and 4. 

            ``nodeGroupLabels = [1, 0, 2, 1, 2]`` 

            ``nodeGroupLabels = ['B', 'A', 'C', 'B', 'C']`` 
        
        groupEdgesAlphaLevel : number between 0 and 1.
            Probability of falsely placing an edge between pairs of electrodes-group nodes.

    Returns:
        elecGraph : networkx object.
            Each node is an electrode, absence of an edge indicates conditional
            dependence. This is an undirected and unweighted network graph 
            (see https://networkx.org/)

        groupEleGraph : networkx object.
            When nodeGroupLabels is provided, this is like elecGraph but
            where each node represents a group of electrodes. 

        num : dictionary of ints with number properties.
            * nodes:    number of electrode nodes (dimensionality).
            * trials:   number of repeated observations.
            * nodePairs:    number of electrode node pairs.
            * param:    number of estimated parameters (dimensionality of phiHat).
            * groupNodes:   number of electrode-group nodes.
            * groupNodePairs:   number of electrode-group pairs.
    
        nodePairs : dictionary of node pair properties.
            * nodes: (num['nodePairs'], 2) numpy array, each column is a (j, k) pair of nodes, see :meth:`funsTG.nodePairs`.
            * paramInds: list of lists with num['nodePairs'] elements, see :meth:`funsTG.compNodePairsParamInds`.
            * DoFs: list with num['nodePairs'] elements, degrees of freedom associated with each pair of nodes. ``[len(ind) for ind in nodePairs['paramInds']]``
            * condCoupling : numpy array with num['nodePairs'] elements, conditional coupling coefficient for model with only phase-dif statistics, i.e. selMode = (False, True, False). See section 4.4. in Klein*Orellana*.

            params, (num['nodePairs'], ) each entry has the corresponding
            estimated parameters
            pVals, (num['nodePairs'], ) edge p-values
            active, (num['nodePairs'], ) boolean, 
            True when pVal_jk <= edgesAlphaLevel
            condCoupling: (num['nodePairs'], ) when selMode = [False, True, False]
            these are the conditional coupling coefficients, which can be used
            to weigh each edge in the graph. See Section 4.4. in Klein*Orellana*

        groupNodePairs : dictionary of group-node pair properties.
            Empty tuples when nodeGroupLabels is not provided.
            
            * nodes: (num['groupNodePairs'], 2) numpy array, each column is a (j, k) pair of group-nodes, see :meth:`funsTG.compNodePairsNodes`.
            * paramInds: list of lists with num['groupNodePairs'] elements, see :meth:`funsTG.compGroupNodePairsParamInds`.
            * DoFs: list with num['groupNodePairs'] elements, degrees of freedom associated with each pair of group-nodes. ``[len(ind) for ind in groupNodePairs['paramInds']]``
            
            params, (num['groupNodePairs'], )
            pVals, (num['groupNodePairs'], ) edge p-values
            active, (num['groupNodePairs'], ) boolean, 
            True when pVal_jk <= groupEdgesAlphaLevel
    
        phiHat : numpy array with num['param'] elements.
            Model parameters estimates.
        
        covPhiHat : numpy matrix (num['param'], num['param']) 
            Covariance for the model parameter estimates.

    >>> X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
    >>> *otherOutputs, phiHat, covPhiHat = torusGraphs(X)
    >>> phiHat
    array([-39.14836993,   7.87810064,  39.42105888, -16.63875025,
            64.03174047, -47.82433269,  25.35498747, -31.41571978,
            58.81343556, -50.01232377,  -5.12549116,  11.0160441 ,
           -18.56673785,  -1.31352256,  16.1610135 , -23.23622016,
           -31.69057573, -34.39433503])
    >>> sum(covPhiHat.flatten())
    23556.69806086992

    """

    # %% num: dictionary of ints with number properties.
    num = dict()
    num['nodes'], num['trials'] = np.shape(X)
    num['nodePairs'] =  int(num['nodes']*(num['nodes']-1)/2)
    num['param'] = int( sum( np.array(selMode)*
                        np.array([ 2*num['nodes'],
                                   2*num['nodePairs'],
                                   2*num['nodePairs'] ]) ) )

    if (len(groupLabels) == 0) and (len(nodeGroupLabels)>0):
        raise ValueError('groupLabels input must be provided.')
    elif (len(groupLabels) > 0) and (len(nodeGroupLabels) == 0):
        raise ValueError('nodeGroupLabels input must be provided.')
    elif (len(groupLabels) == 0) and (len(nodeGroupLabels) == 0):
        num['groupNodes'] = 0
        num['groupNodePairs'] = 0
    else:
        num['groupNodes'] = len(set(nodeGroupLabels))
        num['groupNodePairs'] = int( num['groupNodes']*
                                    (num['groupNodes']-1)/2 )

    # print(num)

    #%% nodePairs: dictionary of node pair properties.
    nodePairs = dict()
    nodePairs['nodes'] = compNodePairsNodes(num['nodes'])
    nodePairs['paramInds'] = compNodePairsParamInds(selMode,num)
    nodePairs['DoFs'] = [len(ind) for ind in nodePairs['paramInds']]
    # print(nodePairs)

    #%% groupNodepairs: dictionary of group-node pair properties.
    groupNodePairs = dict()
    if ( num['groupNodes'] < 2 ):
        groupNodeInds = ()
        groupNodePairs['nodes'] = ()
        groupNodePairs['nodePairInds'] = ()
        groupNodePairs['paramInds'] = ()
        groupNodePairs['DoFs'] = ()
    else:
        groupNodeInds = compGroupNodeInds(num, groupLabels, nodeGroupLabels) 
        groupNodePairs['nodes'] = compNodePairsNodes(num['groupNodes'])
        groupNodePairs['nodePairInds'] \
            = compGroupNodePairsNodePairInds(num, groupLabels, \
                                    nodeGroupLabels, nodePairs, groupNodePairs)
        groupNodePairs['paramInds'] \
            = compGroupNodePairsParamInds(num, nodePairs, groupNodePairs)
        groupNodePairs['DoFs'] = [len(ind) for ind in groupNodePairs['paramInds']]

    # print(groupNodePairs)

    # %% PhiHat and its covariance
    phiHat, covPhiHat = compPhiHatAndCovPhiHat(X, num, selMode, nodePairs)

    # Conditional coupling coefficient for model with only phase-dif statistics
    if selMode == (False, True, False):
        paramG = phiToParamGroups(phiHat, num, selMode)
        condKap =  np.abs(paramG['cosMuDif'] + paramG['sinMuDif']*1j)
        nodePairs['condCoupling'] = besseli(1, condKap) / besseli(0, condKap)
    else:
        nodePairs['condCoupling'] = ()

    #%% test statistic for each pair of nodes 
    nodePairs['tStats'] = compPairsTestStat(phiHat, covPhiHat, nodePairs['paramInds'])
    nodePairs['pVals'] = chi2.sf(nodePairs['tStats'], nodePairs['DoFs'])
    nodePairs['active'] = nodePairs['nodes'][ nodePairs['pVals'] <= edgesAlphaLevel, : ]
    #%% test statistic for each group-node pair 
    if ( num['groupNodes'] < 2 ):
        groupNodePairs['tStats'] = ()
        groupNodePairs['pVals'] = ()
        allElecGroupEdges = ()
        groupNodePairs['active'] = ()
    else:
        groupNodePairs['tStats'] = \
        compPairsTestStat(phiHat, covPhiHat, groupNodePairs['paramInds'])
        groupNodePairs['pVals'] = \
            chi2.sf(groupNodePairs['tStats'],  groupNodePairs['DoFs'])
        groupNodePairs['active'] = \
            groupNodePairs['nodes'][ groupNodePairs['pVals'] <= groupEdgesAlphaLevel, :]
    
    #%% create network-graph objects
    nodeGraph = nx.Graph()
    nodeGraph.add_nodes_from(range(num['nodes']))
    nodeGraph.add_edges_from(nodePairs['active'])

    if ( num['groupNodes'] < 2 ):
        groupNodeGraph = None
    else:
        groupNodeGraph = nx.Graph()
        groupNodeGraph.add_nodes_from(range(num['groupNodes']))
        groupNodeGraph.add_edges_from(groupNodePairs['active'])

        # adding sub-region labels to node-groups in graph
        mapping = dict(zip(groupNodeGraph, groupLabels))
        groupNodeGraph = nx.relabel_nodes(groupNodeGraph, mapping)  
        

    return (nodeGraph, groupNodeGraph, num, nodePairs, groupNodePairs, phiHat, covPhiHat)

def harmonicAddition(amps, phases):
    # inputs are numpy arrays
    bx = sum(amps*np.cos(phases))
    by = sum(amps*np.sin(phases))

    resAmp = np.sqrt(bx**2 + by**2)
    resPhase = np.arctan2(by, bx)

    return (resAmp, resPhase)

def sampleTG(numSamp, phi, nodePairs, selMode = (True, True, True),\
             burnIn = 500, nThin = 100):
    """ Gibbs sampler, generates samples from a torus graph model. 
    
    Args:
        phi, selMode, and nodePairs are described in :meth:`funsTG.torusGraphs`.

        numSamp: dictionary of ints with number properties.
            * nodes: number of electrode nodes (dimensionality).

            * trials: number of repeated observations (samples).

        burnIn: positive int
            Number of samples to ignore at the beginning of the chain. 
            
        nThin: positive int, greater than 1
            Only every nThin-th sample is kept to obtain uncorrelated samples. 

    Outputs: 
        S : numpy array (numSamp['nodes'], numSamp['trials'] (repeated observations) ).
            Angle vector of repeated observations.

    >>> numSamp = {'nodes':2, 'trials': 1000}
    >>> nodePairs = {'nodes' : np.array([[0,1]])}
    >>> newPhi = np.block([ 0, 0, 0, 0, 8*np.cos(pi), 8*np.sin(pi), 0, 0 ])
    >>> Xsampled = sampleTG(numSamp, newPhi, nodePairs)
    >>> p0 = plt.plot(Xsampled[0,:], Xsampled[1,:], 'ko');
    >>> p1 = plt.xlabel('X1', fontsize=14)
    >>> p2 = plt.ylabel('X2', fontsize=14)
    >>> plt.show()

    >>> numSamp = {'nodes':3, 'trials': 1000}
    >>> nodePairs = {'nodes' : np.array([[0,1], [0,2], [1,2]])}
    >>> kap = np.array([8, 0, 8])
    >>> newPhi = np.block([ kap*np.cos(pi), kap*np.sin(pi)])
    >>> Xsampled = sampleTG(numSamp, newPhi, nodePairs, selMode = (False, True, False))
    >>> XsampledDf = pd.DataFrame(Xsampled.T, columns = ['X1', 'X2', 'X3'])
    >>> p3 = sns.pairplot(XsampledDf)

    """

    # number of electrode node pairs.
    numSamp['nodePairs'] = int(numSamp['nodes']*( numSamp['nodes'] - 1)/2)

    # dimensionality of phiHat
    numSamp['param'] = len(phi)

    totalSamples = burnIn + numSamp['trials']*nThin

    # initialize random values uniformily on circle
    x = np.random.vonmises(0, 0, numSamp['nodes'])

    # Formatting parameters
    paramG = phiToParamGroups(phi, numSamp, selMode)
    matParamG = phiParamGroupsToMats(numSamp, paramG, nodePairs)

    mu = np.angle(matParamG['cosMu'] + matParamG['sinMu']*1j)
    kap = np.abs(matParamG['cosMu'] + matParamG['sinMu']*1j)

    indsNodesAll = np.arange(numSamp['nodes'])
    indsSampKeep = list(range(burnIn,totalSamples,nThin))
    iKeep = 0

    S = np.zeros((numSamp['nodes'], numSamp['trials']))
    for i in range(totalSamples):
        for k in range(numSamp['nodes']):
            smDelta = np.concatenate(( x[:k]     - np.pi/2, 
                                       x[(k+1):] + np.pi/2 ))

            indsNoK = np.concatenate(( indsNodesAll[:k], indsNodesAll[(k+1):] ))

            amps = np.concatenate(( kap[k:(k+1)], 
                                    matParamG['cosMuDif'][k,indsNoK],
                                    matParamG['sinMuDif'][k,indsNoK],
                                    matParamG['cosMuSum'][k,indsNoK],
                                    matParamG['cosMuSum'][k,indsNoK] )) 
            phases = np.concatenate(( mu[k:(k+1)],
                                    x[indsNoK],
                                    smDelta,
                                    -x[indsNoK],
                                    -x[indsNoK] + np.pi/2 ))                                                    
            resAmp, resPhase = harmonicAddition(amps, phases)
            x[k] = np.random.vonmises(resPhase, resAmp, 1)
        if (i in indsSampKeep): 
            S[:,iKeep]=x
            iKeep += 1
    return S

def rayleigh(Y):
    """Computes Rayleigh test for non-uniformity of circular data.

    H0: the population is uniformly distributed around the circle.

    HA: the populatoin is not distributed uniformly around the circle.

    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    Reference:
        https://github.com/circstat/pycircstat

    Args:
        Y : numpy array of phase differences 
            (num r.v. pairs, num trials (repeated observations) ).

    Returns:
        pval : two-tailed p-value.

        z :    value of the z-statistic
    """
    n = Y.shape[1]
     
    # r is the magnitude of the complex mean
    r = np.abs(np.mean(np.exp(Y*1j), axis = 1))

    # compute Rayleigh's R
    # the magnitude of the complex sum
    R = n * r

    # compute Rayleigh's z 
    z = (R ** 2) / n

    # compute p value using approxation 
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    return (pval, z)

def phaseLockingValue(X, nodePairs = None):
    """ Computes the Phase Locking Value for observations from pairs of random variables. 

    Args:
        X : numpy array (Num nodes, Num trials (repeated observations) ).
            Angle vector of repeated observations
        
        nodePairs : dictionary with one key 'nodes'
            nodePairs['nodes'] is a numpy array (numNodePairs, 2)
            Each row of ints is a (j, k) index pair, with j starting at 0 
            and such that j<k<numNodes. See :meth:`funsTG.compNodePairsNodes`.

    Outputs: 
        plv: numpy array with numNodePairs elements
            When X is more than 2 rows, each entry corresponds to the given pairs in nodePairs

        plvPvals: numpy array with numNodePairs elements
            Corresponding p-values obtained using a Rayleigh test.
    
    >>> numSamp = {'nodes':2, 'trials': 1000}
    >>> nodePairs = {'nodes' : np.array([[0,1]])}
    >>> kap = 8
    >>> newPhi = np.block([ kap*np.cos(pi), kap*np.sin(pi)])
    >>> Xsampled = sampleTG(numSamp, newPhi, nodePairs, selMode = (False, True, False))
    >>> plv, pVal = phaseLockingValue(Xsampled)
    >>> bool(abs(plv - besseli(1, kap)/ besseli(0, kap)) < 1e-2)  # close to besseli(1, kap)/ besseli(0, kap)
    True
    >>> bool(pVal < 0.01)
    True

    >>> numSamp = {'nodes':3, 'trials': 1000}
    >>> nodePairs = {'nodes' : np.array([[0,1], [0,2], [1,2]])}
    >>> kap = np.array([8, 0, 8])
    >>> newPhi = np.block([ kap*np.cos(pi), kap*np.sin(pi)])
    >>> Xsampled = sampleTG(numSamp, newPhi, nodePairs, selMode = (False, True, False))
    >>> plv, pVal = phaseLockingValue(Xsampled, nodePairs)
    >>> pVal < 0.01 # PLV p-values cannot distinguish conditional independence on edge (0,2)
    array([ True,  True,  True])

    """

    if (nodePairs == None):
        if (X.shape[0]==2):
            nodePairs = {'nodes': np.array([[0,1]])}
        else:
            raise ValueError('When X has more than 2 rows, you need to include nodePairs. See :meth:`funsTG.compNodePairsNodes`.')

    Xi = X[nodePairs['nodes'][:, 0], :]
    Xj = X[nodePairs['nodes'][:, 1], :]
    Xdif = Xi - Xj

    plv = np.abs(np.mean(np.exp(1j*(Xdif)), axis =1))
    pVals, z = rayleigh(Xdif)

    return (plv, pVals)

def wrapRadAngleToPi(radAngle):
    """ Convert radians to [-pi, pi] range.

    Args:
        radAngle : array_like
            Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    
    Outputs:
        numpy array of angles in [-pi, pi] range.

    >>> abs(wrapRadAngleToPi(2*pi)) < 1e-12 
    True
    >>> sum(abs(wrapRadAngleToPi([0, pi, -3*pi/2]) - np.array([0 , pi, pi/2]) ) < 1e-12 ) 
    3
    
    """

    radAngle = np.array(radAngle)

    return np.arctan2(np.sin(radAngle), np.cos(radAngle))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
