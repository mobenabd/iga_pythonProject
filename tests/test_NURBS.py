"""
test_NURBS.py
-------------

"""
import __context__
import numpy as np
from core.geo import Import2DGeoData
from core.NURBS_tools import *
from core.mesh1D import *
from core.quadratureRules import getWeightsNodes
from core.mesh2D import *


# test FindSpan
n = 3
p = 2
u = 1.5
U = np.array([0, 0, 0, 1, 2, 2, 2])  # len(KnotVec)-1 = m = n+p+1

knotSpanIndex = FindSpan(n, p, u, U)
print("Knot Span Index:", knotSpanIndex)


# test RefineKnotVectCurve
n = 3
p = 2
U = np.array([0, 0, 0, 1, 2, 2, 2])
Pw = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1]])
X = np.array([0., 1.5])
r = len(X) - 1

Ubar, Qw = RefineKnotVectCurve(n, p, U, Pw, X, r)
print("Ubar:", Ubar)
print("Qw:", Qw)

# test quadrature
dim = 2
order = 1
Nodes, Weights = getWeightsNodes(order, dim)
print("Node:\n", Nodes)
print("Weights:\n", Weights)


# Test BasisFuns
U = np.array([0, 0, 0, 1, 2, 3, 4, 4, 4])
u = 1.5
p = 3
i = FindSpan(n, p, u, U)
print("Basis values:\n", BasisFuns(i, u, p, U))


order = 2
ders = DersBasisFuns(i, u, p, order, U)
print("Derivatives:\n", ders)


# Test 2D mesh
shape = 'L-shape'
dataType = 'C1'
refinement = 2
uKnotVec, vKnotVec, ctrlPts, p, q, quadOrdr, weights = Import2DGeoData(
    dataType, shape)

plot_2Dmesh(ctrlPts, weights, uKnotVec, vKnotVec, p, q, plotMeshOnly=True)

if (refinement > 0):
    ctrlPts, weights, uKnotVec, vKnotVec = Refine2DMesh(
        p, q, ctrlPts, weights, uKnotVec, vKnotVec, refinement)

plot_2Dmesh(ctrlPts, weights, uKnotVec, vKnotVec, p, q, plotMeshOnly=True)


shape = 'circle'
dataType = '-'
refinement = 2
uKnotVec, vKnotVec, ctrlPts, p, q, quadOrdr, weights = Import2DGeoData(
    dataType, shape)

if (refinement > 0):
    ctrlPts, weights, uKnotVec, vKnotVec = Refine2DMesh(
        p, q, ctrlPts, weights, uKnotVec, vKnotVec, refinement)
plot_2Dmesh(ctrlPts, weights, uKnotVec, vKnotVec, p, q, plotMeshOnly=True)
