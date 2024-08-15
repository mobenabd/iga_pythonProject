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


#Test NDF scheme when A is identity
# Test 3 (Discontinuous matrix, Adapted from Blechschmidt 2020)
def A_mat(x, y): return np.identity(2)

# Exact solution
def uexct(x, y): return np.sin(np.pi*x) * np.sin(np.pi*y)

# Source term function
def sourceFunc(x, y):
    A = A_mat(x, y)
    return - np.pi**2 * (- A[0, 0] * np.sin(np.pi*x) * np.sin(np.pi*y)
                         + 2 * A[0, 1] * np.cos(np.pi*x) * np.cos(np.pi*y)
                         - A[1, 1] * np.sin(np.pi*x) * np.sin(np.pi*y))

# Define the dmain shape 'unit square', 'L-shape', 'circle' etc..
shape = 'unit square'
# The required regularity of parameterization is at least 'C1' !!
dataType = 'C1'
# Refinement level
refinement = 3

from core.geo import GEO
from core.solver import IGA2D_NDF

geo = GEO(dataType, shape, refinement, dim=2)
iga2dNDF = IGA2D_NDF(geo, sourceFunc, A_mat)
iga2dNDF.assemble
iga2dNDF.apply_boundary_conditions()
iga2dNDF.solve
iga2dNDF.plot_solution(uexct)
