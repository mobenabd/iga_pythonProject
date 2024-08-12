"""
geo.py
--------------

Geometry data.
"""
import numpy as np
from core.mesh2D import Refine2DMesh, plot_2Dmesh
from core.mesh1D import Refine1DMesh


class GEO:
    """
    Define the geometry data. 

    Attributes:
    data type (string)  : data type (C0, C1, ...)
    shape (string)      : physical domain shape: unite square or L-shape
    dim (int)           : number of spatial directions (1d or 2d)
    ctrlPts (array)     : Control points vector.
    p (int)             : Degree of the basis functions along the xi direction.
    uknotVec (array)    : Open knot vector along the xi direction.
    quadOrdr (int)      : Quadrature order.
    weights (array)     : Vector of weights.
    -- In addition for dim = 2 --
    vknotVec (array)    : Initial open knot vector along the eta direction.
    q (int)             : Degree of the basis functions along the eta direction.
    """

    def __init__(self, dataType, shape, refinement, dim):
        """Initialize GEO and refine the geometry."""
        self.dataType = dataType
        self.shape = shape
        self.dim = dim

        if dim == 1:
            self.uKnotVec, self.ctrlPts, self.p, self.quadOrdr, \
                self.weights = Import1DGeoData(dataType)
            if (refinement > 0):
                self.ctrlPts, self.weights, self.uKnotVec = Refine1DMesh(
                    self.p, self.ctrlPts, self.weights, self.uKnotVec, refinement)
        elif dim == 2:
            self.uKnotVec, self.vKnotVec, self.ctrlPts, self.p, self.q, \
                self.quadOrdr, self.weights = Import2DGeoData(dataType, shape)
            if (refinement > 0):
                self.ctrlPts, self.weights, self.uKnotVec, \
                    self.vKnotVec = Refine2DMesh(self.p, self.q, self.ctrlPts,
                                                 self.weights, self.uKnotVec, self.vKnotVec, refinement)

    def plot_mesh(self):
        """Plot the knot mesh and control point grid."""
        if self.dim == 1:
            raise Exception("Sorry, no 1d mesh plotting available.")
        elif self.dim == 2:
            plot_2Dmesh(self.ctrlPts, self.weights, self.uKnotVec,
                        self.vKnotVec, self.p, self.q, plotMeshOnly=True)


def Import1DGeoData(data):
    """
    Define 1d geometry data. The physical space is [0 1]

    Input:
    data type (string)  : data type (C0, C1, ...)

    Returns:
    knotVec (array)     : Initial open knot vector.
    ctrlPts (array)     : Control points vector.
    p (int)             : Degree of the basis functions.
    quadOrdr (int)      : Quadrature order.
    weights (array)     : Vector of weights.
    """

    if data == 'C0':  # Equivalent to FEM
        knotVec = np.array([0, 0, 1, 1], dtype=float)
        ctrlPts = np.array([[0, 0], [1, 0]], dtype=float)
        p = 1
        quadOrdr = 2
    elif data == 'C1':
        knotVec = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        ctrlPts = np.array([[0, 0], [0.5, 0], [1, 0]], dtype=float)
        p = 2
        quadOrdr = 3
    elif data == 'C2':
        knotVec = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        ctrlPts = np.array(
            [[0, 0], [0.3, 0], [0.5, 0], [1, 0]], dtype=float)
        p = 3
        quadOrdr = 4
    else:
        raise Exception("Sorry, this geometry data is not available")

    knotVec = knotVec/np.max(knotVec)
    weights = np.ones(ctrlPts.shape[0], dtype=float)

    return knotVec, ctrlPts, p, quadOrdr, weights


def Import2DGeoData(data, shape):
    """
    Define 2d geometry data. 

    Input:
    data type (string)  : data type (C0, C1, ...)
    shape (string)      : physical domain shape: unite square or L-shape

    Returns:
    uknotVec (array)    : Initial open knot vector along the xi direction.
    vknotVec (array)    : Initial open knot vector along the eta direction.
    ctrlPts (array)     : Control points vector.
    p (int)             : Degree of the basis functions along the xi direction.
    q (int)             : Degree of the basis functions along the eta direction.
    quadOrdr (int)      : Quadrature order.
    weights (array)     : Vector of weights.
    """

    if shape == 'unit square':
        if data == 'C0':
            pass
        elif data == 'C1':
            p = 2
            q = 2
            uKnotVec = np.array([0, 0, 0, 1, 1, 1])
            vKnotVec = np.array([0, 0, 0, 1, 1, 1])
            ctrlPts = np.array([[0, 0], [0.5, 0], [1, 0],
                                [0, 0.5], [0.5, 0.5], [1, 0.5],
                                [0, 1], [0.5, 1], [1, 1]])
            weights = np.ones(ctrlPts.shape[0], dtype=float)
            quadOrdr = 3
    elif shape == 'L-shape':
        if data == 'C0':
            p = 1
            q = 1
            uKnotVec = np.array([0, 0, 0.5, 1, 1])
            vKnotVec = np.array([0, 0, 1, 1])
            ctrlPts = np.array([[-1, 1], [-1, -1], [1, -1],
                                [0, 1], [0, 0], [1, 0]])
            weights = np.ones(ctrlPts.shape[0], dtype=float)
            quadOrdr = 2
        elif data == 'C1':
            p = 2
            q = 2
            uKnotVec = np.array([0, 0, 0, 0.5, 1, 1, 1])
            vKnotVec = np.array([0, 0, 0, 1, 1, 1])
            ctrlPts = np.array([[-1, 1], [-1, -1], [-1, -1], [1, -1],
                                [-0.65, 1], [-0.7, 0], [0, -0.7], [1, -0.65],
                                [0, 1], [0, 0], [0, 0], [1, 0]])
            weights = np.ones(ctrlPts.shape[0], dtype=float)
            quadOrdr = 3
    elif shape == 'circle':
        p = 2
        q = 2
        uKnotVec = np.array([0, 0, 0, 1, 1, 1])
        vKnotVec = np.array([0, 0, 0, 1, 1, 1])
        ctrlPts = np.array([
            [-np.sqrt(2)/4, np.sqrt(2)/4], [-np.sqrt(2)/2,
                                            0], [-np.sqrt(2)/4, -np.sqrt(2)/4],
            [0, np.sqrt(2)/2], [0, 0], [0, -np.sqrt(2) /
                                        2], [np.sqrt(2)/4, np.sqrt(2)/4], [np.sqrt(2)/2, 0],
            [np.sqrt(2)/4, -np.sqrt(2)/4]])
        weights = np.array([1, np.sqrt(2)/2, 1, np.sqrt(2)/2,
                           1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1])
        quadOrdr = 3
    else:
        raise Exception("Sorry, this geometry data is not available")

    uKnotVec = uKnotVec/np.max(uKnotVec)
    vKnotVec = vKnotVec/np.max(vKnotVec)

    return uKnotVec, vKnotVec, ctrlPts, p, q, quadOrdr, weights
