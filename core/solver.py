"""
solve.py
--------------

System assembly tools
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from core.quadratureRules import getWeightsNodes
from core.NURBS_tools import NURBSDersBasis_1D, NURBSDersBasis_2D
from core.mesh2D import buildElmtConnMatrix_2D, plot_2Dsol
from core.mesh1D import buildElmtConnMatrix, plot_1Dsol
from abc import ABCMeta, abstractmethod


class IGASOLVER(metaclass=ABCMeta):
    """
    Abstract class for the IgA solver. 

    Attributes:
    geo (class GEO)         : Geometry data
    sourceFunc (function)   : Source term function f(x)
    KnotIntervals_u (array) : Element range in the 1d parametric mesh [xi_i, xi_i+1]
        = KnotIntervals
    ConnMatrix  (array)     : Connectivity matrix
    nElemts (int)           : Global number of knot mesh elements
    nCtrVrb (int)           : Number of control variables (unknows)
    K (sparse array)        : Stiffness matrix
    U (array)               : Unknowns vector 
    F (array)               : Rhs vector
    -- In addition for dim = 2 --
    KnotIntervals_v (array) : Element range in the 1d parametric mesh [eta_j, eta_j+1]
    elt_idx (array)         : Knot indices (i,j) for the element [xi_i, xi_i+1] * [eta_j, eta_j+1]
    """

    def __init__(self, geo, sourceFunc):
        self.geo = geo
        self.sourceFunc = sourceFunc

    @property
    @abstractmethod
    def assemble(self):
        """Assemble the global stiffness matrix and rhs vector."""
        pass

    @abstractmethod
    def apply_boundary_conditions(self, bc_func=None):
        """Apply homogeneous Dirichlet boundary conditions."""
        pass

    @property
    def solve(self):
        """Solve the system of equations."""
        self.U = spla.spsolve(self.K.tocsr(), self.F)

    @abstractmethod
    def plot_solution(self, uexct=None):
        """Plot the solution."""
        pass


class IGA1D(IGASOLVER):
    """ IgA solver for 1d problems"""

    def __init__(self, geo, sourceFunc):
        super().__init__(geo, sourceFunc)

        self.KnotIntervals, self.ConnMatrix = buildElmtConnMatrix(
            geo.p, geo.uKnotVec)

        # Initialization
        self.nElemts = self.ConnMatrix.shape[0]  # Number of kont mesh elements
        # Number of control variables (unknows)
        self.nCtrVrb = geo.ctrlPts.shape[0]

        # Global stiffness matrix
        self.K = sp.lil_matrix((self.nCtrVrb, self.nCtrVrb))
        self.U = np.zeros(self.nCtrVrb)  # unknown vector
        self.F = np.zeros(self.nCtrVrb)  # rhs vector

    @property
    def assemble(self):
        # Assemling matrix
        QuadNodes, QuadWeights = getWeightsNodes(self.geo.quadOrdr, dim=1)
        for elm in range(self.nElemts):
            xiE = self.KnotIntervals[elm, :]  # [xi_i, xi_i+1]

            connE = self.ConnMatrix[elm, :]
            local_cpts = self.geo.ctrlPts[connE, :]

            # Loop over quadrature points
            for qp in range(len(QuadWeights)):
                s = QuadNodes[qp]
                w = QuadWeights[qp]

                # map Gauss node from parent (integration) sapce to parameteric space
                xi = 0.5 * ((xiE[1] - xiE[0]) * s + xiE[1] + xiE[0])
                J_int = 0.5 * (xiE[1] - xiE[0])    # the Jacobian determinant

                R, dRdxi = NURBSDersBasis_1D(
                    xi, self.geo.p, self.geo.uKnotVec, self.geo.weights)

                # Compute the Jacobian of the mapping (parameteric to physical domain) and the derivative w.r.t physical coordinates
                jacobian_xi = dRdxi @ local_cpts
                J = np.abs(jacobian_xi[0])  # We have only 1 spatial direction
                # (deriviative w.r.t physical coordinates)
                dRdx = (1 / J) * dRdxi

                # Compute the elementary matrix and assemble it to the global stiffness matrix
                self.K[np.ix_(connE, connE)] += np.outer(dRdx,
                                                         dRdx) * J * J_int * w

                # Compute elementary rhs vector and assemble it
                X = R @ local_cpts
                fX = self.sourceFunc(X[0])
                self.F[connE] += fX * R * J * J_int * w

    def apply_boundary_conditions(self, bc_func):
        # a weight constant to improve the conditioning: trace(K)/N
        alpha = np.mean(self.K.diagonal())
        # Global indices of the Dirichlet control variables
        bc_idx = [0, self.nCtrVrb - 1]
        bc_values = bc_func(np.array([0, 1]))

        # Modify the rhs vector
        self.F = self.F - self.K[:, bc_idx] @ bc_values
        self.F[bc_idx] = alpha * bc_values

        # Adapting the matrix
        # modifying rows and columns of the matrix to zero
        self.K[bc_idx, :] = 0
        self.K[:, bc_idx] = 0
        # Put alpha on the diagonal of the matrix corresponding to the Dirichlet control variables
        self.K[bc_idx, bc_idx] = alpha

    def plot_solution(self, uexct):
        plot_1Dsol(self.geo.p, self.geo.uKnotVec, self.geo.ctrlPts, self.geo.weights,
                   self.U, self.geo.dataType, uexct)


class IGA2D(IGASOLVER):
    """ IgA solver for 2d problems"""

    def __init__(self, geo, sourceFunc):
        super().__init__(geo, sourceFunc)

        self.KnotIntervals_u, self.KnotIntervals_v, \
            self.elt_idx, self.ConnMatrix = buildElmtConnMatrix_2D(
                geo.p, geo.q, geo.uKnotVec, geo.vKnotVec)

        # Initialization
        self.nElemts = self.ConnMatrix.shape[0]   # Global number of elements
        # Number of control variables (unknows)
        self.nCtrVrb = geo.ctrlPts.shape[0]

        # Initialize stiffness matrix and rhs vector
        # Global stiffness matrix
        self.K = sp.lil_matrix((self.nCtrVrb, self.nCtrVrb))
        self.U = np.zeros(self.nCtrVrb)  # unknowns vector
        self.F = np.zeros(self.nCtrVrb)  # rhs vector

    @property
    def assemble(self):
       # Assemling matrix
        QuadNodes, QuadWeights = getWeightsNodes(self.geo.quadOrdr, dim=2)

        for elm in range(self.nElemts):
            # [xi_i, xi_i+1]
            xiE = self.KnotIntervals_u[self.elt_idx[elm, 0], :]
            # [eta_j, eta_j+1]
            etaE = self.KnotIntervals_v[self.elt_idx[elm, 1], :]

            connE = self.ConnMatrix[elm, :]
            local_cpts = self.geo.ctrlPts[connE, :]

            # Loop over quadrature points
            for qp in range(len(QuadWeights)):
                s = QuadNodes[qp, :]
                w = QuadWeights[qp]

                # map Gauss node from parent (integration) sapce to parameteric space
                xi = 0.5 * ((xiE[1] - xiE[0]) * s[0] + xiE[1] + xiE[0])
                eta = 0.5 * ((etaE[1] - etaE[0]) * s[1] + etaE[1] + etaE[0])

                # the Jacobian determinant
                J_int = 1./4 * (xiE[1] - xiE[0]) * (etaE[1] - etaE[0])

                R, dRdxi, dRdeta = NURBSDersBasis_2D(
                    xi, eta, self.geo.p, self.geo.q, self.geo.uKnotVec, self.geo.vKnotVec, self.geo.weights)

                # Compute the Jacobian of the mapping (parameteric to physical domain) and the derivative w.r.t physical coordinates
                dRdxi_dRdeta = np.vstack((dRdxi, dRdeta))
                jacobian_xi_eta = (dRdxi_dRdeta @ local_cpts).T

                J = np.linalg.det(jacobian_xi_eta)

                # inverse of the Jacobian matrix
                jacobian_inv = np.linalg.inv(jacobian_xi_eta)

                # (deriviative w.r.t physical coordinates)
                dRdx = (jacobian_inv.T @ dRdxi_dRdeta).T
                # Compute the elementary matrix and assemble it to the global stiffness matrix
                self.K[np.ix_(connE, connE)] += (np.outer(dRdx[:, 0], dRdx[:, 0]) +
                                                 np.outer(dRdx[:, 1], dRdx[:, 1])) * J * J_int * w

                # Compute elementary rhs vector and assemble it
                X = R @ local_cpts
                fx = self.sourceFunc(X[0], X[1])
                self.F[connE] += fx * R * J * J_int * w

    def apply_boundary_conditions(self):
        tol = 1e-6
        # a weight constant to improve the conditioning: trace(K)/N
        alpha = np.mean(self.K.diagonal())

        # Global indices of the Dirichlet control variables
        if 'square' in self.geo.shape:
            x_min, x_max = np.min(self.geo.ctrlPts[:, 0]), np.max(
                self.geo.ctrlPts[:, 0])
            y_min, y_max = np.min(self.geo.ctrlPts[:, 1]), np.max(
                self.geo.ctrlPts[:, 1])
            # Find indices of control points on the boundary (within a tolerance)
            bc_idx_u = np.where((np.abs(
                self.geo.ctrlPts[:, 0] - x_min) < tol) | (np.abs(self.geo.ctrlPts[:, 0] - x_max) < tol))[0]
            bc_idx_v = np.where((np.abs(
                self.geo.ctrlPts[:, 1] - y_min) < tol) | (np.abs(self.geo.ctrlPts[:, 1] - y_max) < tol))[0]
            bc_idx = np.concatenate(bc_idx_u, bc_idx_v)

        elif 'circle' in self.geo.shape:
            x_center = 0.0
            y_center = 0.0
            R = 0.5  # np.sqrt(2)/2

            # Calculate the distance of each control point from the center
            distances = np.sqrt(
                (self.geo.ctrlPts[:, 0] - x_center)**2 + (self.geo.ctrlPts[:, 1] - y_center)**2)

            # Find indices of control points on the boundary (within a tolerance)
            bc_idx = np.where(R - tol < distances)[0]

        elif 'L-shape' in self.geo.shape:
            # Identify boundary control points
            boundary_indices = []
            for i, (x, y) in enumerate(self.geo.ctrlPts):
                if (x == -1 and -1 <= y <= 1) or (x == 1 and -1 <= y <= 0) \
                        or (y == -1 and -1 <= x <= 1) or (y == 1 and -1 <= x <= 0) \
                        or (x == 0 and 0 <= y <= 1) or (y == 0 and 0 <= x <= 1):
                    boundary_indices.append(i)
            bc_idx = np.array(boundary_indices)

        # Essential boundary conditions
        bc_values = np.zeros(bc_idx.shape)

        # Modify the rhs vector
        self.F = self.F - self.K[:, bc_idx] @ bc_values
        self.F[bc_idx] = alpha * bc_values

        # Adapting the matrix
        # modifying rows and columns of the matrix to zero
        self.K[bc_idx, :] = 0
        self.K[:, bc_idx] = 0
        # Put alpha on the diagonal of the matrix corresponding to the Dirichlet control variables
        self.K[bc_idx, bc_idx] = alpha

    def plot_solution(self):
        plot_2Dsol(self.geo.p, self.geo.q, self.geo.uKnotVec, self.geo.vKnotVec,
                   self.geo.ctrlPts, self.geo.weights, self.U, self.geo.dataType)
