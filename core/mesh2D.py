"""
mesh2D.py
--------------

Tools for 2D mesh
"""
import numpy as np
from core.NURBS_tools import SurfacePoint, computeWeighted, computeNonWeighted, NURBS_2Deval
from core.mesh1D import RefineKnotVectCurve, buildElmtConnMatrix
import matplotlib.pyplot as plt


def Refine2DMesh(p, q, ctrlPts, weights, u_knot, v_knot, refinement):
    """
    Adapt initial mesh for IgA.

    Input:
    p (int)             : Degree of the basis functions along the xi direction.
    q (int)             : Degree of the basis functions along the eta direction.
    ctrlPts (array)     : Initial control points vector.
    weights (array)     : Corresponding weights.
    uknotVec (array)    : Initial open knot vector along the xi direction.
    vknotVec (array)    : Initial open knot vector along the eta direction.
    refinement (int)    : h-refinement level

    Returns:
    ctrlPts (array)         : Adated control points and wights (last column) after refinement
    knotVec (array)         : Adated knot vector after refinement
    KnotIntervals (array)   : Element range in the parametric mesh [xi_i, xi_i+1]
    ConnMatrix  (array)     : Connectivity matrix
    """

    for rr in range(refinement):
        n = len(u_knot) - p - 1  # Number of basis functions in xi direction
        m = len(v_knot) - q - 1  # Same for eta

        # Extracting unique knots
        _, idx = np.unique(u_knot, return_index=True)
        uniqueKnots_u = u_knot[np.sort(idx)]
        _, idx = np.unique(v_knot, return_index=True)
        uniqueKnots_v = v_knot[np.sort(idx)]

        # ---------- uniform h-refinement along the xi axis
        # Compute new knots
        newKnotsX_u = uniqueKnots_u[:-1] + 0.5 * np.diff(uniqueKnots_u)
        lengthX = len(newKnotsX_u)  # Count new knots
        newWeightedPts = np.zeros(((n + lengthX) * m, 3))

        idxBeg_old = 0  # Index of the first element in ctrlPts for a row in eta direction
        idxBeg_new = 0  # Index of the first element in newWeightedPts for a row in eta direction

        for k in range(m):
            idxEnd_old = idxBeg_old + n  # Index of the last element
            idxEnd_new = idxBeg_new + n + lengthX
            wightedPts_slice = computeWeighted(ctrlPts[idxBeg_old:idxEnd_old, :],
                                               weights[idxBeg_old:idxEnd_old])

            # TO DO: ewKnotVec is repeated each time, to be improved !
            newKnotVec, newControlPts = RefineKnotVectCurve(n - 1, p,
                                                            u_knot, wightedPts_slice, newKnotsX_u, lengthX - 1)
            newWeightedPts[idxBeg_new:idxEnd_new, :] = newControlPts

            idxBeg_new = idxEnd_new
            idxBeg_old = idxEnd_old

        u_knot = newKnotVec
        ctrlPts, weights = computeNonWeighted(newWeightedPts)
        n += lengthX

        # ---------- uniform h-refinement along the eta axis
        # Compute new knots
        newKnotsX_v = uniqueKnots_v[:-1] + 0.5 * np.diff(uniqueKnots_v)
        lengthX = len(newKnotsX_v)  # Count new knots
        newWeightedPts = np.zeros(((m + lengthX) * n, 3))

        for i in range(n):
            # Indices for a column in the xi direction
            idx_old = np.arange(i, n * m, n)
            wightedPts_slice = computeWeighted(ctrlPts[idx_old, :],
                                               weights[idx_old])

            # TO DO: newKnotVec is repeated each time, to be improved !
            newKnotVec, newControlPts = RefineKnotVectCurve(m - 1, q,
                                                            v_knot, wightedPts_slice, newKnotsX_v, lengthX - 1)

            idx_new = np.arange(i, n * (m + lengthX), n)
            newWeightedPts[idx_new, :] = newControlPts

        v_knot = newKnotVec
        ctrlPts, weights = computeNonWeighted(newWeightedPts)
        m += lengthX

    return ctrlPts, weights, u_knot, v_knot


def plot_2Dmesh(ctrlPts, weights, u_knot, v_knot, p, q, plotMeshOnly=False):
    """
    Plot the image of the knots mesh in the physical domain and the control mesh.

    Input:
    ctrlPts (array)     : Control points vector.
    weights (array)     : Vector of weights.
    uknotVec (array)    : Open knot vector along the xi direction.
    vknotVec (array)    : Open knot vector along the eta direction.
    p (int)             : Degree of the basis functions along the xi direction.
    q (int)             : Degree of the basis functions along the eta direction.
    plotMeshOnly (bool) : Defines whether we want to plot the mesh only, or whether 
                            it will be plotted along another figure.

    Returns:
    NONE

    """
    n = len(u_knot) - p - 1  # Number of basis functions in xi direction
    m = len(v_knot) - q - 1  # Same for eta

    # Extracting unique knots
    _, idx = np.unique(u_knot, return_index=True)
    uniqueKnots_u = u_knot[np.sort(idx)]
    _, idx = np.unique(v_knot, return_index=True)
    uniqueKnots_v = v_knot[np.sort(idx)]

    # lenght of unique knots
    ne_u = len(uniqueKnots_u)
    ne_v = len(uniqueKnots_v)

    # Samling in the xi and eta directions
    n_sampling = 100
    xi_sampling = np.linspace(u_knot[0], u_knot[-1], n_sampling)
    eta_sampling = np.linspace(v_knot[0], v_knot[-1], n_sampling)

    # Compute the knot mesh lines in the xi direction
    weightedPts = computeWeighted(ctrlPts, weights)
    x_xi = np.zeros((n_sampling, ne_u))
    y_xi = np.zeros((n_sampling, ne_u))
    for ii in range(ne_u):
        xi = uniqueKnots_u[ii]
        for i in range(n_sampling):
            eta = eta_sampling[i]
            X = SurfacePoint(n - 1, p, u_knot,
                             m - 1, q, v_knot, weightedPts, xi, eta)
            x_xi[i, ii] = X[0] / X[2]
            y_xi[i, ii] = X[1] / X[2]

    # Compute  the knot mesh lines in the eta direction
    x_eta = np.zeros((n_sampling, ne_v))
    y_eta = np.zeros((n_sampling, ne_v))
    for ii in range(ne_v):
        eta = uniqueKnots_v[ii]
        for i in range(n_sampling):
            xi = xi_sampling[i]
            X = SurfacePoint(n - 1, p, u_knot,
                             m - 1, q, v_knot, weightedPts, xi, eta)
            x_eta[i, ii] = X[0] / X[2]
            y_eta[i, ii] = X[1] / X[2]

    plt.figure()
    plt.axis('tight')
    if (plotMeshOnly):
        plt.subplot(1, 2, 1)  # plot Knot mesh image
        plt.title('Physical knot mesh')
        plt.plot(x_xi, y_xi, 'k-', linewidth=0.8)
        plt.plot(x_eta, y_eta, 'k-', linewidth=0.8)
        plt.subplot(1, 2, 2)  # plot connected control points
        plt.title('Control point grid')
        # plot vertical lines of the control point grid
        for i in range(m):
            k = np.arange(i * n, (i + 1) * n)
            plt.plot(ctrlPts[k, 0], ctrlPts[k, 1], 'ro-',
                     markersize=4, linewidth=0.8, markerfacecolor='k', markeredgecolor='k')
        # plot horizontal lines of the control point grid
        for j in range(n):
            k = np.arange(j, m * n, n)
            plt.plot(ctrlPts[k, 0], ctrlPts[k, 1], 'ro-',
                     markersize=4, linewidth=0.8, markerfacecolor='k', markeredgecolor='k')

        plt.show()
        # plt.show(block=False)
    else:
        # plot Knot mesh image
        plt.plot(x_xi, y_xi, 'k-', linewidth=0.8)
        plt.plot(x_eta, y_eta, 'k-', linewidth=0.8)


def buildElmtConnMatrix_2D(p, q, uKnotVec, vKnotVec):
    """
    Build the 2d connectivity matrix and element ranges in the parametric knot mesh.

    Input:
    p (int)             : Degree of the basis functions along the xi direction.
    q (int)             : Degree of the basis functions along the eta direction.
    uknotVec (array)    : Open knot vector along the xi direction.
    vknotVec (array)    : Open knot vector along the eta direction.

    Returns:
    KnotIntervals_u (array) : Element range in the 1d parametric mesh [xi_i, xi_i+1]
    KnotIntervals_v (array) : Element range in the 1d parametric mesh [eta_j, eta_j+1]
    elt_idx (array)         : Knot indices (i,j) for the element [xi_i, xi_i+1] * [eta_j, eta_j+1]
    ConnMatrix  (array)     : Connectivity matrix
    """

    # Build 1d connectivity matrices in xi and eta directions
    KnotIntervals_u, ConnMatrix_u = buildElmtConnMatrix(p, uKnotVec)
    KnotIntervals_v, ConnMatrix_v = buildElmtConnMatrix(q, vKnotVec)

    # Number of knot mesh elements in xi axis
    nElemts_u = ConnMatrix_u.shape[0]
    # Number of knot mesh elements in eta axis
    nElemts_v = ConnMatrix_v.shape[0]
    nElemts = nElemts_v * nElemts_u     # Number of knot mesh elements

    # Knot indices that determine the element [xi_i, xi_i+1] * [eta_j, eta_j+1]
    elt_idx = np.zeros((nElemts, 2), dtype=int)
    for j in range(nElemts_v):
        for i in range(nElemts_u):
            elt_idx[j * nElemts_u + i, 0] = i
            elt_idx[j * nElemts_u + i, 1] = j

    # Build 2d connectivity matrix
    n = len(uKnotVec) - p - 1  # Number of basis functions in xi direction
    ConnMatrix = np.zeros((nElemts, (p+1)*(q+1)), dtype=int)
    e = 0
    for elm_v in range(nElemts_v):
        connE_v = ConnMatrix_v[elm_v, :]
        for elm_u in range(nElemts_u):
            connE_u = ConnMatrix_u[elm_u, :]
            for i in range(q+1):
                for j in range(p+1):
                    ConnMatrix[e, i * (p+1) + j] = connE_u[j] + connE_v[i] * n
            e += 1

    return KnotIntervals_u, KnotIntervals_v, elt_idx, ConnMatrix


def plot_2Dsol(p, q, uKnotVec, vKnotVec, ctrlPts, weights, controlVbls, dataType):
    """
    Plot the solution using computed controle variables.

    Input:
    p (int)             : Degree of the basis functions along the xi direction.
    q (int)             : Degree of the basis functions along the eta direction.
    uknotVec (array)    : Knot vector along the xi direction.
    vknotVec (array)    : Knot vector along the eta direction.
    ctrlPts (array)     : Control points vector.
    weights (array)     : Corresponding weights.
    controlVbls (array) : Control variables vector
    data type (string)  : data type (C0, C1, ...)

    Returns:
    None
    """

    n = len(uKnotVec) - p - 1  # Number of basis functions in xi direction
    m = len(vKnotVec) - q - 1  # Same for eta

    noPts = 100
    xi = np.linspace(uKnotVec[0], uKnotVec[-1], noPts)
    eta = np.linspace(vKnotVec[0], vKnotVec[-1], noPts)

    x_sampling = np.zeros((noPts, noPts))
    y_sampling = np.zeros((noPts, noPts))
    u_values = np.zeros((noPts, noPts))

    weightedPts = computeWeighted(ctrlPts, weights)

    for i in range(noPts):
        for j in range(noPts):
            # To be adapted and replaced with NURBS_2Deval ?
            X = SurfacePoint(n - 1, p, uKnotVec,
                             m - 1, q, vKnotVec, weightedPts, xi[i], eta[j])

            x_sampling[i, j] = X[0] / X[2]
            y_sampling[i, j] = X[1] / X[2]

            u_values[i, j], _, _ = NURBS_2Deval(xi[i], eta[j], p, q,
                                                uKnotVec, vKnotVec, controlVbls, weights)

    # plt.figure()
    plot_2Dmesh(ctrlPts, weights, uKnotVec, vKnotVec, p, q)
    plt.contourf(x_sampling, y_sampling, u_values)
    # ax = plt.axes(projection ='3d')
    # Creating plot
    # ax.plot_surface(x_sampling, y_sampling, u_values)
    # plt.show()
    # return

    plt.title("IgA sol with " + dataType + " geometry")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.colorbar()
    plt.show()
