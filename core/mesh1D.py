"""
mesh1D.py
--------------

Tools for 1D mesh
"""
import numpy as np
from core.NURBS_tools import FindSpan, computeWeighted, computeNonWeighted, NURBS_1Deval
import matplotlib.pyplot as plt


def RefineKnotVectCurve(n, p, U, Pw, X, r):
    """
    Insert multiple knots into a NURBS curve.
    ALGORITHM A5.4 (NURBS-Book)

    Input:
    n (int)     : Number of basis functions - 1.
    p (int)     : Degree of the basis functions.
    U (array)   : Old knot vector.
    Pw (array)  : Old control points. (wighted)
    X (array)   : Vector of new knots (multiple entries possible).
    r (int)     : Size of X - 1 (count the multiple entries as well).

    Returns:
    Ubar (array): New knot vector
    Qw (array)  : New control points (wighted)
    """

    m = n+p+1
    a = FindSpan(n, p, X[0], U)
    b = FindSpan(n, p, X[r], U)
    b += 1

    Qw = np.zeros((n+r+2, Pw.shape[1]))
    Ubar = np.zeros(len(U)+r+1)

    for j in range(a-p+1):
        Qw[j] = Pw[j]
    for j in range(b-1, n+1):
        Qw[j+r+1] = Pw[j]
    for j in range(a+1):
        Ubar[j] = U[j]
    for j in range(b+p, m+1):
        Ubar[j+r+1] = U[j]

    i = b+p-1
    k = b+p+r

    for j in range(r, -1, -1):
        while X[j] <= U[i] and i > a:
            Qw[k-p-1] = Pw[i-p-1]
            Ubar[k] = U[i]
            k -= 1
            i -= 1
        Qw[k-p-1] = Qw[k-p]
        for l in range(1, p+1):
            ind = k-p+l
            alfa = Ubar[k+l] - X[j]
            if np.abs(alfa) == 0.:
                Qw[ind-1] = Qw[ind]
            else:
                alfa = alfa / (Ubar[k+l] - U[i-p+l])
                Qw[ind-1] = alfa * Qw[ind-1] + (1.-alfa) * Qw[ind]
        Ubar[k] = X[j]
        k -= 1

    return Ubar, Qw


def Refine1DMesh(p, ctrlPts, weights, knotVec, refinement):
    """
    Adapt initial mesh for IgA

    Input:
    p (int)             : Degree of the basis functions.
    ctrlPts (array)     : Initial control points vector.
    weights (array)     : Corresponding weights.
    knotVec (array)     : Initial open knot vector.
    refinement (int)    : h-refinement level

    Returns:
    ctrlPts (array)      : Adapted control points and wights (last column) after refinement
    weights (array)      : Adapted wights vector after refinement
    knotVec (array)      : Adated knot vector after refinement
    """

    # ---------- uniform h-refinement
    for rr in range(refinement):
        n = len(knotVec) - 1 - p - 1
        # Extracting unique knots
        _, idx = np.unique(knotVec, return_index=True)
        uniqueKnots = knotVec[np.sort(idx)]
        # Compute new knots
        newKnotsX = uniqueKnots[:-1] + 0.5 * np.diff(uniqueKnots)

        # compute wighted control points
        weightedPts = computeWeighted(ctrlPts, weights)

        newKnotVec, newControlPts = RefineKnotVectCurve(
            n, p, knotVec, weightedPts, newKnotsX, len(newKnotsX)-1)

        ctrlPts, weights = computeNonWeighted(newControlPts)

        knotVec = newKnotVec

    return ctrlPts, weights, knotVec


def buildElmtConnMatrix(p, knotVec):
    """
    Build the connectivity matrix and element ranges in the parametric knot mesh.

    Input:
    p (int)                 : Degree of the basis functions.
    knotVec (array)         : Knot vector.

    Returns:
    KnotIntervals (array)   : Element range in the parametric mesh [xi_i, xi_i+1]
    ConnMatrix  (array)     : Connectivity matrix
    """
    # Extracting unique knots
    _, idx = np.unique(knotVec, return_index=True)
    uniqueKnots = knotVec[np.sort(idx)]

    # ---------- connectivity matrix and assembly stuff
    # Number of elements
    ne = len(uniqueKnots) - 1
    # Knot intervals (elemnts in the parametric domain)
    KnotIntervals = np.zeros((ne, 2), dtype=float)
    # Knot indices for elements
    elt_idx = np.zeros((ne, 2), dtype=int)
    # Element connectivities
    ConnMatrix = np.zeros((ne, p + 1), dtype=int)

    # Determine element ranges and corresponding knot indices
    e = 0
    previous_Knot = knotVec[0]
    for i in range(len(knotVec)):
        present_Knot = knotVec[i]
        if knotVec[i] != previous_Knot:
            elt_idx[e, :] = [i-1, i]
            KnotIntervals[e, :] = [previous_Knot, present_Knot]
            e += 1
        previous_Knot = present_Knot
    # Connectivity matrix
    for e in range(ne):
        ConnMatrix[e, :] = np.arange(
            elt_idx[e, 0]-p, elt_idx[e, 0]+1)

    return KnotIntervals, ConnMatrix


def plot_1Dsol(p, knotVec, ctrlPts, weights, controlVbls, dataType, uexct):
    """
    Plot the solution using computed controle variables.

    Input:
    p (int)             : Degree of the basis functions.
    knotVec (array)     : Knot vector.
    ctrlPts (array)     : control points vector.
    weights (array)     : Corresponding weights.
    controlVbls (array) : Control variables vector
    data type (string)  : data type (C0, C1, ...)

    Returns:
    None
    """
    noPts = 30
    xi = np.linspace(knotVec[0], knotVec[-1], noPts)
    x_sampling = np.zeros(noPts)
    u_values = np.zeros(noPts)

    for i in range(noPts):
        x_sampling[i], _ = NURBS_1Deval(
            xi[i], p, knotVec, ctrlPts[:, 0], weights)

        u_values[i], _ = NURBS_1Deval(
            xi[i], p, knotVec, controlVbls, weights)

    x = np.linspace(0, 1, 100)
    plt.plot(x_sampling, u_values, 'kx',
             label='IgA sol with ' + dataType + " geometry")
    plt.plot(x, uexct(x), 'r-', label='Exact sol')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.axis('tight')
    plt.show()
