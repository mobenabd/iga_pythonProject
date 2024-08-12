"""
NURBS_tools.py
--------------

Methods for NURBS evaluation
"""
import numpy as np


def FindSpan(n, p, u, U):
    """
    Find the knot span index for one variable u.
    ALGORITHM A2.1 (NURBS-Book)

    Input:
    n (int)     : Number of basis functions - 1.
    p (int)     : Degree of the basis functions.
    u (float)   : Evaluation point.
    U (array)   : Knot vector.

    Returns:
    int         : the knot span index.
    """
    if u == U[n+1]:
        return n

    low = p
    high = n+1
    mid = (low + high) // 2

    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def BasisFuns(i, u, p, U):
    """
    Compute nonvanishing B-Spline basis functions at point u.
    There are p+1 non-zero functions.
    ALGORITHM A2.2 (NURBS-Book)

    Input:
    i (int)     : Knot span index
    u (float)   : Parametric coordinate
    p (int)     : Degree of the basis functions
    U (array)   : Knot vector

    Returns:
    N (array)   : Contains the non-zero B-Spline basis functions: (i-p+j)th functions where j in [0:p]
    """
    N = np.zeros(p+1)
    N[0] = 1.0

    left = np.zeros(p+1)
    right = np.zeros(p+1)

    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        N[j] = saved

    return N


def DersBasisFuns(i, u, p, n, U):
    """
    Calculate the nonvanishing derivatives of the B-Spline basis functions up to order n.
    ALGORITHM A2.3 (NURBS-Book)

    Input:
    i (int)     : Knot span index
    u (float)   : Parametric coordinate
    p (int)     : Degree of the basis functions
    n (int)     : Order of derivative
    U (array)   : Knot vector

    Returns:
    ders (array): 2D array. ders[k][j] contains the kth non-zero derivatives of the (i-p+j)th B-Spline function where j in [0:p].
    """
    ders = np.zeros((n + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)

    ndu = np.zeros((p + 1, p + 1))
    a = np.zeros((2, p + 1))

    ndu[0, 0] = 1.0

    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r+1] + left[j-r]
            temp = ndu[r, j-1] / ndu[j, r]
            ndu[r, j] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        ndu[j, j] = saved

    ders[0, :] = ndu[:, p]

    if n == 0:
        return ders

    # Compute high order dervs
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0

        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j = s1
            s1, s2 = s2, s1

    r = p
    for k in range(1, n + 1):
        ders[k, :] *= r
        r *= (p - k)

    return ders


def NURBSDersBasis_1D(xi, p, U, weights, j_idx=None):
    """
    Return the 1D NURBS basis functions and first derivatives.
    Based on ALGORITHM A4.1 and A4.2 (NURBS-Book)

    Input:
    xi (float)      : Parametric coordinate where we want to evaluate NURBS functions 
    p (int)         : Degree of the basis functions
    U (array)       : Knot vector
    weights (array) : Vector of weights

    Returns:
    R (array)       : NURBS Basis functions
    dR (array)      : First derivatives of NURBS basis functions (w.r.t Parametric coordinates)
    """
    # Get the number of knots and basis functions
    m1 = len(U)  # =m+1
    n = m1 - 1 - p - 1

    # Adjust xi if it's very close to the last knot value
    tol = np.finfo(float).eps
    xi = xi if abs(xi - U[m1 - 1]) >= tol else U[m1 - 1] - tol

    # Find the span and compute the basis functions and derivatives
    if j_idx == None:
        j_idx = FindSpan(n, p, xi, U)

    dersN = DersBasisFuns(j_idx, xi, p, 1, U)

    # Compute the weight functions W(u) and its derivative W'(u)
    # W = 0.0
    # dW = 0.0
    # for i in range(p+1):
    #    wi = weights[j_idx-p+i]
    #    W += dersN[0][i] * wi
    #    dW += dersN[1][i] * wi

    weights_slice = weights[j_idx - p: j_idx + 1]
    W = np.dot(dersN[0, :p+1], weights_slice)
    dW = np.dot(dersN[1, :p+1], weights_slice)

    # Compute the NURBS basis functions and their derivatives
    # R = np.zeros(p+1)
    # dR = np.zeros(p+1)
    # for i in range(p+1):
    #    cst = weights[j_idx-p+i] / (W * W)
    #    dR[i] = (dersN[1][i] * W - dersN[0][i] * dW) * cst
    #    R[i] = dersN[0][i] * cst * W

    cst = weights_slice / (W * W)
    R = dersN[0, :p+1] * cst * W
    dR = (dersN[1, :p+1] * W - dersN[0, :p+1] * dW) * cst

    return R, dR


def NURBSDersBasis_2D(xi, eta, p, q, U, V, weights, spanU=None, spanV=None):
    """
    Return the 2D NURBS basis functions and first derivatives.
    Based on ALGORITHM A4.3 and A4.4 (NURBS-Book)

    Input:
    xi (float)      : 1st Parametric coordinate where we want to evaluate NURBS functions 
    eta (float)     : 2nd Parametric coordinate where we want to evaluate NURBS functions 
    p (int)         : Degree of the basis functions in xi axis
    q (int)         : Degree of the basis functions in eta axis
    U (array)       : knot vector along the xi direction.
    v (array)       : knot vector along the eta direction.
    weights (array) : Vector of weights

    Returns:
    R (array)       : NURBS Basis functions
    dRdxi (array)   : First derivatives of NURBS basis functions (w.r.t xi variable)
    dRdeta (array)  : First derivatives of NURBS basis functions (w.r.t eta variable)
    """

    tol = np.finfo(float).eps
    xi = xi if abs(xi - U[-1]) >= tol else U[-1] - tol
    eta = eta if abs(eta - V[-1]) >= tol else V[-1] - tol

    n = len(U) - p - 1
    m = len(V) - q - 1

    if spanU == None:
        spanU = FindSpan(n - 1, p, xi, U)
    if spanV == None:
        spanV = FindSpan(m - 1, q, eta, V)

    Nu = np.zeros(p + 1)
    Nv = np.zeros(q + 1)
    dersNu = np.zeros((n, p + 1))
    dersNv = np.zeros((m, q + 1))
    dersNu = DersBasisFuns(spanU, xi, p, 1, U)
    dersNv = DersBasisFuns(spanV, eta, q, 1, V)

    Nu = dersNu[0, :]
    Nv = dersNv[0, :]

    uind = spanU - p
    W = dWdxi = dWdeta = 0.0

    for l in range(q + 1):
        vind = spanV - q + l
        for i in range(p + 1):
            wil = weights[uind + i + vind * n]
            W += Nu[i] * Nv[l] * wil
            dWdxi += dersNu[1, i] * Nv[l] * wil
            dWdeta += dersNv[1, l] * Nu[i] * wil

    R = np.zeros((p + 1) * (q + 1))
    dRdxi = np.zeros((p + 1) * (q + 1))
    dRdeta = np.zeros((p + 1) * (q + 1))

    k = 0
    for l in range(q + 1):
        vind = spanV - q + l
        for i in range(p + 1):
            cst = weights[uind + i + vind * n] / (W * W)
            R[k] = Nu[i] * Nv[l] * cst * W
            dRdxi[k] = (dersNu[1, i] * Nv[l] * W - Nu[i] * Nv[l] * dWdxi) * cst
            dRdeta[k] = (dersNv[1, l] * Nu[i] * W -
                         Nu[i] * Nv[l] * dWdeta) * cst
            k += 1

    return R, dRdxi, dRdeta


def NURBS_1Deval(xi, p, U, ctrlPts, weights):
    """
    Return NURBS function and first derivative evaluated at a point.
    Based on ALGORITHM A4.1 and A4.2 (NURBS-Book)

    Input:
    xi (float)        : Parametric coordinate where we want to evaluate NURBS functions 
    p (int)           : Degree of the basis functions
    U (array)         : Knot vector
    ctrlPts (array)   : Controle variables
    weights (array)   : Vector of weights

    Returns:
    u (array)       : value of the interpolated function at xi
    du (array)      : value of the 1st deriviative of the interpolated function (w.r.t parametric coordinates)
    """

    # Get the number of knots and basis functions
    m1 = len(U)  # =m+1
    n = m1 - 1 - p - 1

    # Adjust xi if it's very close to the last knot value
    tol = np.finfo(float).eps
    if abs(xi - U[m1 - 1]) < tol:
        xi = U[m1 - 1] - tol

    # Find the span and compute the basis functions and derivatives
    j_idx = FindSpan(n, p, xi, U)
    R, dRdxi = NURBSDersBasis_1D(xi, p, U, weights, j_idx)

    ctrlPts_slice = ctrlPts[j_idx - p: j_idx + 1]
    u = np.dot(R, ctrlPts_slice)
    du = np.dot(dRdxi, ctrlPts_slice)

    return u, du


def SurfacePoint(n, p, U, m, q, V, P, u, v):
    """
    Return NURBS or B-spline surface function evaluated at a point (u,v).
    Based on ALGORITHM A3.5 and A4.3 (NURBS-Book)

    Input:
    n (int)           : Number of basis functions - 1 along the xi direction.
    p (int)           : Degree of the basis functions along the xi direction.
    U (array)         : knot vector along the xi direction.
    m (int)           : Number of basis functions - 1 along the eta direction.
    q (int)           : Degree of the basis functions along the eta direction.
    V (array)         : knot vector along the eta direction.
    ctrlPts (array)   : Control points vector (weighted for NURBS).
    u,v (float)       : Parametric coordinates

    Returns:
    S (array)         : Physical coordinates (non weighted)
    """
    tol = np.finfo(float).eps
    u = u if abs(u - U[-1]) >= tol else U[-1] - tol
    v = v if abs(v - V[-1]) >= tol else V[-1] - tol

    uspan = FindSpan(n, p, u, U)
    Nu = BasisFuns(uspan, u, p, U)

    vspan = FindSpan(m, q, v, V)
    Nv = BasisFuns(vspan, v, q, V)

    uind = uspan - p
    sdim = P.shape[1]
    S = np.zeros(sdim)
    for l in range(q + 1):
        temp = np.zeros(sdim)
        vind = vspan - q + l
        for k in range(p + 1):
            temp += Nu[k] * P[uind + k + vind * (n + 1)]
        S += Nv[l] * temp

    return S


def NURBS_2Deval(xi, eta, p, q, uKnotVec, vKnotVec, controlVbls, weights):
    """
    Return NURBS function and first derivatives evaluated at a point.
    Based on ALGORITHM A4.1 and A4.2 (NURBS-Book)

    Input:
    xi, eta (float)     : Parametric coordinates where we want to evaluate NURBS function
    p (int)             : Degree of the basis functions in xi direction
    q (int)             : Degree of the basis functions in eta direction
    uKnotVec (array)    : knot vector along the xi direction.
    vKnotVec (array)    : knot vector along the eta direction.
    controlVbls (array) : Controle variables vector
    weights (array)     : Vector of weights

    Returns:
    val (float)    : value of the interpolated function at (xi, eta)
    val_xi (float) : value of the 1st deriviative of the interpolated function (w.r.t parametric coordinates xi)
    val_eta (float): value of the 1st deriviative of the interpolated function (w.r.t parametric coordinates eta)
    """

    tol = np.finfo(float).eps
    xi = xi if abs(xi - uKnotVec[-1]) >= tol else uKnotVec[-1] - tol
    eta = eta if abs(eta - vKnotVec[-1]) >= tol else vKnotVec[-1] - tol

    n = len(uKnotVec) - p - 2  # Number of basis functions - 1 in xi direction
    m = len(vKnotVec) - q - 2  # Same for eta

    uspan = FindSpan(n, p, xi, uKnotVec)
    vspan = FindSpan(m, q, eta, vKnotVec)

    R, dRdxi, dRdeta = NURBSDersBasis_2D(xi, eta, p, q, uKnotVec,
                                         vKnotVec, weights, uspan, vspan)

    uind = uspan - p
    val = 0.
    val_xi = 0.
    val_eta = 0.

    k = 0
    for j in range(q + 1):
        vind = vspan - q + j
        for i in range(p + 1):
            c = uind + i + vind * (n + 1)
            val += R[k] * controlVbls[c]
            val_xi += dRdxi[k] * controlVbls[c]
            val_eta += dRdeta[k] * controlVbls[c]
            k += 1

    return val, val_xi, val_eta


def computeWeighted(control_points, weights):
    """
    Return weighted control points

    Input:
    control_points (array): Control points vector [Pi,x ; Pi,y].
    weights (array): Corresponding vector of weights [wi].

    Returns: 
    weightedPts (array): weighted control points array [wi Pi,x ; wi Pi,y; wi]
    """
    weightedPts = control_points * weights[:, np.newaxis]
    weightedPts = np.hstack((weightedPts, weights[:, np.newaxis]))
    return weightedPts


def computeNonWeighted(weightedPts):
    """
    Return non-weighted control points and corresponding weights

    Input:
    weightedPts (array): weighted control points array [wi Pi,x ; wi Pi,y; wi]

    Returns: 
    control_points (array): Control points vector [Pi,x ; Pi,y].
    weights (array): Corresponding vector of weights [wi].
    """

    dim = weightedPts.shape[1]  # (It should be 3 !!)
    weights = weightedPts[:, dim - 1]
    control_points = weightedPts[:, :dim - 1]

    # for i in range(weights.size):
    #    control_points[i, :] *= 1 / weights[i]

    control_points = control_points / weights[:, np.newaxis]

    return control_points, weights
