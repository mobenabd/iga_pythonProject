from core.geo import GEO
from core.solver import IGA2D_NDF
import numpy as np


def main():
    """
    Solve the NDF equation -A:H(u) = f in Ω with homogeneous Dirichlet BCs using C1 IgA.
    A is SPD, the smallest eigenvalue must be greater than 1.
    The weak formulation used is: -\int \gamma A:H(u) \Delta v  = \int \gamma f \Delta v.
    """

    # Test 1
    def A_mat(x, y): return np.array([[50., 5.],
                                     [5., 2.]])

    # Test 2 (Adapted from LAKKIS AND PRYER (2010)
    def A_mat(x, y): return np.array([[10., np.power(x**2 * y**2, 1./3)],
                                     [np.power(x**2 * y**2, 1./2),   2.]])

    # Test 3 (Discontinuous matrix, Adapted from Blechschmidt 2020)
    def A_mat(x, y): return np.array([[2, np.sign((x-0.5)*(y-0.5))],
                                     [np.sign((x-0.5)*(y-0.5)), 2]])

    print(np.linalg.eig(A_mat(0.5, 0.5)))

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

    geo = GEO(dataType, shape, refinement, dim=2)
    geo.plot_mesh()

    iga2dNDF = IGA2D_NDF(geo, sourceFunc, A_mat)
    iga2dNDF.assemble
    iga2dNDF.apply_boundary_conditions()
    iga2dNDF.solve
    iga2dNDF.plot_solution(uexct)


if __name__ == "__main__":
    main()
