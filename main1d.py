from core.geo import GEO
from core.solver import IGA1D
import numpy as np


def main():
    """
    Solve the equation -u'' = f in Ω = [0 1] with Dirichlet BCs using IgA.
    """

    # Exact solution
    def uexct(x): return np.cos(2 * np.pi * x)
    # Source term function f(x)
    def sourceFunc(x): return (2 * np.pi)**2 * np.cos(2 * np.pi * x)

    # Define parameterization regularity 'C0', 'C1', 'C2' etc...
    dataType = 'C1'
    # Refinement level
    refinement = 3

    geo = GEO(dataType, None, refinement, dim=1)

    iga1d = IGA1D(geo, sourceFunc)
    iga1d.assemble

    iga1d.apply_boundary_conditions(uexct)
    iga1d.solve
    iga1d.plot_solution(uexct)


if __name__ == "__main__":
    main()
