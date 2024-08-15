from core.geo import GEO
from core.solver import IGA2D
import numpy as np


def main():
    """
    Solve the equation -Δu = f in Ω and homogeneous Dirichlet BCs using IgA.
    """

    # Source term function
    def sourceFunc(x, y): return 2 * (x + y - x**2 - y**2)
    def sourceFunc(x, y): return 2 * np.pi ** 2 * \
        np.sin(np.pi*x) * np.sin(np.pi*y)

    def sourceFunc(x, y): return np.sin(10*np.sqrt(x**2 + y**2))

    # Define the dmain shape 'unit square', 'L-shape', 'circle' etc..
    shape = 'circle'
    # Define parameterization regularity 'C0', 'C1', 'C2' etc...
    dataType = 'C1'
    # Refinement level
    refinement = 2

    geo = GEO(dataType, shape, refinement, dim=2)
    geo.plot_mesh()

    iga2d = IGA2D(geo, sourceFunc)
    iga2d.assemble
    iga2d.apply_boundary_conditions()
    iga2d.solve
    iga2d.plot_solution()


if __name__ == "__main__":
    main()
