## About
Implementation of IsoGeometric Analysis (IGA) for elliptic scalar problems on arbitrary geometries in 1D and 2D.

Refinement is restricted to h-refinement (node insertion).

Boundary conditions are restricted to homogeneous Dirichlet and applied by directly adapting the stiffness matrix and right-hand side vector.


This project is inspired by :

[1] Piegl, Les, and Wayne Tiller. The NURBS book. Springer Science & Business Media, 2012.

[2] Vuong, A-V., Ch Heinrich, and Bernd Simeon. "ISOGAT: A 2D tutorial MATLAB code for Isogeometric Analysis." Computer Aided Geometric Design 27.8 (2010): 644-655.

[3] The doctoral course [Advanced Scientific Programming in Python](https://github.com/JochenHinz/python_seminar) - Jochen Hinz.




## Quick start
Clone the repository and run the script
```[bash]
git clone https://github.com/mobenabd/iga_pythonProject.git
cd iga_pythonProject
main2D.py
```

To generate Doxgen documentation, run :
```[bash]
doxygen
```


## File structure

```
iga_project
|   LICENSE
│   README.md
|   main1D.py                  (solve 1D problems)
|   main2D.py                  (solve 2D problems)
|
└───core                        (IGA solver core scripts)
|   |   ...
└───test                         
|   |   ...
```


