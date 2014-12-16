FEneutroniCS
============

Neutronics using FEniCS.

This package aims at solving the Boltzmann Transport Equation as used in the computational neutron transport community.
Discretization is done using Finite Elements for the spatial variable, and Spherical Harmonics for the angular variable.
It is based on the FEniCS Project (http://fenicsproject.org/), a collection of free software with an extensive list of features for automated, efficient solution of differential equations.
The PETSc back-end is used for solving the resulting linear system.

The finite elements used here are the default tetrahedrons/triangular ones implemented in FEniCS.
The matrix building and boundary condition handling is performed as described in [1],
even if the (non-conforming) finite elements considered in this reference are different than the (conforming) ones used here.

A basic one-group fixed-source example is presented,
based on the Fletcher benchmark [2] benchmark,
for which an extension to 3-D is provided.
The program does not work in parallel because of the matrix kron product - a question has been asked to the FEniCS Q&A forum to solve this issue.
No multi-group nor criticality calculation example is provided for now.

References
[1] S. Van Criekingen, "A 2-D/3-D cartesian geometry non-conforming spherical harmonic neutron transport solver", Annals of Nuclear Energy, Volume 34, Issue 3, March 2007, Pages 177–187
[2] J.K. Fletcher, 'The solution of the time-independent multi-group neutron transport equation using spherical harmonics', Annals of Nuclear Energy, Vol. 4, pp. 401, 1977.

