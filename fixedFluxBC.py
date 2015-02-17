'''
Returns the lists of spherical harmonics to cancel in applying fixed flux BCs.
- fixedFluxVertices = numpy array with local vertices where fixed flux BC applies
- local_indices_fixed_flux1 = list() containing corresponding Y00 components
- local_indices_fixed_flux2 = list() containing other components

Author: S. Van Criekingen
'''
from dolfin import *
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

def return_local_fixedFluxBCinfo(V,numSpherHarmonics,bcsFixedFlux):
        local_indices_fixed_flux1 = list()
        local_indices_fixed_flux2 = list()
        uFixedFlux = Function(V)
        for bc in bcsFixedFlux:
                bc.apply(uFixedFlux.vector())
        d2v = V.dofmap().dofs()
        fixedFluxVertices = d2v[uFixedFlux.vector().get_local() == 1.0]
        local_indices_fixed_flux1 += (fixedFluxVertices * numSpherHarmonics).tolist()
        for ii in range(1,numSpherHarmonics):
                local_indices_fixed_flux2 += ( fixedFluxVertices * numSpherHarmonics + ii ).tolist()

        return fixedFluxVertices, local_indices_fixed_flux1, local_indices_fixed_flux2
