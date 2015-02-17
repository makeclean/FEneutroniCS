'''
Returns the list of spherical harmonics to cancel in applying reflection BCs.
In 2D: Y00, Y20, Y21, Y22, Y40, Y41, Y42, Y43, Y44, Y60, Y61
-> to cancel: P3: +2; P5: +5, +7; P7: +10,+12,+14

Author: S. Van Criekingen
'''
from dolfin import *
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

def return_local_indices_to_cancel(V,PnOrder,numSpherHarmonics,nDim,bcsRefl):
        local_indices_to_cancel = list()
        uRefl = Function(V)
        for bc in bcsRefl:
                bc.apply(uRefl.vector())
        d2v = V.dofmap().dofs()
        reflected_vertices = d2v[uRefl.vector().get_local() == 1.0]
        ii = 1
        for nn in range (2,PnOrder,2):
           for mm in range (0,nn+1):
              if (mm%2==1):  # mm is odd
                 #print "  nn=", nn, "  mm=", mm
                 local_indices_to_cancel += ( reflected_vertices * numSpherHarmonics + ii ).tolist()
              ii +=1
           if (nDim ==3):
              for mm in range (-nn,0): # 0 is not an option
                 if (mm%2==0): # mm is even
                    #print "  nn=", nn, "  mm=", mm
                    local_indices_to_cancel += ( reflected_vertices * numSpherHarmonics + ii ).tolist()
                 ii +=1
        return local_indices_to_cancel
