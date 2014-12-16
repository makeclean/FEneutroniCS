'''
Performs Kronecker products (i.e., tensor products) needed for PN discretization when N>1

Author: S. Van Criekingen
'''
from dolfin import *
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Must be declared before call:   toReturn = PETScVector()
# type(V1) : dolfin.cpp.la.Vector
# type(V2) : numpy array
def kronVectors(toReturn,V1,V2):
        localV1rangeStart = V1.local_range()[0]
        localV1rangeEnd = V1.local_range()[1]
        V2size = V2.size
        rangeStart = localV1rangeStart*V2size
        rangeEnd = localV1rangeEnd*V2size
        toReturn.init(mpi_comm_world(),(rangeStart,rangeEnd))
        toReturn.set_local( np.kron(V1.array(),V2.flatten()) )

# Must be declared before call:   toReturn = PETSc.Mat()
# type(M1) : dolfin.cpp.la.Matrix
# type(M2) : numpy/scipy sparse matrix
def kronMatrices(toReturn,M1,M2):
	import scipy.sparse as ssp
        M1_petsc4py = as_backend_type(M1).mat()
	M1size = M1_petsc4py.size[0]
	M2size = M2.shape[0]
        theSize = M1size * M2size
        rows, columns, values = M1_petsc4py.getValuesCSR()
	M1_ssp = ssp.csr_matrix((values,columns,rows),shape=(M1size,M1size))

	kronProduct_ssp = ssp.kron(M1_ssp,M2)
	csr = kronProduct_ssp.tocsr()

	toReturn.create(PETSc.COMM_WORLD)
	toReturn.setSizes(theSize)
        toReturn.setPreallocationNNZ(kronProduct_ssp.nnz) # crash faster if after setType !?
	toReturn.setType('aij')
	toReturn.setUp()
	toReturn.setValuesCSR(csr.indptr.tolist(), csr.indices.tolist(), csr.data.tolist())
	toReturn.assemble()
