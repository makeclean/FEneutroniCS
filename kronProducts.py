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
# type(M1) : PETSc.Mat
# type(M2) : scipy sparse mat, local matrix
def kronMatrices(toReturn,M1,M2):
	import scipy.sparse as ssp
        M1localNrows = M1.sizes[0][0]
        M1localNcols = M1.sizes[0][1]
        M2size = M2.shape[0]
        localNrows = M1localNrows * M2size
        localNcols = M1localNcols * M2size
        #print "------ M1size=",M1size," M2size=",M2size," theSize=",theSize
        M1localIndptr, M1localIndices, M1localData = M1.getValuesCSR()
        M1_ssp = ssp.csr_matrix((M1localData,M1localIndices,M1localIndptr),shape=(M1localNrows,M1localNcols))

        kronProduct_ssp = ssp.kron(M1_ssp,M2)
        csr = kronProduct_ssp.tocsr()

        toReturn.create(PETSc.COMM_WORLD)
        toReturn.setSizes(((localNrows, PETSc.DETERMINE), (localNcols, PETSc.DETERMINE)), bsize=1)
        toReturn.setPreallocationNNZ(M1localData.size) # crash faster if after setType !?
	toReturn.setType('aij')
	toReturn.setUp()
	toReturn.setValuesCSR(csr.indptr.tolist(), csr.indices.tolist(), csr.data.tolist())
	toReturn.assemble()
