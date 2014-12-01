from dolfin import * # for tic toc
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Must be declared before call:   toReturn = PETSc.Vec()
# type(V1) : dolfin.cpp.la.Vector
# type(V2) : numpy array
def kronVectors(toReturn,V1,V2):
        V1size = V1.size()
	V2size = V2.size
        theSize = V1size * V2size
        toReturn.create(PETSc.COMM_WORLD)
        toReturn.setSizes(theSize,theSize)
        toReturn.setUp()
        #for i in range(0,V1size):
        #        for j in range(0,V2size):
        #                toReturn.setValue(i*V2size+j, V1[i] * V2.flatten()[j])
        indices = range( 0, theSize)
	scalars = np.kron(V1.array(),V2.flatten())
	toReturn.setValues(indices, scalars)

# Must be declared before call:   toReturn = PETSc.Mat()
# type(M1,M2) : numpy or scipy.sparse array
def kronMatrices(toReturn,M1,M2):
	import scipy.sparse as ssp
	M1size = M1.shape[0]
	M2size = M2.shape[0]
        theSize = M1size * M2size
	kronProduct_ssp = ssp.kron(M1,M2)
	csr = kronProduct_ssp.tocsr()

	toReturn.create(PETSc.COMM_WORLD)
	toReturn.setSizes(theSize)
        toReturn.setPreallocationNNZ(kronProduct_ssp.nnz) # crash faster if after setType !?
	toReturn.setType('aij')
	toReturn.setUp()
	toReturn.setValuesCSR(csr.indptr.tolist(), csr.indices.tolist(), csr.data.tolist())
	toReturn.assemble()

# Must be declared before call:   toReturn = PETSc.Mat()
# type(M1) : dolfin.cpp.la.Matrix assembled with backend = uBLAS
# type(M2) : numpy or scipy.sparse array
def kronMatricesOLD(toReturn,M1,M2):
	import scipy.sparse as ssp
        M1size = M1.size(0)
	M2size = M2.shape[0]
        theSize = M1size * M2size
        #print "------ M1size=",M1size," M2size=",M2size," theSize=",theSize
        rows, columns, values = M1.data()
	M1_ssp = ssp.csr_matrix((values,columns,rows),shape=(M1size,M1size))

	kronProduct_ssp = ssp.kron(M1_ssp,M2)
	csr = kronProduct_ssp.tocsr()

	toReturn.create(PETSc.COMM_WORLD)
	toReturn.setSizes(theSize)
        toReturn.setPreallocationNNZ(values.size) # crash faster if after setType !?
	toReturn.setType('aij')
	toReturn.setUp()
	toReturn.setValuesCSR(csr.indptr.tolist(), csr.indices.tolist(), csr.data.tolist())
	toReturn.assemble()
