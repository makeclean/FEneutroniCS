"""
Fletcher benchmark:
- Reference: J.K. Fletcher, 'The solution of the time-independent multi-group neutron transport equation using spherical harmonics', Annals of Nuclear Energy, Vol. 4, pp. 401, 1977.
- Geometry = square-in-a-square (4 and 1.2 cm, respectively)
- Source = 1./1.44 in bottom-left square, zero elsewhere.
- B.C.: Reflection on x=0 and y=0.  Vacuum (can be replaced by fixed flux here) at x=1. and y=1.
- 3-D version also enabled here

Author: S. Van Criekingen
"""
# ------------
# To be fixed by the user:
nx   = 4
ny   = nx
nz   = nx
degree = 1
PnOrder = 3
nDim = 2

## Chose BC at x=1. and y=1. by un-commenting the following:
## To apply vacuum BC:
#BC='vacuum'
## To apply fixed flux BC:
BC='fixed_flux'
fixedFluxValue = 1.0
# ------------

import sys
sys.path.append('../../')
from dolfin import *
import numpy as np
import kronProducts
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

MPIsize = MPI.size(mpi_comm_world())
MPIrank = MPI.rank(mpi_comm_world())

xmin = 0.
xmax = 4.0
ymin = 0.
ymax = 4.0
zmin = 0.
zmax = 4.0

if (nDim==2):
	mesh = RectangleMesh(xmin, ymin, xmax, ymax, nx, ny)  # nx, ny might need to be even given subdomain definition
	numSpherHarmonics = int((PnOrder + 1)/2.)**2
elif (nDim==3):
	mesh = BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz)
	numSpherHarmonics = int( 1 + (PnOrder-1)*(PnOrder+2)/2 )

numVertices_local = mesh.num_vertices()
V = FunctionSpace(mesh, "Lagrange", degree)
numSpatialDOFs = V.dim()
numDOFs = numSpatialDOFs*numSpherHarmonics

if (MPIrank==0):
      print "---- MPIsize = ", MPIsize
      print "---- MPIrank = ", MPIrank
      print "---- nDim = ", nDim
      print "---- nx = ", nx
      print "---- degree = ", degree
      print "---- PnOrder = ", PnOrder
      print "---- numSpatialDOFs = ", numSpatialDOFs
      print "---- numSpherHarmonics = ", numSpherHarmonics
      print "---- numDOFs = ", numSpatialDOFs*numSpherHarmonics
      print "---- BC = ", BC
      if (BC=='fixed_flux'):
              print "---- fixed flux value = ", fixedFluxValue

print "---- on MPIrank ", MPIrank, " numCells_local = ", mesh.num_cells()
print "---- on MPIrank ", MPIrank, " numVertices_local = ", numVertices_local

boundaries = FacetFunction("size_t", mesh)
subdomains = MeshFunction('size_t', mesh, nDim)
V0 = FunctionSpace(mesh, 'DG', 0)

from boundaryClasses import *
left   = Left(); left.myInit(xmin); left.mark(boundaries, 1)
right  = Right(); right.myInit(xmax); right.mark(boundaries, 2)
bottom = Bottom(); bottom.myInit(ymin); bottom.mark(boundaries, 3)
top    = Top(); top.myInit(ymax); top.mark(boundaries, 4)
if (nDim==3):
	back   = Back(); back.myInit(zmin); back.mark(boundaries, 5)
	front  = Front(); front.myInit(zmax); front.mark(boundaries, 6)

from subdomainClasses import *
subdomain0 = Omega0()
if (nDim==2):
	subdomain1 = Omega1_2D()
elif (nDim==3):
	subdomain1 = Omega1_3D()
subdomain0.mark(subdomains, 0)
subdomain1.mark(subdomains, 1)

### Elementary matrices building
# Spatial
u  = TrialFunction(V)
v  = TestFunction(V)
ds = Measure("ds")[boundaries] # must come after subdomain definition

import subdomainFunctionDefinition
values = [0., .6944444444]  # values of f in the two subdomains
f = subdomainFunctionDefinition.returnFunction(V0, subdomains, values)
L  = f*v*dx
b_vec = assemble(L)

N_op = u*v*dx
du_x = nabla_grad(u)[0]; dv_x = nabla_grad(v)[0];
du_y = nabla_grad(u)[1]; dv_y = nabla_grad(v)[1];
Kxx_op = inner(du_x, dv_x)*dx
Kyy_op = inner(du_y, dv_y)*dx
Kxy_op = inner(du_x, dv_y)*dx
if (nDim==3):
	du_z = nabla_grad(u)[2]; dv_z = nabla_grad(v)[2]
	Kzz_op = inner(du_z, dv_z)*dx
	Kxz_op = inner(du_x, dv_z)*dx
	Kyz_op = inner(du_y, dv_z)*dx

N_mat = assemble(N_op)
Kxx_mat = assemble(Kxx_op)
Kxy_mat = assemble(Kxy_op)
Kyy_mat = assemble(Kyy_op) # ! not symmetric but Kxy + Kxy.transpose is symmetric => H symmetric
if (nDim==3):
	Kzz_mat = assemble(Kzz_op)
	Kxz_mat = assemble(Kxz_op)
	Kyz_mat = assemble(Kyz_op)
if (BC=='vacuum'):
	Ngamma_op2 = u*v*ds(2)
	Ngamma_op4 = u*v*ds(4)
	Ngamma_mat2 = assemble(Ngamma_op2)
	Ngamma_mat4 = assemble(Ngamma_op4)
	if (nDim==3):
		Ngamma_op6 = u*v*ds(6)
		Ngamma_mat6 = assemble(Ngamma_op6)

# Angular
from angularMat import *
P = returnMatP(numSpherHarmonics)
import scipy.sparse as ssp
ZZ = ssp.identity(numSpherHarmonics)

iPnOrder = 1
ExEx = np.zeros((numSpherHarmonics,numSpherHarmonics))
EyEy = np.zeros((numSpherHarmonics,numSpherHarmonics))
ExEy = np.zeros((numSpherHarmonics,numSpherHarmonics))
if (nDim==3):
	EzEz = np.zeros((numSpherHarmonics,numSpherHarmonics))
	EyEz = np.zeros((numSpherHarmonics,numSpherHarmonics))
	ExEz = np.zeros((numSpherHarmonics,numSpherHarmonics))
while (iPnOrder <= PnOrder):
   tic()
   jPnOrder = 0
   while (jPnOrder <= iPnOrder):
      Exij = returnMatEx(PnOrder,numSpherHarmonics,iPnOrder,jPnOrder);
      Eyij = returnMatEy(PnOrder,numSpherHarmonics,iPnOrder,jPnOrder);
      ExEx += Exij*np.transpose(Exij)
      EyEy += Eyij*np.transpose(Eyij)
      ExEy += Exij*np.transpose(Eyij)
      if (nDim==3):
         Ezij = returnMatEz(PnOrder,numSpherHarmonics,iPnOrder,jPnOrder);
         EzEz += Ezij*np.transpose(Ezij)
         EyEz += Eyij*np.transpose(Ezij)
         ExEz += Exij*np.transpose(Ezij)
      jPnOrder += 1
   print "---- on MPIrank ", MPIrank, ": time in Angular P", iPnOrder, " =", toc()
   iPnOrder += 2

if (BC=='vacuum'):
	tic()
	Lpx = returnMatLpx(PnOrder,numSpherHarmonics)
	print "Time in returnMatLpx=", toc()
	tic()
	Lpy = returnMatLpy(PnOrder,numSpherHarmonics)
	print "Time in returnMatLpy=", toc()
	if (nDim==3):
	   tic()
  	   Lpz = returnMatLpz(PnOrder,numSpherHarmonics)
	   print "Time in returnMatLpz=", toc()

# Energy
sigma = Constant(0.5)
sigma_r = sigma
SigmaInv = 1./sigma

#### BC preparation: establish the list of indices be treated later on
# For fixed flux BC:
if (BC=='fixed_flux'):
	if (nDim ==2):
		bcsFixedFlux = [DirichletBC(V, 1.0, boundaries, 2),DirichletBC(V, 1.0, boundaries, 4)]
	elif (nDim ==3):
		bcsFixedFlux = [DirichletBC(V, 1.0, boundaries, 2),DirichletBC(V, 1.0, boundaries, 4),DirichletBC(V, 1.0, boundaries, 6)]
	list_indices_fixed_flux1 = list()
	list_indices_fixed_flux2 = list()
	uFixedFlux = Function(V)
	for bc in bcsFixedFlux:
		bc.apply(uFixedFlux.vector())
	d2v = V.dofmap().dofs()
	fixedFluxVertices = d2v[uFixedFlux.vector().get_local() == 1.0]
	list_indices_fixed_flux1 += (fixedFluxVertices * numSpherHarmonics).tolist()
        for ii in range(1,numSpherHarmonics):
		list_indices_fixed_flux2 += ( fixedFluxVertices * numSpherHarmonics + ii ).tolist()

# For reflection BCs:
#  Spherical harmonics to cancel in 2D: Y00, Y20, Y21, Y22, Y40, Y41, Y42, Y43, Y44, Y60, Y61
#  -> to cancel: P3: +2; P5: +5, +7; P7: +10,+12,+14
if (PnOrder>=3):
	if (nDim ==2):
		bcsRefl = [DirichletBC(V, 1.0, boundaries, 1),DirichletBC(V, 1.0, boundaries, 3)]
	elif (nDim ==3):
		bcsRefl = [DirichletBC(V, 1.0, boundaries, 1),DirichletBC(V, 1.0, boundaries, 3),DirichletBC(V, 1.0, boundaries, 5)]
	list_indices_to_cancel = list()
	uRefl = Function(V)
	for bc in bcsRefl:
		bc.apply(uRefl.vector())
	d2v = V.dofmap().dofs()
	reflected_vertices = d2v[uRefl.vector().get_local() == 1.0]
        ii = 1
        for nn in range (2,PnOrder,2):
           for mm in range (0,nn+1):
              if (mm%2==1):  # mm is odd
                 print "  nn=", nn, "  mm=", mm
	         list_indices_to_cancel += ( reflected_vertices * numSpherHarmonics + ii ).tolist()
              ii +=1
           if (nDim ==3):
              for mm in range (-nn,0): # 0 is not an option
                 if (mm%2==0): # mm is even
                    print "  nn=", nn, "  mm=", mm
	            list_indices_to_cancel += ( reflected_vertices * numSpherHarmonics + ii ).tolist()
                 ii +=1


#### RHS building

tic()
rhs_dolfin = PETScVector()
kronProducts.kronVectors(rhs_dolfin,b_vec,P)
print "---- on MPIrank ", MPIrank, ": rhs_dolfin.local_size()=",rhs_dolfin.local_size()
print "---- on MPIrank ", MPIrank, ": rhs_dolfin.local_range()=",rhs_dolfin.local_range()
print "Time in RHS building=", toc()
rhs = as_backend_type(rhs_dolfin).vec() # this is a petsc4py object

#### System building

tic()
matNZZ = PETSc.Mat()
matKEEx = PETSc.Mat()
matKEEy = PETSc.Mat()
matKEExy = PETSc.Mat()
if (nDim ==3):
	matKEEz = PETSc.Mat()
	matKEEyz = PETSc.Mat()
	matKEExz = PETSc.Mat()

kronProducts.kronMatrices(matNZZ,N_mat,ZZ)

print "---- on MPIrank ", MPIrank, ": N_mat.size(0)=",N_mat.size(0)
print "---- on MPIrank ", MPIrank, ": N_mat.size(1)=",N_mat.size(1)
print "---- on MPIrank ", MPIrank, ": N_mat.local_range(0)=",N_mat.local_range(0)
print "---- on MPIrank ", MPIrank, ": N_mat.array().shape=",N_mat.array().shape
print "---- on MPIrank ", MPIrank, ": matNZZ.getOwnershipRange() =",matNZZ.getOwnershipRange()

kronProducts.kronMatrices(matKEEx,Kxx_mat,ExEx)
kronProducts.kronMatrices(matKEEy,Kyy_mat,EyEy)
kronProducts.kronMatrices(matKEExy,Kxy_mat,ExEy)
if (nDim ==3):
	kronProducts.kronMatrices(matKEEz,Kzz_mat,EzEz)
	kronProducts.kronMatrices(matKEExz,Kxz_mat,ExEz)
	kronProducts.kronMatrices(matKEEyz,Kyz_mat,EyEz)
Hodd = SigmaInv * (matKEEx + matKEEy + matKEExy + matKEExy.transpose())
if (nDim ==3):
	Hodd += SigmaInv * (matKEEz + matKEEyz + matKEEyz.transpose() + matKEExz + matKEExz.transpose())
Heven = sigma_r * matNZZ
H = Hodd + Heven
if (BC=='vacuum'):
	HvacuumBC2 = PETSc.Mat()
	HvacuumBC4 = PETSc.Mat()
	kronProducts.kronMatrices(HvacuumBC2,Ngamma_mat2,Lpx)
	kronProducts.kronMatrices(HvacuumBC4,Ngamma_mat4,Lpy)
	H = H + HvacuumBC2 + HvacuumBC4
	if (nDim ==3):
		HvacuumBC6 = PETSc.Mat()
		kronProducts.kronMatrices(HvacuumBC6,Ngamma_mat6,Lpz)
		H += HvacuumBC6
print "Time in system building = ", toc()


#### Applying BC 
tic()
for i in range (0,len(list_indices_fixed_flux1)):
	itc = list_indices_fixed_flux1[i]
	#print "itc 1 =",itc
	H.zeroRows(itc,1)
	rhs.setValue(itc,fixedFluxValue)
for i in range (0,len(list_indices_fixed_flux2)):
	itc = list_indices_fixed_flux2[i]
	#print "itc 2 =",itc
	H.zeroRows(itc,1)
	rhs.setValue(itc,0)
print "Time in applying fixed flux BC=", toc()
tic()
if (PnOrder>=3):
        for i in range (0,len(list_indices_to_cancel)):
		itc = list_indices_to_cancel[i]
		#print "itc=",itc
		H.zeroRowsColumns(itc,1)
		rhs.setValue(itc,0)
print "Time in applying reflection BC=", toc()


#### System solution
tic()
rhs.assemblyBegin(); rhs.assemblyEnd()
H.assemblyBegin(); H.assemblyEnd()
ksp = PETSc.KSP().create()
#ksp.setType('cg')
ksp.setType('gmres')
pc = ksp.getPC()
#pc.setType('none')
pc.setType('ilu')
# Allow for solver choice to be set from command line with -ksp_type <solver>.
# Recommended option: -ksp_type preonly -pc_type lu
###ksp.setFromOptions()
print 'Solving with:', ksp.getType()
ksp.setOperators(H)
x = PETSc.Vec()
x = rhs.duplicate()
# ? x.assemblyBegin(); x.assemblyEnd()
print "Time in fenics detailed assembled pre-solve=", toc()
print "Solving process start with:"
ksp.view()
tic()
ksp.solve(rhs, x)
print "Time in solving process:", toc()
print "Iteration Number:", ksp.getIterationNumber()
print "Residual Norm:", ksp.getResidualNorm()

#the_index = numSpherHarmonics*(numVertices_local-1)/2
#if (degree == 1):
#	#for i in range(0,(int) numSpherHarmonics):
#	for i in range(0,numSpherHarmonics):
#		print "---- x[numSpherHarmonics*(numVertices_local-1)/2 +i] = x[", the_index+i, "] = " ,x[the_index+i]
scalarFlux = Function(V)
scalarFlux.vector()[:] = (x.array[0:numDOFs:numSpherHarmonics]).flatten()
if (nDim == 2):
	print scalarFlux(4.,0.),"    ",scalarFlux(4.,2.),"    ",scalarFlux(4.,4.)
	print scalarFlux(2.,0.),"    ",scalarFlux(2.,2.),"    ",scalarFlux(2.,4.)
	print scalarFlux(0.,0.),"    ",scalarFlux(0.,2.),"    ",scalarFlux(0.,4.)
elif (nDim == 3):
	print scalarFlux(4.,0.,4.),"    ",scalarFlux(4.,2.,2.),"    ",scalarFlux(4.,4.,4.)
	print scalarFlux(2.,0.,2.),"    ",scalarFlux(2.,2.,2.),"    ",scalarFlux(2.,4.,2.)
	print scalarFlux(0.,0.,0.),"    ",scalarFlux(0.,2.,2.),"    ",scalarFlux(0.,4.,4.)

	print scalarFlux(2.,2.,0.),"    ",scalarFlux(2.,2.,4.),"    ",scalarFlux(4.,4.,0.)
plot(scalarFlux)
interactive()
