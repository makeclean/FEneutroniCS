from dolfin import *

# Omega0 = all domain, from which ather domains will be cut off
class Omega0(SubDomain):
    def inside(self, x, on_boundary):
	return True
class Omega1_2D(SubDomain):
    def inside(self, x, on_boundary):
	XregionInterfaces = [0,1.2,4.];
	YregionInterfaces = [0,1.2,4.];
	return True if x[0] >= XregionInterfaces[0] and x[0] <= XregionInterfaces[1] and x[1] >= YregionInterfaces[0] and x[1] <= YregionInterfaces[1] else False
class Omega1_3D(SubDomain):
    def inside(self, x, on_boundary):
	XregionInterfaces = [0,1.2,4.];
	YregionInterfaces = [0,1.2,4.];
	ZregionInterfaces = [0,1.2,4.];
	return True if x[0] >= XregionInterfaces[0] and x[0] <= XregionInterfaces[1] and x[1] >= YregionInterfaces[0] and x[1] <= YregionInterfaces[1] and x[2] >= ZregionInterfaces[0] and x[2] <= ZregionInterfaces[1] else False
