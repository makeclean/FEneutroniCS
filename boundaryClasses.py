'''
Defines boundaries for 2D square or 3D cubic domain

Author: S. Van Criekingen
'''
# Boundary Classes definition

from dolfin import *

class Left(SubDomain):
    #def __init__(self,xmin):
    def myInit(self,xmin):
        global _xmin
        _xmin = xmin
    def inside(self, x, on_boundary):
	tol = DOLFIN_EPS
	return on_boundary and abs(x[0] - _xmin) < tol            

class Right(SubDomain):
    def myInit(self,xmax):
        global _xmax
        _xmax = xmax
    def inside(self, x, on_boundary):
	tol = DOLFIN_EPS
	return on_boundary and abs(x[0] - _xmax) < tol

class Bottom(SubDomain):
    def myInit(self,ymin):
        global _ymin
        _ymin = ymin
    def inside(self, x, on_boundary):
	tol = DOLFIN_EPS
	return on_boundary and abs(x[1] - _ymin) < tol

class Top(SubDomain):
    def myInit(self,ymax):
        global _ymax
        _ymax = ymax
    def inside(self, x, on_boundary):
	tol = DOLFIN_EPS
	return on_boundary and abs(x[1] - _ymax) < tol                          

class Back(SubDomain):
    def myInit(self,zmin):
        global _zmin
        _zmin = zmin
    def inside(self, x, on_boundary):
	tol = DOLFIN_EPS
	return on_boundary and abs(x[2] - _zmin) < tol                          

class Front(SubDomain):
    def myInit(self,zmax):
        global _zmax
        _zmax = zmax
    def inside(self, x, on_boundary):
	tol = DOLFIN_EPS
	return on_boundary and abs(x[2] - _zmax) < tol                          
