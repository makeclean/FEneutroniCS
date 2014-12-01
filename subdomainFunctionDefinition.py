from dolfin import *
import numpy as np

# Subdomain Function Definition
def returnFunction(V0,subdomains,values):
	f = Function(V0)
	f_values = values  # values of f in the two subdomains
	helpArray = np.asarray(subdomains.array(), dtype=np.int32)
        f.vector().set_local(np.choose(helpArray, f_values))
	##print 'f degree of freedoms:', f.vector().array()
	return f
