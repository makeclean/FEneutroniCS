import numpy as np
from scipy.special import sph_harm
from scipy.integrate import dblquad

def returnMatP(numSpherHarmonics):
	matP = np.zeros((numSpherHarmonics,1))
        matP[0] = 1.
	return matP

def returnMatEx(PnOrder,numSpherHarmonics,n1,m1):
	#PnOrder must be odd
	# n1 must be in range (0,n_max+1):
	# m1 must be  in range (0,n1+1):
	n_max = PnOrder-1 # must be even
	#print '-- PnOrder = ', PnOrder, '  numSpherHarmonics = ', numSpherHarmonics
	i=0
	matEx = np.zeros((numSpherHarmonics,1))
	for n2 in range (0,n_max+1,2):
	   for m2 in range (0,n2+1):
	      if (m1==0): c1 = 1.
	      else: c1 = np.sqrt(2.)
	      if (m2==0): c2 = 1.
	      else: c2 = np.sqrt(2.)
	      def integrand(mu,phi):
	        theta = np.arccos(mu)
	        if ((m1 >= 0) and (m2>=0)):
	           return mu * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
  	        elif ((m1 >= 0) and (m2<0)):
	           return mu * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(-m2, n2, phi, theta).imag
	        elif ((m1 < 0) and (m2>=0)):
	           return mu * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(m2, n2, phi, theta).real
	        elif ((m1 < 0) and (m2<0)):
	           return mu * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(-m2, n2, phi, theta).imag
	      ans, err = dblquad(lambda mu, phi: integrand(mu,phi), 0, 2*np.pi, lambda mu: -1, lambda mu: 1)
	      matEx[i,0] = ans
	      i = i + 1;
	   # end for m2
	# end for n2
	#print matEx
	return matEx

def returnMatEy(PnOrder,numSpherHarmonics,n1,m1):
	#PnOrder must be odd
	# n1 must be in range (0,n_max+1):
	# m1 must be  in range (0,n1+1):
	n_max = PnOrder-1 # must be even
	#print '-- PnOrder = ', PnOrder, '  numSpherHarmonics = ', numSpherHarmonics
	i=0
	matEy = np.zeros((numSpherHarmonics,1))
	for n2 in range (0,n_max+1,2):
	   for m2 in range (0,n2+1):
	      if (m1==0): c1 = 1.
	      else: c1 = np.sqrt(2.)
	      if (m2==0): c2 = 1.
	      else: c2 = np.sqrt(2.)
	      def integrand(mu,phi):
	        theta = np.arccos(mu)
                if ((m1 >= 0) and (m2>=0)):
                   return np.sin(theta) * np.cos(phi) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
                elif ((m1 >= 0) and (m2<0)):
                   return np.sin(theta) * np.cos(phi) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(-m2, n2, phi, theta).imag
                elif ((m1 < 0) and (m2>=0)):
                   return np.sin(theta) * np.cos(phi) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(m2, n2, phi, theta).real
                elif ((m1 < 0) and (m2<0)):
                   return np.sin(theta) * np.cos(phi) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(-m2, n2, phi, theta).imag
	      ans, err = dblquad(lambda mu, phi: integrand(mu,phi), 0, 2*np.pi, lambda mu: -1, lambda mu: 1)
	      matEy[i,0] = ans
	      i = i + 1;
	   # end for m2
	# end for n2
	#print matEy
	return matEy

def returnMatEz(PnOrder,numSpherHarmonics,n1,m1):
	#PnOrder must be odd
	# n1 must be in range (0,n_max+1):
	# m1 must be  in range (0,n1+1):
	n_max = PnOrder-1 # must be even
	#print '-- PnOrder = ', PnOrder, '  numSpherHarmonics = ', numSpherHarmonics
	i=0
	matEz = np.zeros((numSpherHarmonics,1))
	for n2 in range (0,n_max+1,2):
	   for m2 in range (0,n2+1):
	      if (m1==0): c1 = 1.
	      else: c1 = np.sqrt(2.)
	      if (m2==0): c2 = 1.
	      else: c2 = np.sqrt(2.)
	      def integrand(mu,phi):
	        theta = np.arccos(mu)
                if ((m1 >= 0) and (m2>=0)):
                   return np.sin(theta) * np.sin(phi) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
                elif ((m1 >= 0) and (m2<0)):
                   return np.sin(theta) * np.sin(phi) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(-m2, n2, phi, theta).imag
                elif ((m1 < 0) and (m2>=0)):
                   return np.sin(theta) * np.sin(phi) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(m2, n2, phi, theta).real
                elif ((m1 < 0) and (m2<0)):
                   return np.sin(theta) * np.sin(phi) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(-m2, n2, phi, theta).imag
	      ans, err = dblquad(lambda mu, phi: integrand(mu,phi), 0, 2*np.pi, lambda mu: -1, lambda mu: 1)
	      matEz[i,0] = ans
	      i = i + 1;
	   # end for m2
	# end for n2
	#print matEz
	return matEz

def returnMatLpx(PnOrder,numSpherHarmonics):
	#PnOrder must be odd
	n_max = PnOrder-1 # must be even
	i=0
	j=0
	matLpx = np.zeros((numSpherHarmonics,numSpherHarmonics))
	for n1 in range (0,n_max+1,2):
	   for m1 in range (0,n1+1):
	      for n2 in range (0,n_max+1,2):
		 for m2 in range (0,n2+1):
		    if (m1==0): c1 = 1. 
		    else: c1 = np.sqrt(2.)
		    if (m2==0): c2 = 1. 
		    else: c2 = np.sqrt(2.)
		    def integrandX(mu,phi):
		       theta = np.arccos(mu)
		       if ((m1 >= 0) and (m2>=0)):
			   return abs(mu) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
		       elif ((m1 >= 0) and (m2<0)):
			   return abs(mu) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(-m2, n2, phi, theta).imag
		       elif ((m1 < 0) and (m2>=0)):
			   return abs(mu) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(m2, n2, phi, theta).real
		       elif ((m1 < 0) and (m2<0)):
			   return abs(mu) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(-m2, n2, phi, theta).imag
		    ansX, err = dblquad(lambda mu, phi: integrandX(mu,phi), 0, 2*np.pi, lambda mu: -1, lambda mu: 1)
		    matLpx[i,j] = ansX
		    j = j + 1;
		 # end for m2
	      i = i + 1;
	      j = 0;
	   # end for m1
	return matLpx

def returnMatLpy(PnOrder,numSpherHarmonics):
	#PnOrder must be odd
	n_max = PnOrder-1 # must be even
	i=0
	j=0
	matLpy = np.zeros((numSpherHarmonics,numSpherHarmonics))
	for n1 in range (0,n_max+1,2):
	   for m1 in range (0,n1+1):
	      for n2 in range (0,n_max+1,2):
		 for m2 in range (0,n2+1):
		    if (m1==0): c1 = 1. 
		    else: c1 = np.sqrt(2.)
		    if (m2==0): c2 = 1. 
		    else: c2 = np.sqrt(2.)
		    def integrandY(mu,phi):
		       theta = np.arccos(mu)
		       if ((m1 >= 0) and (m2>=0)):
			   return abs(np.sin(theta) * np.cos(phi)) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
			   #return abs(np.sqrt(1-mu**2) * np.cos(phi)) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
		       elif ((m1 >= 0) and (m2<0)):
			   return abs(np.sin(theta) * np.cos(phi)) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(-m2, n2, phi, theta).imag
		       elif ((m1 < 0) and (m2>=0)):
			   return abs(np.sin(theta) * np.cos(phi)) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(m2, n2, phi, theta).real
		       elif ((m1 < 0) and (m2<0)):
			   return abs(np.sin(theta) * np.cos(phi)) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(-m2, n2, phi, theta).imag
		    ansY, err = dblquad(lambda mu, phi: integrandY(mu,phi), 0, 2*np.pi, lambda mu: -1, lambda mu: 1)
		    matLpy[i,j] = ansY
		    j = j + 1;
		 # end for m2
	      i = i + 1;
	      j = 0;
	   # end for m1
	return matLpy

def returnMatLpz(PnOrder,numSpherHarmonics):
	#PnOrder must be odd
	n_max = PnOrder-1 # must be even
	i=0
	j=0
	matLpz = np.zeros((numSpherHarmonics,numSpherHarmonics))
	for n1 in range (0,n_max+1,2):
	   for m1 in range (0,n1+1):
	      for n2 in range (0,n_max+1,2):
		 for m2 in range (0,n2+1):
		    if (m1==0): c1 = 1. 
		    else: c1 = np.sqrt(2.)
		    if (m2==0): c2 = 1. 
		    else: c2 = np.sqrt(2.)
		    def integrandZ(mu,phi):
		       theta = np.arccos(mu)
		       if ((m1 >= 0) and (m2>=0)):
			   return abs(np.sin(theta) * np.sin(phi)) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
			   #return abs(np.sqrt(1-mu**2) * np.cos(phi)) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(m2, n2, phi, theta).real
		       elif ((m1 >= 0) and (m2<0)):
			   return abs(np.sin(theta) * np.sin(phi)) * c1*sph_harm(m1, n1, phi, theta).real * c2*sph_harm(-m2, n2, phi, theta).imag
		       elif ((m1 < 0) and (m2>=0)):
			   return abs(np.sin(theta) * np.sin(phi)) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(m2, n2, phi, theta).real
		       elif ((m1 < 0) and (m2<0)):
			   return abs(np.sin(theta) * np.sin(phi)) * c1*sph_harm(-m1, n1, phi, theta).imag * c2*sph_harm(-m2, n2, phi, theta).imag
		    ansZ, err = dblquad(lambda mu, phi: integrandZ(mu,phi), 0, 2*np.pi, lambda mu: -1, lambda mu: 1)
		    matLpz[i,j] = ansZ
		    j = j + 1;
		 # end for m2
	      i = i + 1;
	      j = 0;
	   # end for m1
	return matLpz
