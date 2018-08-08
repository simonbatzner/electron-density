#!/usr/bin/env python

import unittest


from generate_1d_data import *


if __name__=="__main__":

	mesh=1000
	pot1 = SHO_Potential(xmin=-12,xmax=12,mesh=1000)
	print(pot1.mesh)

	assert pot1.mesh==mesh
	
	
	

	#psi1 = Wavefunction_1D(xmin=-12,xmax=12, mesh=1000,potential=pot1)

	#psi1.plot(mapfunc= lambda x: np.absolute(x) )

	pot2= Gauss_Potential(xmin=0, xmax=1, mesh=1000)
	pot2.plot()
	
	psi2= Wavefunction_1D(xmin=0, xmax=1, mesh=1000, potential=pot2)
	psi2.plot(mapfunc= lambda x: np.absolute(x)**2)
	