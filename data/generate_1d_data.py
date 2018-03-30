#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../util')
from project_objects import Discrete_Function_1D,Potential_Function
import numpy.random as rand
import scipy.sparse.linalg as spl




def gaussian(x,a,b,c):
	"""
	Return the value of a gaussian with weight a, width b, centered at c
	"""
	return a * np.exp( -(x - b)**2 / (2 * c*c) )

def schrodinger_solve_2d_order(wavefunction,potential):
	"""
	Second-order accurate solver for the Schrodinger Equation.
	Takes input wave-function and potential and writes the ground-state 
	wavefunction values into the wavefunction object.
	"""

	# Get the mesh from the input potential
	mesh=potential.mesh
	if potential.mesh!=wavefunction.mesh:
		print("Warning!! The wavefunction and the potential \
			 appear to be defined on different domains!!")
	domain = potential.domain
	
	# Get the domain spacing Delta x = h
	h=domain[1]-domain[0]
	# Precompute the inverse of h squared
	hhinv=1/(h**2)

	# Instantiate a second-order finite difference matrix
	# Once constructed, when this matrix is applied to a 
	# discretized vector representing a continuous function u(x),
	# will return a vector second derivative  of u
	T= -2 * np.eye(mesh)

    # Add the centered-difference discretization for the second derivative
	for i in range(mesh):
		if i!=mesh-1:
			T[i,i+1]=1
		if i!=0:
			T[i,i-1]=1

	# Scale by the interval
	T= T* hhinv

	# Create a matrix representing the potential 
	V       = np.diag(potential.values)
	# Solve the eigenvalue problem and the ground-state eigenvector
	Eig,Vec = spl.eigs(-T +V,k=1,which='SM')
	

	# Store the wave-function

	wavefunction.energy = np.real(Eig)[0]; 
	wavefunction.values = Vec;





class Gauss_Potential(Potential_Function):
	"""
	Defines a potential as the sum of N gaussians on an interval 0,1.
	"""
	def __init__(self, xmin=0, xmax=1, mesh=500, N=3, amin=-10, 
				amax = -1, bmin = .4, bmax = .6, cmin = .03, 
				cmax =.1, **kwargs):

		super().__init__(xmin,xmax,mesh,**kwargs)

		#We take the minimum and maximum to allow for positive or negative numbers
		self.a_vals = [rand.uniform(min(amin, amax), max(amin,amax)) for n in range(N)]
		self.b_vals = [rand.uniform(min(bmin, bmax), max(bmin,bmax)) for n in range(N)]
		self.c_vals = [rand.uniform(min(cmin, cmax), max(cmin,cmax)) for n in range(N)]
		
		self.N=N


		for m in range(mesh):
			self.values[m]=0.
			for n in range(N):
				self.values[m]+=gaussian(x=self.domain[m], a=self.a_vals[n], 
				  b=self.b_vals[n],c=self.c_vals[n])




class SHO_Potential(Potential_Function):
	"""
	Defines a potential as the sum of N gaussians on an interval 0,1.
	"""
	def __init__(self, xmin=0, xmax=1, mesh=500, omega=1,
			  y_offset=0, x_offset=0, **kwargs):

		
		super().__init__(xmin,xmax,mesh,**kwargs)

		self.omega=omega
		self.y_offset=y_offset
		self.x_offset=x_offset
		
		SHO = lambda x: omega * (x + x_offset)**2 + y_offset

		SHO_func = np.vectorize(SHO)

		self.values = SHO_func(self.domain)



class Wavefunction_1D(Discrete_Function_1D):

	def __init__(self,xmin=0,xmax=1,mesh=500,potential=None,**kwargs):

		self.energy=0

		super().__init__(xmin,xmax,mesh,**kwargs)

		self.density = np.array(mesh)
		

		self.domain = np.linspace(xmin,xmax,mesh)

		if potential != None:
			schrodinger_solve_2d_order(self,potential)
			self.density = np.absolute(self.values)



def main():
	
	Npots=5
	xmin=0
	xmax=1
	mesh=500
	
	domain=np.linspace(xmin,xmax,mesh)
	rhos=np.empty((Npots,mesh))

	for i in range(Npots):
		pot = Gauss_Potential(mesh=mesh)
		psi = Wavefunction_1D(mesh=mesh, potential=pot)
		rhos[i] = psi.density.flatten()
		
	with open('pot_densities.csv','w') as f:
		f.write("x_position/Potential,")
		for m in range((Npots)):
			f.write('%d,'%m)
		f.write('\n')
		
		for n in range(mesh):
			f.write('%f,' % domain[n])
			for m in range(Npots):
				f.write('%f,' %rhos[m,n])
			
			f.write('\n')

if __name__=="__main__":
	main()