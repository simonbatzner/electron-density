#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ase, json
import numpy as np
from util import *
from objects import *
import matplotlib.pyplot as plt
import sys

class Function_1D(Generic):
    """ A domain and range of a function"""
		
    pass

class Discrete_Function_1D(Function_1D):
		
	def __init__(self, xmin=None, xmax=None, mesh=None, **kwargs):
		
		self.xmin=xmin
		self.xmax=xmax
		self.mesh=mesh
		
		self.domain=np.linspace(xmin,xmax,mesh)
		
		if mesh is not None:
			self.values=np.empty(mesh)
		else:
			self.values=[]
			
			
	
	def plot(self,mapfunc=lambda x: x):
		plt.figure()
		ax=plt.gca()
		ax.x_label=('x')
		ax.plot(self.domain,mapfunc(self.values))
		ax.set_ylabel=('$f(x)$')
		plt.show()
		return ax
	


class Potential_Function(Discrete_Function_1D):
	
	def __init__(self,xmin, xmax, mesh, **kwargs):
		super().__init__(xmin, xmax, mesh, **kwargs)


class ESPRESSO_Output(Path):
		
	pass



class PP_Plot_Vectors(Param):
    """
    Data class describing the shape of the vectors used for density writing in the PP.x code
    Example: {'e(1)': [1.0, 0.0, 0.0'], e(2) [0.0, 1.0, 0.0]...}
	"""
	
    pass



class PW_PP_inparam(Param):
	"""
	Data class containing parameters for a Quantum Espresso Post-Processing calculation
	"""
	pass
