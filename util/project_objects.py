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




