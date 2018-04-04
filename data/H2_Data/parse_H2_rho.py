#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:13:18 2018

@author: steven
"""
import numpy


def rho_to_numpy(file_path,natoms=2):
	
	with open(file_path,'r') as f:
		thelines=f.readlines()

	#Parse the first 6 lines by default (the slice is exclusive ,hence :7)
	# Assuming we have 2 lines of comment header,
	# 3 lines of volumetric information,
	#
	header=thelines[0:6+natoms]
	body = thelines[6+natoms:]

	header=[line.strip() for line in header]
	## ASSUMING CUBIC LATTICE FOR NOW/MOLECULE
	nx = int(header[3].split()[0])
	dx = float(header[3].split()[1])
	print(dx)
    
	
	ny = int(header[4].split()[0])
	dy = float(header[4].split()[2])
	print(dy)
	
	nz = int(header[5].split()[0])
	dz = float(header[5].split()[3])
	print(dz)

	
	array=np.empty(shape=(nx,ny,nz))
	body_index=0
	cur_line=body[body_index].split()
	print(cur_line)
	print(nx,ny,nz)
	for x in range(nx):
		for y in range(ny):
			for z in range(nz):
				
				if len(cur_line)!=0:
					array[x,y,z]=cur_line.pop(0)
					#print("just loaded in value",array[x,y,z])
				else:
					body_index+=1
					cur_line=body[body_index].split()
					array[x,y,z]=cur_line.pop(0)
					#print("just loaded in value",array[x,y,z])
					#print("Working on line",body_index)
		

	return array

#if __name__=="__main__":
#	
#	
#	spacings= np.linspace(0.25483192,3,20)
#	ecut=70
#	
#	test_rho='/Users/steven/Documents/Schoolwork/CDMAT275/MLED/ML-electron-density/data/H2_Data/rho_data/H2_a_0.25483192_ecut_70.rho.dat'
#	
#	array=rho_to_numpy(test_rho)
#	print(array.shape)
#
#	print(array[0,0,0])
#	print(np.sum(array) *0.177162**3)

	
	
	
