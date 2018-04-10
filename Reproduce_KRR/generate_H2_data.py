#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:08:46 2018

@author: steven
"""
import sys
import os

sys.path.append('../../util')


from project_pwscf import *
from ase.io import write
from ase import Atoms
from ase.build import molecule
import matplotlib.pyplot as plt

import numpy as np

def make_struc(alat=1.5,vacuum=5.0,dimer=False):
    center=vacuum
    
    	
    if not dimer:
		#H=Atoms('H', positions=[(0,0,0)])
        H=molecule("H",vacuum=vacuum)
        H.positions=[[center,center,center]]
    else:
        H=molecule("H2",vacuum=vacuum)
        H.cell=[vacuum*2,vacuum*2,vacuum*2]
        H.positions=[[center-alat/2.,center,center],[alat/2.+center,center,center]]
    
    	# check how your cell looks like
    	#write('s.cif', gecell)
    structure = Struc(ase2struc(H))
    return structure




def compute_H_energy(ecut=90,alat=.75,vacuum=5.29,relax=False,\
                     verbose=False,dimer=False,ncpu=1):
	"""
	Make an input template and select potential and structure, and the path where to run
	"""	
	#potname = 'H.pz-van_ak.UPF'
	#potname = 'H.rel-pbe-kjpaw.UPF'
	potname= 'H.pbe-kjpaw.UPF'
	

	potpath = os.path.join(os.environ['ESPRESSO_PSEUDO'], potname)
	pseudopots = {'H': PseudoPotential(path=potpath, ptype='uspp', element='H',
										functional='GGA', name=potname)}
	struc = make_struc(alat=alat,dimer=dimer,vacuum=vacuum)
	
	kpts = Kpoints(gridsize=[0, 0, 0], option='gamma', offset=False)
	
	constraint = Constraint(atoms={'0': [1,0,0], '1':[1,0,0]})

	dirname = '{}_a_{}_ecut_{}'.format('H2' if dimer else 'H',alat if dimer else 0, ecut)
	runpath = Dir(path=os.path.join(os.environ['PROJDIR'], "data/H2_DFT/temp_data", dirname))
	input_params = PWscf_inparam({
		'CONTROL': {
				   'prefix': 'H2' if dimer else 'H', #+str(alat) if dimer else 'H'+str(alat),
			'calculation': 'relax' if relax else 'scf',
			'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
			'outdir': runpath.path,
			'wfcdir': runpath.path,
			'disk_io': 'medium',
			#'tprnfor' : '.true.'
			'wf_collect' :True
		},
		'SYSTEM': {
			'ecutwfc': ecut,
			'ecutrho': ecut * 4,
			'nspin': 4 if 'rel' in potname else 1,
#			'occupations': 'smearing',
#			'smearing': 'mp',
#			'degauss': 0.02,
			'noinv': False
			#'lspinorb':True if 'rel' in potname else False,
			 },
		'ELECTRONS': {
			'diagonalization': 'david',
			'mixing_beta': 0.5,
			'conv_thr': 1e-7,
		},
		'IONS': {},
		'CELL': {},
		})

	output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
							   params=input_params, kpoints=kpts,constraint=constraint,
							    ncpu=ncpu)
	output = parse_qe_pwscf_output(outfile=output_file)
	if verbose:
		print("Done with run with ",alat,ecut)
	return output

## TODO: This needs to be cleaned up
def compute_H_density(prefix='H2',outdir='.',datarange=np.linspace(.5,1.5,100)):
	
	## THIS BLOCK IS UNFINISHED
	plot_vecs = PP_Plot_Vectors(atoms={'e1': [1.0,0,0], 'e2':[0,1.0,0],
									 'e3':[0,0,1.0]})

	
	input_params= PW_PP_inparam({
			'inputpp':{
				'prefix' : 'H2',
				'outdir': outdir,
				'filplot' :'H2rho',
				'plot_num': 0    #Outputs Density
				}
			,'plot':{
					'nfile' : 1,
					'filepp(1)': 'H2rho',
					'iflag' : 3,
					'output_format' : 6,
					'fileout' : 'H2.rho.dat'
					}

			})
	
	run_qe_pp(input_params=input_params,plot_vecs=plot_vecs)
	
	input_text = """	
		 &inputpp
	    prefix  = 'H2'
		outdir = '/Users/steven/Documents/Schoolwork/CDMAT275/MLED/ML-electron-density/data/H2_DFT/test_data/'
		filplot = 'H2rho'
	    plot_num= 0
	 /
	 &plot
	    nfile = 1
	    filepp(1) = 'H2rho'
	    weight(1) = 1.0
	    iflag = 3
	    output_format = 6
	    fileout = 'H2.rho.dat'
	    e1(1) =1.0, e1(2)=0.0, e1(3) = 0.0,
	    e2(1) =0.0, e2(2)=1.0, e2(3) = 0.0,
	    e3(1) =0.0, e3(2)=0.0, e2(3) = 1.0,
	"""
	
	
	
	
	#This is temporary, but work
def quick_density_gen(strain_vals,ecut,verbose=True):
	work_dir = Dir(path=os.path.join(os.environ['PROJDIR'], "data/H2_DFT/temp_data"))

	os.chdir(work_dir.path)

	for val in strain_vals:
#		print("Converting the output density of ",val, " to real-space density ")
		
		dirname = '{}_a_{}_ecut_{}'.format('H2', val,ecut)
		
		# KEY:
		# Prefix: String to prefix every file generated by PP
		# outdir: Where to look for the output of a PWscf generated calculation
		#			i.e. where the input and output files are, as well as 
		#		the .save folder
		# filplot: What to call the saved file
		# plot_num: integer which describes the quantity in question, 0=density
		
		# nfile=1; number of output data files
		# filepp(1): Where to find a PP output file to convert into a new form
		#			in our case, this points to the filplot value
		# Weight: not used
		# iflag=3: Specifies we want a 3d plot
		# output_format: Tells us what format we want, here gaussian cube format
		# fileout: Where to write the processed code read in from filepp
		#  e1..e3: The basis vectors in units of alat for the plot
		
		input_text="""
		 &inputpp
		    prefix  = 'H2'
			outdir = '"""+work_dir.path+'/'+dirname + """'
			filplot = './"""+dirname+"""/H2rho'
		    plot_num= 0
		 /
		 &plot
		    nfile = 1
		    filepp(1) = './""" + dirname + """/H2rho'
		    weight(1) = 1.0
		    iflag = 3
		    output_format = 6
		    fileout = './""" + dirname + """/H2.rho.dat'
		    e1(1) =1.0, e1(2)=0.0, e1(3) = 0.0,
		    e2(1) =0.0, e2(2)=1.0, e2(3) = 0.0,
		    e3(1) =0.0, e3(2)=0.0, e3(3) = 1.0,
		/
		"""
	
		#print(input_text)
		with open('H2_rho_temp.pp','w') as f:
			f.write(input_text)
		os.system( os.environ["PP_COMMAND"] + " < h2_rho_temp.pp")
#		print(os.environ["PP_COMMAND"]+ ( " < h2_rho_temp.pp"))
		os.system("cp ./"+dirname+"/H2.rho.dat ../out_data/"+dirname+".rho.dat")
		
		
		
#if __name__=='__main__':
	
	## I found the equilibrium spacing to be .7504 anstrom, which is not far off from the
	## experimental value of .74.
	
#	print(compute_H_energy(alat=.75, ecut=90, dimer=True, relax=True, vacuum=5))

	## Determine the spacings to use and iterate:
#	spacings= np.linspace(.5,1.5,10)
#	spacing_outs=[ compute_H_energy(alat=a, ecut=90, dimer=True,relax=False,verbose=True) for a in spacings]
#	plt.plot(spacings, [out['energy'] for out in spacing_outs])

#	plt.figure()
#	plt.show()
	
#	plt.figure()
#	plt.semilogy(erange, [np.abs(out['energy']-dimer_outs[-1]['energy']) for out in dimer_outs])
#	plt.show()
#	quick_density_gen(strain_vals=np.linspace(.5,1.5,100),ecut=90,verbose=True)
	
	
	
	