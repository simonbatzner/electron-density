from objects import *
import numpy
import os


class PWscf_inparam(Param):
	"""
	Data class containing parameters for a Quantum Espresso PWSCF calculation
	it does not include info on the cell itself, since that will be taken from a Struc object
	"""
	pass


def qe_value_map(value):
	"""
	Function used to interpret correctly values for different
	fields in a Quantum Espresso input file (i.e., if the user
	specifies the string '1.0d-4', the quotes must be removed
	when we write it to the actual input file)
	:param: a string
	:return: formatted string to be used in QE input file
	"""
	if isinstance(value, bool):
		if value:
			return '.true.'
		else:
			return '.false.'
	elif isinstance(value, (float, numpy.float)) or isinstance(value, (int, numpy.int)):
		return str(value)
	elif isinstance(value, str):
		return "'{}'".format(value)
	else:
		print("Strange value ", value)
		raise ValueError


def write_pwscf_input(runpath, params, struc, kpoints, pseudopots, constraint=None):
	"""Make input param string for PW"""
	# automatically fill in missing values
	pcont = params.content
	pcont['SYSTEM']['ntyp'] = struc.n_species
	pcont['SYSTEM']['nat'] = struc.n_atoms
	pcont['SYSTEM']['ibrav'] = 0
	# Write the main input block
	inptxt = ''
	for namelist in ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']:
		inptxt += '&{}\n'.format(namelist)
		for key, value in pcont[namelist].items():
			inptxt += '	{} = {}\n'.format(key, qe_value_map(value))
		inptxt += '/ \n'
	# write the K_POINTS block
	if kpoints.content['option'] == 'automatic':
		inptxt += 'K_POINTS {automatic}\n'
	
	
	if kpoints.content['option'] == 'gamma':
		inptxt +="K_POINTS {gamma}\n"

	else:			
		inptxt += ' {:d} {:d} {:d}'.format(*kpoints.content['gridsize'])
	
	
		if kpoints.content['offset']:
			inptxt += '  1 1 1\n'
		else:
			inptxt += '  0 0 0\n'

	# write the ATOMIC_SPECIES block
	inptxt += 'ATOMIC_SPECIES\n'
	for elem, spec in struc.species.items():
		inptxt += '  {} {} {}\n'.format(elem, spec['mass'], pseudopots[elem].content['name'])

	# Write the CELL_PARAMETERS block
	inptxt += 'CELL_PARAMETERS {angstrom}\n'
	for vector in struc.content['cell']:
		inptxt += ' {} {} {}\n'.format(*vector)

	# Write the ATOMIC_POSITIONS in crystal coords
	inptxt += 'ATOMIC_POSITIONS {angstrom}\n'
	for index, positions in enumerate(struc.content['positions']):
		inptxt += '  {} {:1.5f} {:1.5f} {:1.5f}'.format(positions[0], *positions[1])
		if constraint and constraint.content['atoms'] and str(index) in constraint.content['atoms']:
			inptxt += ' {} {} {} \n'.format(*constraint.content['atoms'][str(index)])
		else:
			inptxt += '\n'

	infile = TextFile(path=os.path.join(runpath.path, 'pwscf.in'), text=inptxt)
	infile.write()
	return infile


def run_qe_pwscf(struc, runpath, pseudopots, params, kpoints, constraint=None, ncpu=1,parallelization={}):
	pwscf_code = ExternalCode({'path': os.environ['PWSCF_COMMAND']})
	prepare_dir(runpath.path)
	infile = write_pwscf_input(params=params, struc=struc, kpoints=kpoints, runpath=runpath,
							   pseudopots=pseudopots, constraint=constraint)
	outfile = File({'path': os.path.join(runpath.path, 'pwscf.out')})
    
	parallelization_str=""
	if parallelization!={}:
		for key,value in parallelization.items():
			if value!=0:
				parallelization_str+= '-%s %d '%(key,value)
	else:
		parallelization_str="-np %d"%(ncpu)
	pwscf_command = "mpirun {} {} < {} > {}".format(parallelization_str, pwscf_code.path, infile.path, outfile.path)
	run_command(pwscf_command)
	return outfile


def parse_qe_pwscf_output(outfile):

	forces=[]
	positions=[]
	total_force = None
	pressure = None
	
	#Flags to start collecting final coordinates
	final_coords=False
	get_coords=False
	
	with open(outfile.path, 'r') as outf:
		for line in outf:
			if line.lower().startswith('     pwscf'):
				walltime = line.split()[-3] + line.split()[-2]
			
			if line.lower().startswith('     total force'):
				total_force = float(line.split()[3]) * (13.605698066 / 0.529177249)
			
			if line.lower().startswith('!    total energy'):
				total_energy = float(line.split()[-2]) * 13.605698066
				
			if line.lower().startswith('          total   stress'):
				pressure = float(line.split('=')[-1])
			
			if line.lower().startswith('     number of k points='):
				kpoints  = float(line.split('=')[-1])

			if line.lower().startswith('     unit-cell volume'):
				line=line.split('=')[-1]
				line=line.split('(')[0]
				line=line.strip()
				volume   = float(line)
				
			## Chunk of code meant to get the final coordinates of the atomic positions
			if line.lower().startswith('begin final coordinates'):
				final_coords=True
				continue
			if line.lower().startswith('atomic_positions') and final_coords==True:
				continue
			if line.lower().startswith('end final coordinates'):
				final_coords=False
			if final_coords==True and line.split()!=[]:
				positions.append( [ line.split()[0], [float(line.split()[1]),
						      float(line.split()[2]),float(line.split()[3])  ] ] )
				
				
			if line.find('force')!=-1 and line.find('atom')!=-1:
				line=line.split('force =')[-1]
				line=line.strip()
				line=line.split(' ')
				#print("Parsed line",line,"\n") 
				line=[x for x in line if x!='']
				temp_forces=[]
				for x in line:
					temp_forces.append(float(x))
				forces.append(list(temp_forces))
				
				
				

	result = {'energy': total_energy, 'kpoints':kpoints, 'volume': volume, 'positions':positions}
	if forces!=[]:
		result['forces'] = forces
	if total_force!=None:
		result['total_force'] = total_force
	if pressure!=None:
		result['pressure'] = pressure
	return result



##################### NEW BELOW

def parse_qe_pwscf_md_output(outfile):
#def parse_qe_pwscf_md_output(path):
	
	steps={}
	
	# Get the lines out of the file first
	with open(outfile.path, 'r') as outf:
		lines = outf.readlines()
		
	# Because every step is marked by a total energy printing with the !
	# as the first character of the line, partition the file of output
	# into all different chunks of run data
	
	# Get the indexes to carve up the document later
	split_indexes=[N for N in range(len(lines)) if '!'==lines[N][0]]

	# Cut out the first chunk 
	# TODO: Analyze first chunk
	first_chunk=lines[0:split_indexes[0]]
	
	step_chunks = []
	# Carve up into chunks	
	for n in range(len(split_indexes)):
		step_chunks.append(lines[split_indexes[n]:split_indexes[n+1] if n!=len(split_indexes)-1 else len(lines)]) 
			
			
	
	# Iterate through chunks
	for current_chunk in step_chunks:
		
		
		# Iterate through to find the bounds of regions of interest
		
		# Forces
		force_start_line = [line for line in current_chunk if 'Forces acting on atoms' in line][0]
		force_end_line   = [line for line in current_chunk if 'Total force' in line][0]
		force_start_index = current_chunk.index(force_start_line)+2
		force_end_index = current_chunk.index(force_end_line)-2
		 
		# Positions
		atoms_start_line = [line for line in current_chunk if 'ATOMIC_POSITIONS' in line][0]
		atoms_end_line   = [line for line in current_chunk if 'kinetic energy' in line][0]
		atoms_start_index = current_chunk.index(atoms_start_line)+1
		atoms_end_index = current_chunk.index(atoms_end_line)-3
		
		# Misc Facts
		temperature_line = [ line for line in current_chunk if 'temperature' in line][0]
		dyn_line = [line for line in current_chunk if 'Entering Dynamics' in line][0]
		dyn_index = current_chunk.index(dyn_line)
		time_index = dyn_index+1
		
		# Parse through said regions of interest to get the information out
		
		forces = []
		for line in current_chunk[force_start_index:force_end_index+1]:
			forceline= line.split('=')[-1].split()
			forces.append([float(forceline[0]),float(forceline[1]),float(forceline[2])])
		total_force = float(force_end_line.split('=')[1].strip().split()[0])
		SCF_corr    = float(force_end_line.split('=')[2].strip()[0])
		
		
		positions =[]
		elements=[]
		for line in current_chunk[atoms_start_index:atoms_end_index+1]:
			atomline = line.split()
			elements.append(atomline[0])
			positions.append([float(atomline[1]),float(atomline[2]),float(atomline[3])])
		
		# Get Misc info 
		toten = float(current_chunk[0].split('=')[-1].strip().split()[0])
		temperature_line = temperature_line.split('=')[-1]
		temperature = float(temperature_line.split()[0])
		iteration = int(dyn_line.split('=')[-1])
		timeline = current_chunk[time_index].split('=')[-1].strip().split()[0]
		time = float( timeline)
		Ekin = float(atoms_end_line.split('=')[1].strip().split()[0])
		

		# Record the data associated with this step
		steps[iteration]={'iteration':iteration,
						   'forces':forces, 
						   'positions':positions,
						   'elements':elements,
						   'temperature':temperature,
						   'time':time,
						   'energy':toten,
						   'ekin':Ekin,
						   'kinetic energy':Ekin,
						   'total energy':toten,
						   'total force':total_force,
						   'SCF correction':SCF_corr}
		
	return(steps)


class PW_PP_inparam(Param):
	"""
	Data class containing parameters for a Quantum Espresso PWSCF post-processing calculation.
	"""
	pass


# THIS IS UNFINISHED
def run_qe_pp( runpath,  params,plot_vecs,   ncpu=1):
	pp_code = ExternalCode({'path': os.environ['PP_COMMAND']})
	prepare_dir(runpath.path)
	infile = write_pp_input(params=params, plot_vecs=plot_vecs)
	outfile = File({'path': os.path.join(runpath.path, 'pwscf.out')})
	pp_command = "mpirun -np {} {} < {} > {}".format(ncpu, pp_code.path, infile.path, outfile.path)
	run_command(pwscf_command)
	return outfile



#TODO: MAKE THIS CONNECT TO PWSCF RUN OBJECTS.
# THIS IS ALSO UNFINISHED
def write_pp_input(runpath, params, kpoints, pseudopots, constraint=None):
	"""Make input param string for PW"""
	# automatically fill in missing values
	pcont = params.content
	# Write the main input block
	inptxt = ''
	for namelist in ['inputpp', 'plot']:
		inptxt += '&{}\n'.format(namelist)
		for key, value in pcont[namelist].items():
			inptxt += '	{} = {}\n'.format(key, qe_value_map(value))
		inptxt += '/ \n'

	infile = TextFile(path=os.path.join(runpath.path, 'pwscf.in'), text=inptxt)
	infile.write()
	return infile