import numpy as np
import sys
import os
import time

spacings = np.linspace(.5,1.5,150)
ecut=90
file_names = [ '{}_a_{}_ecut_{}'.format('H2', sp, ecut) for sp in spacings]


def parse_jastrow_file(path):
    with open(path,'r') as f:
        thelines = f.readlines()
    
    jastsize = len(thelines)
    jast_array = np.empty((jastsize,4))
    counter = 0
    for line in thelines:
        line = line.split()
        for n in range(4):
            jast_array[counter][n]=float(line[n])
        counter+=1

    return jast_array


if __name__=='__main__':

    work_dir=os.environ['PROJDIR'] + '/data/H2_qmc/QMC_wfs'
    
    output='analyze.out'

    os.system("echo  > "+work_dir+"/analyze.out")
    for path in file_names:
        qmc_path=work_dir + '/' + path
        os.chdir(qmc_path)
        print('Doing'+qmc_path)
        print('printing to '+work_dir+'/analyze.out')
        os.system("echo " + 'a='+str(path.split('_')[2])+' >>'+work_dir+'/analyze.out')
        os.system('qmca -o -q ev -u eV *scalar* >>'+work_dir+'/analyze.out')
        #os.system('qmca -o -q ar *scalar* >> analyze.out')
        # The model: >>../job.out


    os.chdir(work_dir)
    
    with open('analyze.out','r') as f:
        thelines = f.readlines()
    topop=[]
    for n in range(len(thelines)):
        if 'Local' in thelines[n]:
            topop.append(n)
        if len(thelines[n])<6:
            topop.append(n)

    for m in reversed(topop):
        thelines.pop(m)
    with open('analyze2.out','w') as f:
        for line in thelines:
            f.write(line.strip())
    
    
    
    
    with open('analyze2.out','r') as f:
        spacings=[]
        energies=[]
        for line in thelines:
            if 'a=' in line:
                spacings.append(float(line.split('a=')[-1].strip()))

            if 'series 8' in line and 'dmc' in line and len(line.split())>9:
                energies.append(float(line.split()[3]))



    
    with open('energies.out','w') as f:
        for n in range(len(spacings)):
            f.write(str(spacings[n]) +','+str(energies[n])+'\n')


