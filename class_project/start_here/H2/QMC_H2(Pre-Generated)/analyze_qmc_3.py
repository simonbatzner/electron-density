import os

import matplotlib.pyplot as plt


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
    
    with open('analyze2.out','r') as f:
        thelines=f.readlines()

spacings=[]
energies=[]
plusminus=[]

N=len(thelines)
for line in thelines:
    if 'a=' in line:
        spacings.append(float(line.split('=')[1]))
    else:
        energies.append(float(line.split()[3]))
        plusminus.append(float(line.split()[5]))


with open('analyze3.out','w') as f:

    for n in range(len(spacings)):
        thestr=str(spacings[n])+','+str(energies[n])+','+str(plusminus[n])+'\n'
        f.write(thestr)


