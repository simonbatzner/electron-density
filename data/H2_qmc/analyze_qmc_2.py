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
 
    newlines=[]
    with open('analyze.out','r') as f:
        thelines = f.readlines()
    
    for line in thelines:
        if 'a=' in line:
            newlines.append(line)

        if 'dmc' in line and 'series' in line:
            newlines.append(line)
    with open('analyze2.out','w') as f:
        for line in newlines:
            if 'a=' in line:
                f.write(line)
            if 'series 8' in line:
                f.write(line)

    """topop=[]
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
        thelines=f.readlines()
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


    with open('energies.out','r') as f:
        thelines= f.readlines()

    spacings=[]
    energies=[]

    for x in thelines:
        energies.append(float(x.split(',')[1]))
        spacings.append(float(x.split(',')[0]))

    print(spacings,energies)
    plt.figure()
    plt.plot(spacings,energies)
    plt.show()
    plt.savefig('dmc_energy.png')

    """
