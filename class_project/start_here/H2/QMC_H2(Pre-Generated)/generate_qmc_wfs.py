import sys
import os
from shutil import copyfile

sys.path.append('../../util')

import numpy as np
import matplotlib.pyplot as plt

pw2qmc_template="""&inputpp
prefix='H2'
outdir='./'
write_psir=.true.
/
"""

spacings = np.linspace(.5,1.5,150)
ecut = 90

print(os.environ['PROJDIR'])

head_dir = os.environ['PROJDIR'] + '/data/H2_qmc/'

dft_dir  = os.environ['PROJDIR'] + '/data/H2_DFT/temp_data/'
work_dir = os.environ['PROJDIR'] + '/data/H2_qmc/QMC_wfs/'

file_names = [ '{}_a_{}_ecut_{}'.format('H2', sp, ecut) for sp in spacings]

for path in file_names:
    dft_path = dft_dir+path
    qmc_path = work_dir+path

    try:
        os.mkdir(work_dir+path)
    except:
        pass
    os.chdir(dft_dir+path)

    #copyfile(dft_path,work_path)
    
    with open("pw2qmc.pp",'w') as f:
        f.write(pw2qmc_template)

    print(os.environ['PW2QMC_COMMAND'])
    os.system( os.environ['PW2QMC_COMMAND']+ '< pw2qmc.pp')
    copyfile(dft_path+'/H2.ptcl.xml', qmc_path+'/H2.ptcl.xml')
    copyfile(dft_path+'/H2.pwscf.h5', qmc_path+'/H2.pwscf.h5')
    copyfile(dft_path+'/H2.wfc1'    , qmc_path+'/H2.wfc1')
    copyfile(dft_path+'/H2.wfs.xml' , qmc_path+'/H2.wfs.xml')
    copyfile(head_dir+'/H.BFD.xml'  , qmc_path+'/H.BFD.xml')

