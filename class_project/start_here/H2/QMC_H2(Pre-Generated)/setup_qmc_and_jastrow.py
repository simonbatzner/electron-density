import numpy as np
import os

import time


import xml.etree.ElementTree as ET


input_template="""<?xml version ="1.0"?>

<simulation>
    <project id="H2" series="0">
            <application name="qmcapp" role="molecu" class="serial" version="1.0"/>
                </project>

                    <include href="H2.ptcl.xml"/>
                    <include href="H2.wfs.xml"/>
                    <!--include href="H2.pwscf.h5"/-->



                    <hamiltonian target="e">
                    <pairpot name="ElecElec" type="coulomb" source="e" target="e"/>
                    <pairpot name="IonIon" type="coulomb" source="ion0" target="ion0"/>
                    <pairpot name="PseudoPot" type="pseudo" source="ion0" wavefunction="psi0" format="xml"> 
                    <pseudo elementType="H" href="H.BFD.xml"/>
                    </pairpot>                
        </hamiltonian>

<loop max="3">
    <qmc method="linear" move="pbyp" gpu="no">
        <cost name="energy"> 0.2 </cost>
          <cost name="unweightedvariance"> 0.8  </cost>
           <cost name="reweightedvariance"> 0.0 </cost>    
           <estimator name="LocalEnergy" hdf5="no"/>
           <parameter name="walkers"> 1 </parameter>
           <parameter name="samplesperthread"> 25000 </parameter>
           <parameter name="stepsbetweensamples"> 1 </parameter>
           <parameter name="substeps"> 5 </parameter>
           <parameter name="warmupSteps"> 1000 </parameter>
           <parameter name="blocks"> 800 </parameter>
           <parameter name="timestep"> .31 </parameter>
           <parameter name="usedrift"> no </parameter>
           <parameter name="useBuffer"> yes </parameter>                                                                                                                                                                             </qmc>
</loop> 
<qmc method="linear" move="pbyp" gpu="no">
<cost name="energy"> 0.0 </cost>
<cost name="unweightedvariance"> 1.0  </cost>
<cost name="reweightedvariance"> 0.0 </cost>    
<estimator name="LocalEnergy" hdf5="no"/>
<parameter name="walkers"> 1 </parameter>
<parameter name="samplesperthread"> 2500 </parameter>
<parameter name="stepsbetweensamples"> 1 </parameter>
<parameter name="substeps"> 5 </parameter>
<parameter name="warmupSteps"> 10 </parameter>
<parameter name="blocks"> 800 </parameter>
<parameter name="timestep"> .4 </parameter>
<parameter name="usedrift"> no </parameter>
<parameter name="useBuffer"> yes </parameter>
 </qmc>

</simulation>
"""

spacings = np.linspace(.5,1.5,150)
ecut=90
file_names = [ '{}_a_{}_ecut_{}'.format('H2', sp, ecut) for sp in spacings]

work_dir=os.environ['PROJDIR'] + '/data/H2_qmc/QMC_wfs'
for path in file_names:
    qmc_path= work_dir+'/'+path
    
    os.chdir(qmc_path)
    with open("H2.in.xml" ,'w') as f:
        f.write(input_template)

    """ tree = ET.parse('H2.wfs.xml')
    root = tree.getroot()
    print(root)
    for element in root.iter():
        for subelement in element:
            print(element,subelement)
            la = subelement.get('correlation')
            print('la:',la)
            if la is not None and la != lang:
                element.remove(subelement)

    """

    with open("H2.wfs.xml",'r') as f:
        thelines=f.readlines()
    theindex=0

    correlationindices=[]
    
    for n in range(len(thelines)):
        if 'speciesA="u" speciesB="u"' in thelines[n]:
            #print("Found it at",n)
            theindex=n
        if 'size="8"' in thelines[n]:
            correlationindices.append(n)




    if theindex!=0:
        
        for m in correlationindices:
            thelines[m] = thelines[m].strip()
            thelines[m]=thelines[m][:-1]
            thelines[m]+= ' rcut = "9.0" > \n'

        newlines=thelines[:theindex] + thelines[theindex+5:]
        
        

        with open("H2.wfs.xml",'w') as f:
            for x in newlines:
                f.write(x)
    
        print(newlines)

    atomicnumberset=False
    pbcs=True

    correlationindices=[]
    with open("H2.ptcl.xml") as f:
        thelines=f.readlines()
        for n in range(len(thelines)):
            if 'atomicnumber' in thelines[n]:
                atomicnumberset=True
            if 'group name="H"' in thelines[n]:
                theindex1=n
            if 'p p p' in thelines[n]:
                theindex2=n
            if 'n n n ' in thelines[n]:
                pbcs=False
            if 'size="8"' in thelines[n]:
                correlationindices.append(n)


    if pbcs:
        thelines[theindex2]='n n n \n'
        thelines[theindex2+2]=''
        thelines[theindex2+3]=''
        thelines[theindex2+4]=''

        for m in correlationindices:
            thelines[m]=thelines[m].strip()
            thelines[m][-1]=' '
            thelines[m]+= ' rcut="9.0" >'


    if atomicnumberset==False:
        thelines.insert(theindex1+1,'<parameter name="atomicnumber"> 1 </parameter>')

    with open('H2.ptcl.xml','w') as f:
        for x in thelines:
            f.write(x)

    os.system("pwd")
    os.system(os.environ['QMC_COMMAND']+" H2.in.xml")
    time.sleep(1)
