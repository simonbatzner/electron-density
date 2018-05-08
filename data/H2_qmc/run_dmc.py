import numpy as np
import os

import time

from pathlib import Path

import xml.etree.ElementTree as ET




vmc_input_template="""<?xml version="1.0"?>
<simulation>
   <project id="H2.dmc" series="0">
      <application name="qmcapp" role="molecu" class="serial" version="1.0"/>
   </project>

   <include href="H2.ptcl.xml"/>

   <!-- OPT_XML is from optimization, e.g. O.q0.opt.s008.opt.xml -->
   <include href="H2.s003.opt.xml"/>

   <hamiltonian name="h0" type="generic" target="e">
      <pairpot type="coulomb" name="ElecElec" source="e" target="e"/>
      <pairpot type="coulomb" name="IonIon" source="ion0" target="ion0"/>
      <pairpot type="pseudo" name="PseudoPot" source="ion0" wavefunction="psi0" format="xml">
         <pseudo elementType="H" href="H.BFD.xml"/>
      </pairpot>
   </hamiltonian>

   <!-- fill in VWARMUP, VBLOCKS, VSTEPS, VTIMESTEP, DWALKERS -->

   <loop max="3">
   <qmc method="vmc" move="pbyp">
      <parameter name="walkers"             >    1               </parameter>
      <parameter name="warmupSteps"         >    10         </parameter>
      <parameter name="blocks"              >    1000         </parameter>
      <parameter name="samplesperthread"               >    100000          </parameter>
      <parameter name="timestep"            >    2.0       </parameter>
   </qmc>
   </loop>

   <qmc method="vmc" move="pbyp">
      <parameter name="walkers"             >    1               </parameter>
      <parameter name="warmupSteps"         >    10         </parameter>
      <parameter name="blocks"              >    1000         </parameter>
      <parameter name="samplesperthread"               >    10          </parameter>
      <parameter name="timestep"            >    2.0       </parameter>
   </qmc>


   <!-- fill in DWARMUP, DBLOCKS, DSTEPS, DTIMESTEP -->
   <!-- make multiple copies w/ different timesteps (largest first) -->
   <!--  (blocks*steps*timestep = const1, warmup*timestep = const2) -->
<qmc method="dmc" move="pbyp">
      <parameter name="targetwalkers"       >   200          </parameter>	
      <parameter name="warmupSteps"         >    100        </parameter>
      <parameter name="blocks"              >    400        </parameter>
      <parameter name="steps"               >     20        </parameter>
      <parameter name="timestep"            >   0.04        </parameter>
      <parameter name="nonlocalmoves"       >    yes        </parameter>
   </qmc>
   <qmc method="dmc" move="pbyp">
      <parameter name="warmupSteps"         >     40        </parameter>
      <parameter name="blocks"              >    400        </parameter>
      <parameter name="steps"               >      8        </parameter>
      <parameter name="timestep"            >   0.02        </parameter>
      <parameter name="nonlocalmoves"       >    yes        </parameter>
   </qmc>
   <qmc method="dmc" move="pbyp">
      <parameter name="warmupSteps"         >     80        </parameter>
      <parameter name="blocks"              >    400        </parameter>
      <parameter name="steps"               >     16        </parameter>
      <parameter name="timestep"            >   0.01        </parameter>
      <parameter name="nonlocalmoves"       >    yes        </parameter>
   </qmc>
   <qmc method="dmc" move="pbyp">
      <parameter name="warmupSteps"         >    160        </parameter>
      <parameter name="blocks"              >    400        </parameter>
      <parameter name="steps"               >     32        </parameter>
      <parameter name="timestep"            >   0.005       </parameter>
      <parameter name="nonlocalmoves"       >    yes        </parameter>
   </qmc>
   <qmc method="dmc" move="pbyp">
      <parameter name="warmupSteps"         >    320        </parameter>
      <parameter name="blocks"              >    400        </parameter>
      <parameter name="steps"               >     64        </parameter>
      <parameter name="timestep"            >   0.0025      </parameter>
      <parameter name="nonlocalmoves"       >    yes        </parameter>
   </qmc>





</simulation>


"""




spacings = np.linspace(.5,1.5,150)
ecut=90
file_names = [ '{}_a_{}_ecut_{}'.format('H2', sp, ecut) for sp in spacings]


work_dir=os.environ['PROJDIR'] + '/data/H2_qmc/QMC_wfs'
print(file_names)
for path in reversed(file_names):
    qmc_path= work_dir+'/'+path
    
    os.chdir(qmc_path)
    with open("H2.dmc.in.xml" ,'w') as f:
        f.write(vmc_input_template)


    print("Now running", qmc_path)
    os.system("pwd")
    os.system(os.environ['QMC_COMMAND']+" H2.dmc.in.xml")
    time.sleep(5)
