
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy.random as rand
import numpy.linalg as la
from sklearn.kernel_ridge import KernelRidge

from parse import load_config_yaml, QE_Config, Structure_Config, ml_config,MD_Config
from utility import first_derivative_2nd, first_derivative_4th
mass_dict = {'H': 1.0, "Al": 26.981539, "Si": 28.0855,'O':15.9994}
from parse import *




class MD_Engine(MD_Config):
    def __init__(self, structure, md_config,qe_config,ml_model,hpc_config=None):
        """
        Initialize the features of the system, which include:
        input_atoms: Atoms in the unit cell, list of objects of type Atom (defined later on in this notebook)
        cell:  3x3 matrix [[a11 a12 a13],..] which define the dimensions of the unit cell.
        dx:    Perturbation distance used to evaluate the forces by finite differences in potential energy.
        verbosity: integer from 0 to 5 which determines how much information will be printed about runs.
                    Ranging from 0 as silent to 5 is TMI and is mostly for debugging.
        model: The energy model used.
        store_trajectory: Boolean which determines if positions will be saved after every time step.
        qe_config: Espresso_config object which determines how on-the-fly quantum espresso runs will be
                        parameteried.
        thermo_config:   Thermo config object which determines the thermostate methodology that will be used
                        for molecular dynamics.
        ML_model: Object which parameterizes a ML model which returns energies when certain configurations are input.
        assert_boundaries : Determines if boundary conditions are asserted at the end of each position update
                            (which forces atoms to lie within the unit cell).
        fd_accuracy: Integer 2 or 4. Determines if the finite difference force calculation will be evaluated to
                        second or fourth order in dx. Note that this increases the number of energy calculations
                        and thus increases run time.
        uncertainty_threshold: Defines an uncertainty cutoff above which a DFT run will be called.
        energy_or_force_driven: Maps to 'driver': dictates if the finite difference energy model drives the simulation
                                or if the force comes directly from the model (such as with AIMD.)
        """

        # Set configurations
        super(MD_Engine,self).__init__(md_config)
        self.structure = structure
        self.qe_config = qe_config
        self.ml_model = ml_model

    def get_energy(self):
        """
        Uses:

        self.ml_model
        self.Structure
        :return: float Energy of configuration
        """

        if self['mode']=='ML':
            return self.ml_model.get_energy(self.structure)

        if self['mode']=='AIMD':
            result = self.qe_config.run_espresso()
            return result['energy']
        if self['mode']=='LJ':

            eps = self['LJ_eps']
            rm =  self['LJ_rm']
            E = 0.
            for at in self.structure:
                for at2 in self.structure:
                    if at.fingerprint != at2.fingerprint:
                        disp = la.norm(at.position - at2.position)
                        if self['verbosity']>=4:
                            print('Current LJ disp between atoms is:', disp)

                        E += .5 * eps * ((rm / disp) ** 12 - 2 * (rm / disp) ** 6)
            return E


    def set_forces(self):
        if self['mode']=='AIMD':
                results = self.qe_config.run_espresso(self.structure)

                if self.verbosity == 4: print("E0:", results['energy'])

                forces = results['forces']
                for n, at in enumerate(self.structure):
                    at.force = list(np.array(results['forces'][n]) * 13.6 / 0.529177)

                return

        elif self.mode=='LJ':
            self.set_fd_forces()
            pass

        elif self.md_config['mode']=='ML':
            if self.ml_model.type=='energy':
                self.set_fd_forces()
            elif self.ml_model.type=='force':
                forces= self.ml_model.get_forces(self.structure)
                for n, at in enumerate(self.structure):
                    at.force = list(np.array(forces[n]) * 13.6 / 0.529177)
                return

    def set_fd_forces(self):
        """
        Perturbs the atoms by a small amount dx in each direction
        and returns the gradient (and thus the force)
        via a finite-difference approximation.

        Or, if the engine's model is ab-initio molecular dynamics,
        runs Quantum ESPRESSO on the current configuration and then
        stores the forces output from ESPRESSO as the atomic forces.
        """

        dx = self.fd_dx
        fd_accuracy = self['fd_accuracy']

        if self.energy_or_force_driven == "energy" and self.md_config.mode == "AIMD":
            print("WARNING! You are asking for an energy driven model with Ab-Initio MD;",
                  " AIMD is configured only to work on a force-driven basis.")
        """
        
        """

        E0 = self.get_energy()
        if self.verbosity == 4: print('E0:', E0)

        # Main loop of setting forces
        for atom in self.structure:
            for coord in range(3):
                # Perturb to x + dx
                atom.position[coord] += dx
                Eplus = self.get_energy()

                # Perturb to x - dx
                atom.position[coord] -= 2 * dx
                Eminus = self.get_energy()

                if fd_accuracy==4:
                    # Perturb to x + 2dx
                    atom.position[coord] += 3* dx
                    Epp = self.get_energy()

                    # Perturb to x - 2dx
                    atom.position[coord] -= 4* dx
                    Emm = self.get_energy()

                    # Perturb to x - dx
                    atom.position[coord] += dx

                # Return atom to initial position
                atom.position[coord] += dx

                if fd_accuracy==2:
                    atom.force[coord] = -first_derivative_2nd(Eminus,Eplus,dx)
                elif fd_accuracy==4:
                    atom.force[coord] = -first_derivative_4th(Emm,Eminus, Eplus,Epp, dx)
                if self.verbosity == 5:
                    print("Just assigned force on atom's coordinate", coord, " to be ",
                          -first_derivative_2nd(Eminus, Eplus, self.dx))

        for at in self.structure:
            at.apply_constraint()

    #TODO
    def take_timestep(self,dt = None,method = None):
        """
        Propagate the atoms forward by timestep dt according to it's current force and previous position.
        Note that the first time step does not have the benefit of a previous position, so, we use
        a standard third-order Euler timestep incorporating information about the velocity and the force.
        """

        self.set_forces()
        self.structure.record_trajectory(time=self.dt,force=True)

        if self['verbosity'] >= 3:
            print(self.get_report(forces=True))

        dt = dt or self.dt
        dtdt = dt*dt

        method = method or self["timestep_method"]
        self['time'] += dt
        self.frame += 1

        # Third-order euler method
        # Is a suggested way to begin a Verlet run, see:
        # https://en.wikipedia.org/wiki/Verlet_integration#Starting_the_iteration

        if method == 'TO_Euler':
            for atom in self.structure:
                for i in range(3):
                    atom.prev_pos[i] = np.copy(atom.position[i])
                    atom.position[i] += atom.velocity[i] * dt + atom.force[i] * dtdt*.5 / atom.mass
                    atom.velocity[i] += atom.force[i] * dt / atom.mass

        ######
        # Superior Verlet integration
        # Citation:  https://en.wikipedia.org/wiki/Verlet_integration
        ######

        elif method == 'Verlet':
            for atom in self.structure:
                for i in range(3):
                    # Store the current position to later store as the previous position
                    # After using it to update the position

                    temp_coord = np.copy(atom.position[i])
                    atom.position[i] = 2 * atom.position[i] - atom.prev_pos[i] + \
                                       atom.force[i] * dtdt*.5 /atom.mass
                    atom.velocity[i] += atom.force[i] * dt / atom.mass
                    atom.prev_pos[i] = np.copy(temp_coord)
                    if self.verbosity == 5: print("Just propagated a distance of ",
                                                  atom.position[i] - atom.prev_pos[i])

        if self['assert_boundaries']:
            self.assert_boundary_conditions()
        self.structure.record_trajectory(time=self.dt,position=True)


    #TODO
    def setup_run(self):

        """
        Checks a few classes within the 5 parameters before run begins.
        :return:
        """


    #TODO
    def run(self):
        """
        Handles timestepping; at each step, calculates the force and then
        advances via the take_timestep method.
        """

        if self['time'] == 0:
            self.take_timestep(method='TO_Euler')

        # Primary iteration loop
        # Most details are handled in the timestep function
        while (self.time < self.get('tf',np.inf) and self['frame']<self.get('frames',np.inf)):

            self.take_timestep(method=self['timestep_method'])

            if (self.mode == 'ML'):

                if self.gauge_model_uncertainty():
                    continue
                else:
                    # print("Timestep with unacceptable uncertainty detected! \n Rewinding one step, and calling espresso to re-train the model.")
                    self.qe_config.run_espresso(self.atoms, self.cell,
                                                iscorrection=True)
                    self.retrain_ml_model(self.model, self.ML_model)
                    self.take_timestep(dt=-dt)

        self.end_run()

    #TODO: Implement end-run report
    def end_run(self):
        pass

    def get_report(self, forces=True, velocities=False):
        """
        Prints out the current positions of the atoms line-by-line,
        and also the forces
        """
        report ='Frame #:{},Time:{}\n'.format(self['frame'],np.round(self['time'],3))
        report += 'Atom,Element,Position'
        report += ',Force' if forces else ''
        report += ',Velocity'if velocities else ''
        report += '\n'
        for n, at in enumerate(self.structure):


            report+= '{},{},{}{}{} \n'.format(n,at.element,str(tuple([np.round(x,4) for x in at.position])),
                     ','+str(tuple([np.round(v,4) for v in at.velocity])) if velocities else '',
                     ',' + str(tuple([np.round(f, 4) for f in at.force])) if forces else '')

        return report


def main():
    pass

if __name__ == '__main__':
    main()





config = load_config_yaml('H2_test.yaml')
print(config)
qe_conf = QE_Config(config['qe_params'], warn=True)
structure = Structure_Config(config['structure_params']).to_structure()
ml_fig = ml_config(params=config['ml_params'], print_warn=True)
md_fig = MD_Config(params=config['md_params'], warn=True)

a = MD_Engine(structure, md_fig, qe_conf, ml_fig)

print(structure)

a.run()
