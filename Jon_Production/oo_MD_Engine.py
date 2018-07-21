
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy.random as rand
import numpy.linalg as la
from sklearn.kernel_ridge import KernelRidge

from utility import first_derivative_2nd, first_derivative_4th
from utility import
mass_dict = {'H': 1.0, "Al": 26.981539, "Si": 28.0855,'O':15.9994}


class MD_Engine(object):
    def __init__(self, structure, md_config,qe_config,ml_model,hpc_config):
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
        self.verbosity = md_config.get('verbosity',1)

        self.dx = md_config.get('fd_dx',.1)
        self.structure = structure
        # self.ml_model = model

        # Set configurations
        self.qe_config = qe_config
        self.ml_model = ml_model
        self.md_config = md_config
        self.assert_boundaries = md_config.get('assert_boundaries',False)

        self.fd_accuracy = None or md_config['fd_accuracy']

        self.energy_or_force_driven = self.ml_model.energy_or_force

    def get_energy(self):
        """
        Uses:

        self.ml_model
        self.Structure
        :return: float Energy of configuration
        """
        return self.ml_model.get_energy(self.structure)


    def set_fd_forces(self):
        """
        Perturbs the atoms by a small amount dx in each direction
        and returns the gradient (and thus the force)
        via a finite-difference approximation.

        Or, if the engine's model is ab-initio molecular dynamics,
        runs Quantum ESPRESSO on the current configuration and then
        stores the forces output from ESPRESSO as the atomic forces.
        """

        dx = self.md_config['fd_dx']
        fd_accuracy = self.md_config['fd_accuracy']

        if self.energy_or_force_driven == "energy" and self.md_config.mode == "AIMD":
            print("WARNING! You are asking for an energy driven model with Ab-Initio MD;",
                  " AIMD is configured only to work on a force-driven basis.")
        """
        if self.md_config['mode'] == 'AIMD':
            results = qe_config.run_espresso(self.structure)

            if self.verbosity == 4: print("E0:", results['energy'])

            force_list = results['forces']
            for n, at in enumerate(self.structure):
                at.force = list(np.array(results['forces'][n]) * 13.6 / 0.529177)

            return
        """




        E0 = self.get_energy(); if self.verbosity == 4: print('E0:', E0)

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

        self.update_atom_forces()

        dt = dt or self.md_config['dt']
        dtdt = dt*dt
        method = method or self.md_config["timestep_method"]
        self.time += dt

        temp_num = 0.

        # Third-order euler method
        # Is a suggested way to begin a Verlet run, see:
        # https://en.wikipedia.org/wiki/Verlet_integration#Starting_the_iteration

        if method == 'TO_Euler':
            for atom in self.Structure.atom_list:
                for coord in range(3):
                    atom.prev_pos[coord] = np.copy(atom.position[coord])
                    atom.position[coord] += atom.velocity[coord] * dt + atom.force[coord] * dt ** 2 / atom.mass
                    atom.velocity[coord] += atom.force[coord] * dt / atom.mass

        ######
        # Superior Verlet integration
        # Citation:  https://en.wikipedia.org/wiki/Verlet_integration
        ######

        elif method == 'Verlet':
            for atom in self.Structure.atom_list:
                for coord in range(3):
                    # Store the current position to later store as the previous position
                    # After using it to update the position

                    temp_num = np.copy(atom.position[coord])
                    atom.position[coord] = 2 * atom.position[coord] - atom.prev_pos[coord] + atom.force[
                                                                                                 coord] * dtdt /(2* atom.mass)
                    atom.velocity[coord] += atom.force[coord] * dt / atom.mass
                    atom.prev_pos[coord] = np.copy(temp_num)
                    if self.verbosity == 5: print("Just propagated a distance of ",
                                                  atom.position[coord] - atom.prev_pos[coord])

        if self.assert_boundaries: self.assert_boundary_conditions()
        if self.verbosity > 3: self.print_positions()

    #TODO
    def setup_run(self):

        """
        Checks a few classes within the 5 parameters before run begins.
        :return:
        """


    #TODO
    def run(self, tf, dt, ti=0):
        """
        Handles timestepping; at each step, calculates the force and then
        advances via the take_timestep method.
        """

        # The very first timestep often doesn't have the 'previous position' to use,
        # so the third-order Euler method starts us off using already-present
        # information about the position, velocity (if provided) and force:

        if self.time == 0:
            self.take_timestep(dt=dt, method='TO_Euler')

        # Primary iteration loop
        # Most details are handled in the timestep function
        while self.time < tf:

            self.take_timestep(dt=dt)

            if (self.model == 'GP' or self.model == 'KRR') and self.uncertainty_threshold > 0:

                if self.gauge_model_uncertainty():
                    continue
                else:
                    # print("Timestep with unacceptable uncertainty detected! \n Rewinding one step, and calling espresso to re-train the model.")
                    self.qe_config.run_espresso(self.atoms, self.cell,
                                                iscorrection=True)
                    self.retrain_ml_model(self.model, self.ML_model)
                    self.take_timestep(dt=-dt)

    #TODO
    def assert_boundary_conditions(self):
        """
        We seek to have our atoms be entirely within the unit cell, that is,
        for bravais lattice vectors a1 a2 and a3, we want our atoms to be at positions

         x= a a1 + b a2 + c a3
         where a, b, c in [0,1)

         So in order to impose this condition, we invert the matrix equation such that

          [a11 a12 a13] [a] = [x1]
          [a21 a22 a23] [b] = [x2]
          [a31 a32 a33] [c] = [x3]

          And if we find that a, b, c not in [0,1)e modulo by 1.
        """
        a1 = self.cell[0]
        a2 = self.cell[1]
        a3 = self.cell[2]

        for atom in self.atoms:

            coords = np.dot(la.inv(self.cell), atom.position)
            if self.verbosity == 5:
                print('Atom positions before BC check:', atom.position)
                print('Resultant coords:', coords)

            if any([coord > 1.0 for coord in coords]) or any([coord < 0.0 for coord in coords]):
                if self.verbosity == 4: print('Atom positions before BC check:', atom.position)

                atom.position = a1 * (coords[0] % 1) + a2 * (coords[1] % 1) + a3 * (coords[2] % 1)
                if self.verbosity == 4: print("BC updated position:", atom.position)

    #TODO
    def print_positions(self, forces=True):
        """
        Prints out the current positions of the atoms line-by-line,
        and also the forces
        """
        print("T=", self.time)
        for n in range(len(self.atoms)):

            pos = self.atoms[n].position
            force = self.atoms[n].force
            if forces:
                print('Atom %d:' % n, np.round(pos, decimals=4), ' Force:', np.round(force, decimals=6))
            else:
                print('Atom %d:' % n, np.round(pos, decimals=4))
