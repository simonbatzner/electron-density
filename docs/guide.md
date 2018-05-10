
# Machine Learning Ground State Electron Densities
## A project by S. Batzner, S. Torrisi, and J. Vandermause


## Project Overview

We present a comparative study in which we benchmark various machine learning-based regression frameworks to learning the ground-state electron energy and density of a H2 dimer and an Aluminum crystal. This extends the work of Brockherde _et al_ to the Solid State and explores the possibiliy of a potential-energy mapping across other ML frameworks.

We present our work across datasets generated in Quantum ESPRESSO and one generated from the Quantum Monte Carlo code QMCPACK. Included in this work is a library of functions and objects to assist in the automated generation and analysis of these data sets. The exposition proceeds through a series of Jupyter Notebooks, so that the results of our study are easily replicable.


## Workflow Roadmap

Our workflow proceeds as follows.

I. Data Generation.

1. Generate a configurational dataset for the system of interest.
  a. For H2, which has a one-dimensional configuration space, we simply use 150 different interatomic distances obtained via linearly spaced values between \[.5,1.5\].
  b. For Al, which is a periodic solid, we run a LAMMPS Molecular Dynamics simulation for a long duration, printing the positions at periodic intervals. 
2. Load the configurations into a Quantum ESPRESSO input file, then run the Density-Functional Theory calculation to optimize the electron density and obtain the ground-state energy.
3. Run ESPRESSO's Post-Processing module to print the optimized density into a parsable text file.

II. Machine Learning.
4. Run a regression model of your choosing (like Kernel Ridge Regression, or Gaussian Processes) on the configuration->energy mapping. This is known as the Kohn-Sham mapping.
5. Run a regression model of your choosing on the density-> energy mapping. This is known as the Hohenberg-Kohn mapping.
6. Compare!


The [/start_here](https://github.com/simonbatzner/ML-electron-density/tree/master/start_here) folder in our repository is organized first by material, and then by method. It is more instructive to view the H2 system first because of the simple process of generating the configuration.


## Overview of Code

Our code has the following prerequisites. To generate the data sets, you must have the following on your local machine or available on a supercomputer cluster:

- Python 3.6 (with Numpy, Scipy, Matplotlib)
- Quantum ESPRESSO (for H2)
- LAMMPS (for Al)

Python can be obtained from a variety of sources; a distribution like Anaconda should come with Numpy, Scipy, and Matplotlib. Instructions for installing ESPRESSO and LAMMPS are available on their respective websites. 

We also make use of the following Python libraries and tools for our analysis:

- Sklearn
- PyMC
- Jupyter Notebooks


To facilitate the generation of the datasets and the analysis, we have developed a suite of helper objects and functions in the dataflow programming paradigm which are included in the [/util](https://github.com/simonbatzner/ML-electron-density/tree/master/util) folder (we suggest that you add this folder to your local Python path while using this code). The PWSCF suite of functions are designed to facilitate Quantum ESPRESSO calculations in a highly modular way; the PP suite of functions, for the use of the ESPRESSO postprocessing module. We also have code to assist the LAMMPS calculation. All of these functions are included in the [\util](https://github.com/simonbatzner/ML-electron-density/tree/master/util) folder in the repo.

We present an exposition of our analysis in iPython notebooks throghout the Al and H2 subdirectories. Each one is labelled according to the regression method that it attempts (corresponding to steps 4 and 5 of our workflow).

## Individual folders 


The PWSCF set of objects is from Prof. Boris Kozisnky, the instructor for Applied Physics 275 (the course that this Repo began as a term project for). We developed a set of functions in order to facilitate the calculations. They are included in the util folder.

Interested readers should visit the GitHub repo of our project; the 'start_here' folder includes various examples.
