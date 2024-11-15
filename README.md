This repository provides an implementation of the analysis described in arXiv:nucl-th/9706029. The project explores the construction of an effective theory for the Schrödinger equation, focusing on a potential composed of a Coulomb term and an arbitrary short-range potential.

**Overview:**  
The starting point for these calculations is the file Schrodinger.py, which demonstrates how to set up the Hamiltonian for a potential defined as the sum of:
- A Coulomb potential.
- A short-range potential, modeled in this implementation using a Yukawa potential.
The Yukawa potential has been chosen arbitrarily by the authors and can be replaced with any short-range potential to repeat or adapt the analysis.

**Files Description**
- "Schrodinger.py"
Contains the primary implementation for setting up and solving the Schrödinger equation for the given potential. This file is the foundation of the analysis and includes tools for defining the Hamiltonian, as well as plots to visualize eigenfunctions and potentials behavior

- "DeltaPotential.py"
Introduces a naive attempt to approximate the short-range potential using a delta function, scaled by a constant. This constant is fine-tuned using First Order Perturbation Theory, as described in the cited paper.
Note: This file does not produce plots; it focuses solely on theoretical approximations.

- "EffectiveTheory.py"
Implements the construction of the effective theory, incorporating second and fourth-order correction terms derived from the system's symmetries. Plots for visualizing results are generated in this file.  
**Status: Work in progress (many functions are still missing).**
