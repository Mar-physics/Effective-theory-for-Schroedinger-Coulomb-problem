Construction of an Effective Theory for the Schrodinger Equation.

Implementation for the analysis described in https://arxiv.org/abs/nucl-th/9706029. The starting point for these calculations is the file "Schrodinger.py", where one can find how to setup the Hamiltonian for a potential given by the sum of the Coulomb and Yukawa potentials.
The last one has been chosen arbitrarily by the authors, one can substitute it with any short-ranged potential and repeat the analysis. 

The first attempt to approximate such a potential can be found in "DeltaPotential.py", where the short range potential is approximated by a delta function multiplied by a constant, which is fine tuned via First Order Perturbation Theory, according to what described in the cited paper.
No plots are present in this file, as they will be implemented in the "EffectiveTheory.py".

