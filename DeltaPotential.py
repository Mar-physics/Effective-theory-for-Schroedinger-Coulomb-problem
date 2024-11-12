import numpy as np
from Schrodinger import RadialHydrogenSolver


class DeltaPotentialSolver(RadialHydrogenSolver):
    def __init__(self, alpha, m, r_min, r_max, N, c=0, l=0):
        # Call the constructor of the parent class and initialize parameters with the same values
        super().__init__(alpha, m, r_min, r_max, N, l) 
        self.c = c  # Parameter for the delta potential

    def approximate_energy_levels(self):
        """ Calculates the approximate energy levels using the first-order perturbation model. """
        n_values = np.arange(1, 21)  # Array of values from 1 to 20
        energies_approx = []
        for n in n_values:
            # Calculate the Coulomb energy for level n
            energy_coul = - 1 / (2 * n**2)

            # Calculate the contribution from the delta potential only for l = 0
            if self.l == 0:
                correction = self.c / (np.sqrt(np.pi) * n**3)
            else:
                correction = 0

            # Total approximate energy
            energy_approx = energy_coul + correction
            energies_approx.append(energy_approx)

        return energies_approx
    
    @staticmethod    
    def find_c_from_energy(eigenvalues, n=20):
        """ Calculates the value of c based on the target energy value and the level n. """
        energy_coul = - 1 / (2 * n**2)
        c = (eigenvalues[19] - energy_coul) * (np.sqrt(np.pi) * n**3)
        return c
    
    def set_c(self, eigenvalues, n=20):
        """ Sets the value of c based on the given eigenvalues and level n. """
        self.c = self.find_c_from_energy(eigenvalues, n)



""" TESTING CODE FOR VERIFICATION (REMOVE BEFORE FINAL USE) """

solver = RadialHydrogenSolver(alpha=1, g=1, m=1, r_min=0.1, r_max=1000.0, N=10000, l=0)

solver2 = DeltaPotentialSolver(alpha=1, m=1, r_min=0.1, r_max=1000.0, N=10000, l=0)

eigenvalues, wavefunctions = solver.solve_hamiltonian(40)

solver2.set_c(eigenvalues)

print(f"Valore di c calcolato: {solver2.c}")

