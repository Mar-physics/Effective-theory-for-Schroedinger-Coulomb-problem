import numpy as np
from scipy.linalg import eigh
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

class RadialHydrogenSolver:
    def __init__(self, alpha, g, m, r_min, r_max, N, l=0):
        self.alpha = alpha
        self.g = g
        self.m = m
        self.r_min = r_min
        self.r_max = r_max
        self.N = N
        self.l = l
        self.r = np.linspace(r_min, r_max, N)
        self.dr = (r_max - r_min) / (N-1)
        
    def coulomb_potential(self):
        V_c = - self.alpha / self.r
        return V_c
    
    def yukawa_potential(self):
        V_y = - (self.g / self.r) * np.exp(- self.m * self.r)
        return V_y
    
    def total_potential(self):
        V_c = self.coulomb_potential()
        V_y = self.yukawa_potential()
        V_tot = V_c + V_y
        return V_tot
    
    """ Here we test two different methods to obtain the eigenvalues and eigenfunctions of the Hamiltonian.
        The two methods provide the same results, but the first one is computationally faster.
    """
    
    """ FIRST METHOD """
    def solve_hamiltonian(self, num_eigenvalues=None):
        """
        Calculates and returns the eigenvalues and eigenvectors of the tridiagonal Hamiltonian matrix,
        constructed from the radial differential equation.
        
        Args:
            num_eigenvalues (int, optional): Number of lowest eigenvalues to compute.
                                             If None, all eigenvalues are computed.
                                              
        Returns:
            eigenvalues (array): Array of eigenvalues of the Hamiltonian matrix.
            wavefunctions (array): Matrix whose columns are the corresponding eigenvectors.
        """
    
        # Calculate the effective potential
        V_eff = self.total_potential() + self.l * (self.l + 1) / (2 * self.r ** 2)
    
        # Define the diagonal and off-diagonal of the tridiagonal Hamiltonian matrix
        diagonal = (1 / self.dr**2) + V_eff
        off_diagonal = -1.0 / (2 * self.dr**2) * np.ones(self.N - 1)  # Second-order finite difference for Laplacian
        
        # Compute the eigenvalues and eigenvectors using `eigh_tridiagonal`
        if num_eigenvalues is not None:
            eigenvalues, wavefunctions = eigh_tridiagonal(diagonal, off_diagonal, select='i', select_range=(0, num_eigenvalues - 1))
        else:
            eigenvalues, wavefunctions = eigh_tridiagonal(diagonal, off_diagonal)
                
        return eigenvalues, wavefunctions
    
    
    """ SECOND METHOD """
    
    def hamiltonian_matrix(self):
        """
        Returns:
            Matrix: The Hamiltonian matrix created from the second-order differential equation 
            for the radial eigenfunctions. 
        """
        V_eff = self.total_potential() + self.l * (self.l + 1) / (2 * self.r ** 2)

        # Diagonal and off-diagonal terms using higher-order finite differences
        diag = (1 / self.dr**2) + V_eff[:-1]
        off_diag = (-1.0 / (2 * self.dr**2)) * np.ones(self.N - 2)
        
        # Construct the Hamiltonian matrix with fourth-order finite differences
        H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        return H

    def solve(self, num_eigenvalues=None):
        """
        Calculates and returns the eigenvalues and eigenvectors of the Hamiltonian matrix.

        Args:
            num_eigenvalues (int, optional): Number of lowest eigenvalues to compute.
                                             If None, all eigenvalues are computed.

        Returns:
            eigenvalues, wavefunctions
        """
        H = self.hamiltonian_matrix()

        # Compute specified number of eigenvalues and eigenvectors, or all if not specified
        if num_eigenvalues is not None:
            eigenvalues, wavefunctions = eigh(H, subset_by_index=[0, num_eigenvalues-1])
        else:
            eigenvalues, wavefunctions = eigh(H)
        
        return eigenvalues, wavefunctions
    
    def test_orthonormality(self, wavefunctions, num_wavefunctions = None):
        if num_wavefunctions is None:
            num_wavefunctions = 5
        for j in range(num_wavefunctions):
            for i in range(num_wavefunctions):
                print("{:16.9e}".format(np.sum(wavefunctions.T[j]*wavefunctions.T[i])),end=" ")
            print()
    
    def compute_phase_shift(self, energies, wavefunctions):
        """
        Calculates the phase shift for an array of arbitrary energies by comparing the numerical wavefunctions 
        with the theoretical asymptotic form of the Coulomb potential.
    
        Parameters:
            - wavefunctions: 2D array with values of numerically obtained wavefunctions for various energies.
                             Each row corresponds to a wavefunction for a specific energy.
            - r_values: array of r values.
            - energies: array of energies corresponding to the wavefunctions.
            - alpha: Coulomb constant (Z * e^2 / ħ).

        Returns:
            - phase_shifts: array of phase shifts for each energy.
        """
        r_target = 50  # Distance at which to calculate the phase shift
        idx_r = np.argmin(np.abs(self.r - r_target))  # Find index where r = 50
        phase_shifts = []

        for i, E in enumerate(energies):
            k = np.sqrt(2 * self.m * E)  # Calculate wave number for energy E
            numerical_value = wavefunctions[i, idx_r]  # Numerical wavefunction value at r = 50
            asymptotic_value = np.exp(1j * (k * r_target + np.log(k * r_target) / r_target))
        
            # Calculate the phase shift by comparing imaginary and real parts
            delta_l = np.angle(numerical_value / asymptotic_value)
            phase_shifts.append(delta_l)

        return np.array(phase_shifts)
    
    
    """ ---
               PLOTTING METHODS
                                 ---"""
                             
    def plot_potentials(self):
        V_c = self.coulomb_potential()
        V_y = self.yukawa_potential()
        V_tot = self.total_potential()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.r, V_c, label="Coulomb Potential", color="blue")
        plt.plot(self.r, V_y, label="Yukawa Potential", color="green")
        plt.plot(self.r, V_tot, label="Total Potential", color="red")
        plt.xlabel("r")
        plt.ylabel("V(r)")
        plt.title("Potential Curves")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 10)
        plt.ylim(-3, 0.5)
        plt.show()
    
    def plot_wavefunctions(self, wavefunctions):
        # Plot first few eigenfunctions
        plt.figure(figsize=(10, 6))
        for i in range(min(7, len(wavefunctions.T))):  # Plot first 7 eigenfunctions
            plt.plot(self.r, wavefunctions.T[i], label=f'Wavefunction {i+1}')
        
        plt.xlabel("r")
        plt.ylabel("u(r)")
        plt.title("Radial Eigenfunctions")
        plt.legend()
        plt.grid(True)
        plt.xlim(self.r_min, 100)
        plt.ylim(-0.25, 1)  # Adjust this limit based on your eigenfunctions
        plt.show()



""" TESTING CODE FOR VERIFICATION (REMOVE BEFORE FINAL USE) """
# Create and use the solver
solver = RadialHydrogenSolver(alpha=1, g=1, m=1, r_min=0.1, r_max=200, N=2000, l=0)
eigenvalues, wavefunctions = solver.solve_hamiltonian(40)
solver.plot_wavefunctions(wavefunctions)
solver.plot_potentials()

print("\nConfronto con i valori teorici:")
for n in range(1, len(eigenvalues) + 1):
    E_coulomb = -1 / (2 * n**2)
    print(f"n = {n}, E_num = {eigenvalues[n-1]:.6f}, E_coulomb = {E_coulomb:.6f}")

# Array of energies to calculate the phase shift
energies = np.array([10e-10, 10e-5, 0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1])
phase_shifts = solver.compute_phase_shift(energies)

# Output phase shift
print("Energy    Phase Shift (rad)")
for energy, delta in zip(energies, phase_shifts):
    print(f"{energy:.12f}      {delta:.12f}")

# First 20 calculated energies
print("First 20 calculated energies (in natural units):")
for i, E in enumerate(eigenvalues[:40], start=1):
    print(f"E_{i} = {E:.6f}")



