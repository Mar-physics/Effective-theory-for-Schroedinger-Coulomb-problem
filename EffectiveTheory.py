import numpy as np
from Schrodinger import RadialHydrogenSolver
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import differential_evolution

class EffectiveTheorySolver(RadialHydrogenSolver):
    """Class to implement an effective theory with an ultraviolet cutoff."""
    
    def __init__(self, alpha, g, m, r_min, r_max, N, l=0, cutoff=1.0, c=1, d_1 = 1):
        super().__init__(alpha, m, r_min, r_max, N, l) 
        self.cutoff = cutoff  # UV cutoff
        self.c = c
        self.d_1 = d_1
        self.N = N
        self.r = np.linspace(r_min, r_max, N)  # Radial coordinate
        self.dr = (r_max - r_min) / N  # Radial step size

    def smeared_coulomb_potential(self):
        """Constructs a regulated Coulomb potential with an ultraviolet cutoff."""
        V_s_coul = -self.alpha / self.r * erf(self.r/(np.sqrt(2) * self.cutoff))
        return V_s_coul

    def delta_contact_term(self):
        """Introduces a smoothed delta contact term with cutoff."""
        delta_smoothed = np.exp(-self.r**2 / (2 * self.cutoff**2)) / ((2 * np.pi)**1.5 * self.cutoff**3)
        return delta_smoothed
    
    def second_order_potential(self):
        """Introduces a potential term proportional to the second power of the cutoff"""
        delta_a = self.delta_contact_term()
        V_delta = self.c * self.cutoff ** 2 * delta_a
        return V_delta
    
    def spherical_laplacian(self):
        """
        Calculates the radial part of the Laplacian for a spherically symmetric function using
        a fourth-order finite difference method.

        Returns:
            np.array: The radial Laplacian of the delta contact term.
            """
        delta_a = self.delta_contact_term()  # Smoothed delta potential
        dr = self.dr  # Radial step size
        r = self.r  # Radial coordinates
        
        # Step 1: Compute the first derivative of delta_a using fourth-order finite differences
        d_delta_a = np.zeros_like(delta_a)
        d_delta_a[2:-2] = (-delta_a[4:] + 8 * delta_a[3:-1] - 8 * delta_a[1:-3] + delta_a[:-4]) / (12 * dr)
        
        # Step 2: Compute r^2 * d_delta_a
        r_squared_d_delta_a = r**2 * d_delta_a
        
        # Step 3: Compute the second derivative of r^2 * d_delta_a
        dd_r2_d_a = np.zeros_like(r_squared_d_delta_a)
        dd_r2_d_a[2:-2] = (-r_squared_d_delta_a[4:] + 8 * r_squared_d_delta_a[3:-1] 
                          - 8 * r_squared_d_delta_a[1:-3] + r_squared_d_delta_a[:-4]) / (12 * dr)

        # Step 4: Divide by r^2 to get the Laplacian
        laplacian = np.zeros_like(r)
        laplacian[2:-2] = dd_r2_d_a[2:-2] / r[2:-2]**2

        return laplacian
        
    def fourth_order_potential(self):
        """" Introduces an additional term to the potential proportional to the fourth power of the cutoff"""
        laplacian_delta = self.spherical_laplacian()
        V_4 = self.d_1 * self.cutoff ** 4 * laplacian_delta
        return V_4
        
    def total_effective_potential(self):
        """Combines the cutoff potential with a delta contact term."""
        V_s_coul = self.smeared_coulomb_potential()
        V_2 = self.second_order_potential()
        V_4 = self.fourth_order_potential()
        V_eff = V_s_coul + V_2 + V_4
        return V_eff
    
    def eff_hamiltonian_matrix(self, include_fourth_order=False):
        """
        Constructs the Hamiltonian matrix using the effective potential.
        
        Args:
            include_fourth_order (bool): Whether to include the fourth-order terms in the effective potential.
            
        Returns:
                np.array: Hamiltonian matrix.
        """
        if include_fourth_order:
            V_eff = self.total_effective_potential() + self.l * (self.l + 1) / (2 * self.r ** 2)
        else:
            V_eff = self.smeared_coulomb_potential() + self.second_order_potential() + self.l * (self.l + 1) / (2 * self.r ** 2)
    
        diag = 1 / self.dr**2 + V_eff
        if self.N <= 1:
            raise ValueError("N deve essere maggiore di 1 per costruire la matrice Hamiltoniana.")
        
        off_diag = (-1.0 / (2 * self.dr**2)) * np.ones(self.N - 1)
        H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
        return H

    def solve_effective_theory(self, include_fourth_order=False, num_eigenvalues=None):
        """
        Calculates and returns the eigenvalues and the eigenvectors of the Hamiltonian Matrix.

        Args:
            include_fourth_order (bool): Whether to include the fourth-order terms in the effective potential.
            num_eigenvalues (int, optional): Number of lowest eigenvalues to compute.
                                         If None, all eigenvalues are computed.

        Returns:
            eigenvalues, wavefunctions
        """
        # Use the appropriate Hamiltonian matrix based on the parameter
        H = self.eff_hamiltonian_matrix(include_fourth_order=include_fourth_order)

        # Compute specified number of eigenvalues and eigenvectors, or all if not specified
        if num_eigenvalues is not None:
            eigenvalues, wavefunctions = eigh(H, subset_by_index=[0, num_eigenvalues - 1])
        else:
            eigenvalues, wavefunctions = eigh(H)

        return eigenvalues, wavefunctions
    
    def normalize_wavefunctions(self, wavefunctions):
        """
        Normalizes the eigenfunctions so that the integral of |u(r)|^2 over r is equal to 1.
        Args:
            wavefunctions: Array of wavefunctions to be normalized.
        Returns:
            normalized_wavefunctions: Array of normalized wavefunctions.
        """
        normalized_wavefunctions = np.zeros_like(wavefunctions)
        
        for i in range(wavefunctions.shape[1]):
            norm = np.sqrt(np.sum(np.abs(wavefunctions[:, i])**2) * self.dr)
            normalized_wavefunctions[:, i] = wavefunctions[:, i] / norm
            
        return normalized_wavefunctions    
    
    def phase_shifts(self, energies):
        """ Identical to the one found in Schrodinger.py, 
        will be rewritten once the correctness of that one will be tested"""
    
        return 
        
    
    def fit_parameters_phase_shift(self, target_phase_shifts, energies):
        
        """Optimizes the parameters c and d_1 so that the phase shifts for the effective potential
        match the provided target phase shifts using differential evolution.

        Parameters:
        - target_phase_shifts (array-like): Array of target phase shifts to match.
        - energies (array-like): Array of energies corresponding to the target phase shifts.

        Returns:
        - Tuple (c, d_1): Optimized values of c and d_1."""

        
        def cost_function(params):
            # Set c and d_1 from input parameters
            self.c, self.d_1 = params
            
            # Difference between computed and target phase shifts for the first two energies
            diff = self.phase_shifts(energies[:2]) - target_phase_shifts[:2]
            
            # Sum of squared differences as the cost function
            return np.sum(diff**2)
        
        # Define bounds for c and d_1 (choose a wide range, e.g., between -100 and 100)
        bounds = [(-100, 100), (-100, 100)]

        # Use differential evolution to minimize the cost function without sign restrictions
        result = differential_evolution(cost_function, bounds, seed=42, tol=1e-6)

        # Check if the optimization was successful
        if result.success:
            self.c, self.d_1 = result.x  # Update class parameters with optimized values
            return result.x  # Return optimized values as tuple (c, d_1)
        else:
            raise ValueError("Optimization failed to converge")

    def set_parameters_from_fit(self, target_phase_shifts, energies):
        
        """Sets the values of c and d_1 based on the optimized values obtained
        from the fit_parameters_phase_shift function.

        Parameters:
            - target_phase_shifts (array-like): Array of target phase shifts to match.
            - energies (array-like): Array of energies corresponding to the target phase shifts.

        Returns:
            - Tuple (c, d_1): Optimized values of c and d_1."""
        
        # Call the fit function to obtain optimized values
        optimized_params = self.fit_parameters_phase_shift(target_phase_shifts, energies)
        
        # Set the class attributes c and d_1 to the optimized values
        self.c, self.d_1 = optimized_params
    
    """ ---
               PLOTTING METHODS
                                 ---"""
    
    def plot_potentials(self):
        """
        Plots the effective potential with corrections up to O(a^2) and O(a^4), alongside the true potential
        as defined by the total potential from RadialHydrogenSolver.
        """
        r = self.r  # Radial grid

        # True potential from the base class RadialHydrogenSolver
        true_potential = self.total_potential()  # This is assumed to be the full Coulomb potential

        # Effective potential with O(a^2) correction
        effective_potential_o2 = self.smeared_coulomb_potential() + self.second_order_potential()

        # Effective potential with O(a^4) correction
        effective_potential_o4 = self.total_effective_potential()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(r, true_potential, label="True Potential (Total)", linestyle='--')
        plt.plot(r, effective_potential_o2, label="Effective Potential O(a^2)", linestyle='-.')
        plt.plot(r, effective_potential_o4, label="Effective Potential O(a^4)", linestyle='-')
        
        # Adding labels and title
        plt.xlabel("r")
        plt.ylabel("Potential V(r)")
        plt.title("Comparison of True and Effective Potentials")
        plt.legend()
        plt.xlim(0, 10)
        plt.ylim([-10, 0.1])  # Set y-limits for clarity around the potential behavior
        plt.grid(True)
        plt.show()
    
    def plot_relative_energy_errors(self, eigenvalues, energy_approx):
        """
        Calculates and plots the relative error in binding energies for S-waves
        comparing the Coulomb potential and the first-order delta correction.

        Args:
            eigenvalues (np.array): Array with exact binding energy values.
            energy_approx (np.array): Array with approximate energies using only the Coulomb potential.
            c_param (float): Parameter for the delta correction (Dirac delta function).

        Returns:
            None
        """
        n = np.arange(1, 11)  # Array of values from 1 to 10
        energy_coul = - 1 / (2 * n**2)

        # Limit to the first 10 values
        eigenvalues_10 = eigenvalues[:10]
        energy_eff_2nd, _ = self.solve_effective_theory(include_fourth_order=False, num_eigenvalues=10)
        energy_eff_4th, _ = self.solve_effective_theory(include_fourth_order=True, num_eigenvalues=10)
        energy_approx_10 = energy_approx[:10]
        
        print("energy_coul:", energy_coul)
        print("eigenvalues_10:", eigenvalues_10)

        # Calculation of relative error for the two approximations
        rel_error_coulomb = np.abs((eigenvalues_10 - energy_coul) / eigenvalues_10)
        rel_error_delta = np.abs((eigenvalues_10 - energy_approx_10) / eigenvalues_10)
        rel_error_effective_2nd = np.abs((eigenvalues_10 - energy_eff_2nd) / eigenvalues_10)
        rel_error_effective_4th = np.abs((eigenvalues_10 - energy_eff_4th) / eigenvalues_10)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.loglog(np.abs(eigenvalues_10), rel_error_coulomb, 'o-', label=r'$-\alpha/r$', color="blue")
        plt.loglog(np.abs(eigenvalues_10), rel_error_delta, 's-', label=r'$-\alpha/r + c\delta^3(r)$ (1st order)', color="orange")
        plt.loglog(np.abs(eigenvalues_10), rel_error_effective_2nd, '^-', label=r'$-\alpha/r + c a^2 \delta^3_a(r)$', color="green")
        plt.loglog(np.abs(eigenvalues_10), rel_error_effective_4th, 'd-', label=r'$-\alpha/r + c a^2 \delta^3_a(r) - d_1 a^4 \nabla^2 \delta^3_a(r)$', color="red")

        # Plot customization
        plt.xlabel("Binding Energy (E)", fontsize=14)
        plt.ylabel(r"$|\Delta E / E|$", fontsize=14)
        plt.legend(loc="upper left")
        plt.title("Relative Errors in S-Wave Binding Energies", fontsize=16)
        
        plt.show()

    

