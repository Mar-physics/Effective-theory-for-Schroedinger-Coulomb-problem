from Schrodinger import RadialHydrogenSolver
from EffectiveTheory import EffectiveTheorySolver
from DeltaPotential import DeltaPotentialSolver
import numpy as np

def main():
    # General parameters
    alpha = 1  # Fine structure constant
    g = 1      # Yukawa potential strength
    m = 1      # Reduced mass
    r_min = 0.1
    r_max = 1000.0
    N = 10000
    l = 0
    cutoff = 1.0
    c = 1
    d_1 = 1
    energies = np.array([10e-10, 10e-5, 0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1])
    
    solver = RadialHydrogenSolver(alpha=alpha, g=g, m=m, r_min=r_min, r_max=r_max, N=N, l=l)
    eigenvalues, wavefunctions = solver.solve(num_eigenvalues=20)
    print("Calculating energies (first 20):")
    for i, energy in enumerate(eigenvalues, 1):
        print(f"E_{i} = {energy:.6f}")

    # 2. Calculate phase shifts
    phase_shifts = solver.compute_phase_shifts(energies)
    print("\nCalculated phase shifts:")
    for e, delta in zip(energies, phase_shifts):
        print(f"Energy: {e:.6e}, Phase Shift: {delta:.6f}")
    
    delta_solver = DeltaPotentialSolver(alpha=alpha, m=m, r_min=r_min, r_max=r_max, N=N, l=l)
    delta_solver.set_c(eigenvalues)
    energy_approx = delta_solver.approximate_energy_levels()
    effective_solver = EffectiveTheorySolver(alpha=alpha, g=g, m=m, r_min=r_min, r_max=r_max, N=N, l=l, cutoff=cutoff, c=c, d_1=d_1)
    
    """effective_solver.set_parameters_from_fit(phase_shifts, energies)
    print(f"Calculated value of c: {effective_solver.c}, Calculated value of d_1: {effective_solver.d_1}")"""
    
    effective_eigenvalues_2nd, _ = effective_solver.solve_effective_theory(include_fourth_order=False, num_eigenvalues=20)
    effective_eigenvalues_4th, _ = effective_solver.solve_effective_theory(include_fourth_order=True, num_eigenvalues=20)
    
    new_eff_phase_shifts = effective_solver.phase_shifts(energies)
    print("Energy    Phase Shift (rad)")
    for energy, delta in zip(energies, new_eff_phase_shifts):
        print(f"{energy:.12f}      {delta:.12f}")
    
    print("\nEffective energy at first order (first 20):")
    for i, energy in enumerate(effective_eigenvalues_2nd, 1):
        print(f"E_{i} = {energy:.6f}")
    
    print("\nEffective energy at second order (first 20):")
    for i, energy in enumerate(effective_eigenvalues_4th, 1):
        print(f"E_{i} = {energy:.6f}")

    # 4. Plot potentials and relative errors
    effective_solver.plot_potentials()
    effective_solver.plot_relative_energy_errors(eigenvalues, energy_approx)

if __name__ == "__main__":
    main()

    
    
    
