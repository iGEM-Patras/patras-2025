"""
Two-Reaction Strand Displacement ODE Model Script
Simulates the complete strand displacement cascade using hardcoded rate constants
from thermodynamic analysis.

System Parameters:
- Temperature: 37°C
- Na+ concentration: 50 mM
- Mg++ concentration: 5 mM
- Initial concentrations: 1 μM each hairpin, 100 nM initiator

Reactions:
1. initiator + h1 ⇌ initiator:h1  (k1f, k1r)
2. initiator:h1 + h2 ⇌ h1:h2 + initiator  (k2f, k2r)

Species: [initiator], [h1], [h2], [initiator:h1], [h1:h2]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class StrandDisplacementODE:
    def __init__(self):
        # System parameters
        self.temperature_celsius = 37.0
        self.temperature_kelvin = self.temperature_celsius + 273.15
        self.na_conc = 0.05  # 50 mM
        self.mg_conc = 0.005   # 5 mM
        self.initial_hairpin_conc = 1e-6  # 1 μM
        self.initial_initiator_conc = 1e-7  # 100 nM
        
        # Hardcoded rate constants from thermodynamic analysis
        self.k1f = 3.00e6  # M^-1 s^-1 - reaction 1 forward
        self.k1r = 1.34e-3  # s^-1 - reaction 1 reverse
        self.k2f = 3.00e6  # M^-1 s^-1 - reaction 2 forward
        self.k2r = 3.40e4  # M^-1 s^-1 - reaction 2 reverse
        
        # Reaction velocities
        self.v1 = None  # Will be calculated during ODE
        self.v2 = None  # Will be calculated during ODE
        
        self.sequences = {
            'initiator': 'CACCTAGCCCGCACACTTT',
            'h1': 'AAAGTGTGCGGGGCTAGGTGACATTAACTACACCTAGCCCC',
            'h2': 'GTAGGTGTAGTTAATGTCACCTAGCCCACATTAACTA'
        }
        
        print("Two-Reaction Strand Displacement ODE Model initialized")
        print(f"Temperature: {self.temperature_celsius}°C")
        print(f"Na+ concentration: {self.na_conc * 1000} mM")
        print(f"Mg++ concentration: {self.mg_conc * 1000} mM")
        print(f"Initial hairpin concentration: {self.initial_hairpin_conc * 1e6} μM")
        print(f"Initial initiator concentration: {self.initial_initiator_conc * 1e9} nM")
        print("\nHardcoded rate constants:")
        print(f"  k1f = {self.k1f:.2e} M^-1 s^-1")
        print(f"  k1r = {self.k1r:.2e} s^-1")
        print(f"  k2f = {self.k2f:.2e} M^-1 s^-1")
        print(f"  k2r = {self.k2r:.2e} M^-1 s^-1")
        print("\nReactions:")
        print("  1. initiator + h1 ⇌ initiator:h1")
        print("  2. initiator:h1 + h2 ⇌ h1:h2 + initiator")
    
    
    def ode_system(self, concentrations, t):
        """
        Define the ODE system for both reactions in the strand displacement cascade
        
        Species order: [initiator], [h1], [h2], [initiator:h1], [h1:h2]
        
        Reactions:
        1. initiator + h1 ⇌ initiator:h1  (k1f, k1r)
        2. initiator:h1 + h2 ⇌ h1:h2 + initiator  (k2f, k2r)
        
        Parameters:
        - concentrations: current concentration vector [I, H1, H2, IH1, H1H2]
        - t: current time (required by odeint but not used in this autonomous system)
        """
        
        # Unpack concentrations
        I, H1, H2, IH1, H1H2 = concentrations
        
        # Calculate reaction velocities
        v1_forward = self.k1f * I * H1      # initiator + h1 → initiator:h1
        v1_reverse = self.k1r * IH1         # initiator:h1 → initiator + h1
        
        v2_forward = self.k2f * IH1 * H2    # initiator:h1 + h2 → h1:h2 + initiator
        v2_reverse = self.k2r * H1H2 * I    # h1:h2 + initiator → initiator:h1 + h2
        
        # Net reaction velocities
        v1 = v1_forward - v1_reverse
        v2 = v2_forward - v2_reverse
        
        # Store velocities for analysis
        self.v1 = v1
        self.v2 = v2
        
        # ODEs for each species using v1 and v2
        dI_dt = -v1 + v2     # initiator: consumed in rxn1, produced in rxn2
        dH1_dt = -v1         # h1: consumed in rxn1
        dH2_dt = -v2         # h2: consumed in rxn2
        dIH1_dt = v1 - v2    # initiator:h1: produced in rxn1, consumed in rxn2
        dH1H2_dt = v2        # h1:h2: produced in rxn2
        
        return [dI_dt, dH1_dt, dH2_dt, dIH1_dt, dH1H2_dt]
    
    def solve_ode_system(self, simulation_time=50, n_points=10000):
        """
        Solve the ODE system and return time course data
        """
        print(f"\n=== ODE System Solution ===")
        print(f"Simulation time: {simulation_time} seconds ({simulation_time*1000:.1f} milliseconds)")
        
        # Initial conditions - all species start separate
        I0 = self.initial_initiator_conc    # 100 nM initiator
        H1_0 = self.initial_hairpin_conc    # 1 μM h1
        H2_0 = self.initial_hairpin_conc    # 1 μM h2  
        IH1_0 = 0.0                         # 0 μM initiator:h1
        H1H2_0 = 0.0                        # 0 μM h1:h2
        
        initial_conditions = [I0, H1_0, H2_0, IH1_0, H1H2_0]
        
        print(f"Initial conditions:")
        print(f"  [initiator] = {I0*1e9:.0f} nM")
        print(f"  [h1] = {H1_0*1e6:.1f} μM")
        print(f"  [h2] = {H2_0*1e6:.1f} μM")
        print(f"  [initiator:h1] = {IH1_0*1e6:.1f} μM")
        print(f"  [h1:h2] = {H1H2_0*1e6:.1f} μM")
        
        # Time points - this creates an array of time values from 0 to simulation_time
        t = np.linspace(0, simulation_time, n_points)
        
        # Solve ODE system using scipy's odeint function
        try:
            solution = odeint(self.ode_system, initial_conditions, t)
            print("ODE system solved successfully!")
            return t, solution
            
        except Exception as e:
            print(f"Error solving ODE system: {e}")
            return None, None
    
    def plot_results(self, t, solution):
        """Plot the concentration time courses for all 5 species"""
        if solution is None:
            print("No solution to plot")
            return
        
        # Convert time to milliseconds and concentrations to μM
        t_ms = t * 1000
        solution_uM = solution * 1e6
        
        # Create the plot - single plot showing all species
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot all species
        species_names = ['Initiator', 'h1', 'h2', 'Initiator:h1', 'h1:h2']
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, (name, color) in enumerate(zip(species_names, colors)):
            ax1.plot(t_ms, solution_uM[:, i], label=name, color=color, linewidth=2)
        
        ax1.set_xlabel('Time (milliseconds)')
        ax1.set_ylabel('Concentration (μM)')
        ax1.set_title('Complete Strand Displacement Cascade - All Species')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, None)
        
        # Add rate constants as text
        rate_text = (f"Rate Constants:\n"
                    f"k1f = {self.k1f:.2e} M⁻¹s⁻¹\n"
                    f"k1r = {self.k1r:.2e} s⁻¹\n"
                    f"k2f = {self.k2f:.2e} M⁻¹s⁻¹\n"
                    f"k2r = {self.k2r:.2e} M⁻¹s⁻¹")
        
        ax1.text(0.02, 0.98, rate_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('two_reaction_strand_displacement.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final concentrations
        print(f"\nFinal concentrations (μM) at t = {t_ms[-1]:.1f} ms:")
        for i, name in enumerate(species_names):
            print(f"  [{name}] = {solution_uM[-1, i]:.3f}")
        
        # Calculate conversion efficiencies
        h1h2_final = solution_uM[-1, 4]  # h1:h2 final concentration
        initial_initiator = solution_uM[0, 0]  # Initial initiator
        initial_h1 = solution_uM[0, 1]  # Initial h1
        initial_h2 = solution_uM[0, 2]  # Initial h2
        
        # Conversion based on limiting reagent (initiator)
        conversion_initiator = h1h2_final / initial_initiator * 100
        conversion_h1 = h1h2_final / initial_h1 * 100
        conversion_h2 = h1h2_final / initial_h2 * 100
        
        print(f"\nConversion efficiencies:")
        print(f"  Based on initiator: {conversion_initiator:.1f}%")
        print(f"  Based on h1: {conversion_h1:.1f}%")
        print(f"  Based on h2: {conversion_h2:.1f}%")
        

    
    def run_full_analysis(self):
        """Run the complete analysis"""
        print("="*70)
        print("TWO-REACTION STRAND DISPLACEMENT ODE MODEL ANALYSIS")
        print("="*70)
        
        # Solve ODE system with hardcoded rates
        t, solution = self.solve_ode_system(simulation_time=50)  
        
        # Plot results
        if solution is not None:
            self.plot_results(t, solution)
        
        return None


def main():
    """Main function to run the analysis"""
    analyzer = StrandDisplacementODE()
    analyzer.run_full_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return None

if __name__ == "__main__":
    main()