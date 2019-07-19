"""
In this function, called PFR_1D_with_solid_phase_porous_two_components.py,
the concentration and temperature profile of the 1D Plug Flow Reactor 
are solved and coupled for both fluid and solid phase.
The solid phase is assumed to be porous. 
The system consist of convection, diffusion and reaction.
The reaction term is defined as A + B --> C.

Acknowledgements: this script is inspired on the work of I.A.W. Filot 
and M.P.C. van Etten
"""
# Importing the package for computations in Python
import numpy as np
# Importing the package for plotting in Python
import matplotlib.pyplot as plt
# import timing library
from timeit import default_timer as timer

def main():
    tstart = timer()
    
    """
    1. Defining the following types of variables:
        - Reactor variables
        - Variables for concentration profile
        - Variables for temperature profile
        - Computational stencil parameters
    """
    # Reactor variables
    L_reactor = 1.0        	# Length of reactor (m)
    velocity_inlet = 1.0    # Velocity of the entering reactant gas mixture (m/s)
    R_cycl = 2.5e-2         # Radius of the reactor (m)
    
    # Variables for concentration profile
    c_A_in = 240.56         # Inlet feed concentration of A (mol/m3)
    c_B_in = 240.56         # Inlet feed concentration of B (mol/m3)
    c_C_in = 0.0            # Inlet feed concentration of C (mol/m3)
    c_N_in = 0.0            # Inlet feed concentration of N (mol/m3)
    
    D_f_A = 1.0e-4          # Diffusion coefficient of reactant A in gas (m2/s)
    D_f_B = 1.0e-4          # Diffusion coefficient of reactant B in gas (m2/s)
    D_f_C = 1.0e-4          # Diffusion coefficient of reactant C in gas (m2/s)
    D_f_N = 1.0e-4          # Diffusion coefficient of reactant N in gas (m2/s)
    D_f_total = (D_f_A+D_f_B+D_f_C+D_f_N)/4         # Diffusion coefficient of gas mixture (m2/s)
    
    mu = 1e-5               # Dynamic viscocity of gas mixture (m2/s)
    
    # Variables for temperature profile
    rho_A = 5.0             # Density of the reactant A (kg / m3)
    rho_B = 5.0             # Density of the reactant B (kg / m3)
    rho_C = 5.0             # Density of the reactant C (kg / m3)
    rho_N = 5.0             # Density of the reactant N (kg / m3)
    rho_total = (rho_A+rho_B+rho_C+rho_N)/4       # Density of gas mixture (kg / m3)
    
    Cp_A = 1.0e3            # Specific heat capacity of the reactant A (Joule/(Kg.K))
    Cp_B = 1.0e3            # Specific heat capacity of the reactant B (Joule/(Kg.K))
    Cp_C = 1.0e3            # Specific heat capacity of the reactant B (Joule/(Kg.K))
    Cp_N = 1.0e3            # Specific heat capacity of the reactant C (Joule/(Kg.K))
    Cp_total = (Cp_A+Cp_B+Cp_C+Cp_N)/4            # Specific heat capacity of the gas mixture (Joule/(Kg.K))
    
    T_in = 500.0            # Inlet temperature (K)
    
    k_f = 0.14              # Thermal conductivity of gas mixture (Watt/(m.K))
    
    T_wall = 373.15         # Temperature of the reactor wall (K) (condensing steam)
    a = 2/R_cycl
    
    # Variables for solid phase 
    e = 0.5                 # Void fraction present in the reactor (-)
    d_particle = 0.02       # Diameter of particle (m)
    r_particle = d_particle/2
    
    D_s_A = 7.5e-7          # Diffusivity of the reactant A in the porous solid catalyst (m2/s)
    D_s_B = 7.5e-7          # Diffusivity of the reactant B in the porous solid catalyst (m2/s)
    D_s_C = 7.5e-7          # Diffusivity of the reactant C in the porous solid catalyst (m2/s)
    k_s = 0.5               # Thermal conductivity of the solid catalyst (Watt/(m.K))
    
    rho_Cp_solid = 1.0e5    # Product of density and specific heat capacity of the solid catalyst

    # Kinetic parameters
    delta_H = 1.2e5         # Exothermicity (J/mol)
    Ea = 8.0e4              # Activation energy (J/mol)
    R = 8.314               # Gas constant (J/(K*mol))
    k0 = 3.98e9             # Arrhenius pre-factor (1/s)
    
    # Combining the variables
    ingoing = [c_A_in, c_B_in, c_C_in, c_N_in, T_in]
    rho = [rho_A, rho_B, rho_C, rho_N, rho_total]  
    Cp = [Cp_A, Cp_B, Cp_C, Cp_N, Cp_total]
    D_f = [D_f_A, D_f_B, D_f_C, D_f_N, D_f_total]
    D_s = [D_s_A, D_s_B, D_s_C]
    
    # Set grid points
    N_grid_reactor = 30
    N_grid_particle = 40
    
    # Computational stencil parameters
    delta_t = 1.0e-3                              # Time-step value (s)
    delta_x = L_reactor / N_grid_reactor          # Grid size for reactor transport (m)
    delta_r = r_particle / N_grid_particle        # Grid size for particle transport (m)
    
    """
    2. Discretization of space and determination of the total time duration
    """
    time = 1.0                                     # Total time duration
    
    spacesteps = N_grid_reactor                     # Number of steps in reactor
    x = np.linspace(0,L_reactor,spacesteps+1)       # Vector with the steps in reactor
    
    spacesteps_solid = N_grid_particle              # Number of steps in catalytic particle
    r = np.linspace(0,r_particle,spacesteps_solid)  # Vector with the steps in catalytic particle
    
    """
    3. Calculating the numerical solution
    """
    # Initial concentration profile for fluid phase (at t=0, only inert gas is present in the reactor)
    conc_A_current = np.zeros(spacesteps+1)
    conc_B_current = np.zeros(spacesteps+1)
    conc_C_current = np.zeros(spacesteps+1)
    conc_N_current = np.ones(spacesteps+1)*481.12
    # Initial concentration profile for solid phase
    conc_A_solid_current = np.zeros((spacesteps+1,spacesteps_solid))
    conc_B_solid_current = np.zeros((spacesteps+1,spacesteps_solid))
    conc_C_solid_current = np.zeros((spacesteps+1,spacesteps_solid))
    
    # Initial temperature profile for fluid phase (at t=0, overall temperature in reactor is equal to T_wall)
    T_current = np.ones(spacesteps+1)*T_wall
    # Initial temperature profile for solid phase
    T_solid_current = np.ones((spacesteps+1,spacesteps_solid))*T_wall
   
    current_time = 0
    
    # Solving the concentration- and temperature profile over time
    while current_time < time:
        y, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, conc_A_solid_current, conc_B_solid_current, conc_C_solid_current,T_solid_current = solver(L_reactor, d_particle, velocity_inlet, spacesteps, spacesteps_solid, delta_x, delta_r, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, conc_A_solid_current, conc_B_solid_current, conc_C_solid_current, T_solid_current, ingoing, rho, Cp, delta_H, k0, Ea, R, a, T_wall, e, mu, D_s, k_s, rho_Cp_solid)
        current_time = current_time + delta_t
        
    """
    4. Calculating the molfractions of the reactants
    """
    molfraction_A = conc_A_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current)
    molfraction_B = conc_B_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current)
    molfraction_C = conc_C_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current)
    molfraction_N = conc_N_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current)
    
    """
    5. Plotting the numerical solution
    """
    # Plot of the molar fractions of reactants and temperature in the fluid phase
    fig = plt.subplots(2,2, figsize = (12,8))
    fig = plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.suptitle('PFR 1D Fluid Phase with porous solid catalytic particle')
    
    # Concentration profile of fluid phase
    plt.subplot(2,2,1)
    plt.plot(x, conc_A_current, '--o',label='A', markersize=3)
    plt.plot(x, conc_B_current, '--o',label='B', markersize=3)
    plt.plot(x, conc_C_current, '--o',label='C', markersize=3)
    plt.plot(x, conc_N_current, '--o',label='N', markersize=3)
    plt.xlabel('Position in reactor [m]')
    plt.ylabel('Concentration [mol/m3]')
    plt.title('Concentration of reactants in reactor')
    plt.legend(loc=7)
    plt.grid(linestyle='--', color='0.9')
    
    # Temperature profile of fluid phase
    plt.subplot(2,2,2)
    plt.plot(x,T_current, '--o',label='Temperature', markersize=3)
    plt.xlabel('Position in reactor [m]')
    plt.ylabel('Temperature [K]')
    plt.title('Temperature in the reactor')
    plt.legend(loc=7)
    plt.grid(linestyle='--', color='0.9')
    
    # Concentration profile of solid phase
    plt.subplot(2,2,3)
    idx = 14
    plt.plot(r, conc_A_solid_current[idx,:], '--o', markersize=3, label='A')
    plt.plot(r, conc_B_solid_current[idx,:], '--o', markersize=3, label='B')
    plt.plot(r, conc_C_solid_current[idx,:], '--o', markersize=3, label='C')
    plt.xlabel('Position in solid particle [m]')
    plt.ylabel('Concentration [mol/m$^3$]')
    plt.title('Concentration profile within porous particle at L=%0.4f' % x[idx])
    plt.legend(loc=7)
    plt.grid(linestyle='--', color='0.9')

    # Temperature profile of solid phase
    plt.subplot(2,2,4)
    for i in np.linspace(delta_x, N_grid_reactor, 7):
        idx = int(i)
        plt.plot(r, T_solid_current[idx,:], '--o', markersize=3, label='L = %0.4f m' % x[idx])
    idx = 13
    plt.plot(r, T_solid_current[idx,:], '--o', markersize=3, label='L = %0.4f m' % x[idx])
    plt.xlabel('Position in solid particle [m]')
    plt.ylabel('Temperature [K]')
    plt.title('Temperature profile within porous particle')
    plt.legend(loc=7)
    plt.grid(linestyle='--', color='0.9')
    
    plt.show()
    
    tend = timer()
    print("Performed simulation in %f seconds" % (tend - tstart))
    
def solver(L_reactor, d_particle, velocity_inlet, spacesteps, spacesteps_solid, delta_x, delta_r, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, conc_A_solid_current, conc_B_solid_current, conc_C_solid_current, T_solid_current, ingoing, rho, Cp, delta_H, k0, Ea, R, a, T_wall, e, mu, D_s, k_s, rho_Cp_solid):
    # Generate matrices for fluid phase: concentrations of reactants A, B and N
    matrix_A = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 0)
    matrix_B = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 1)
    matrix_C = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 2)
    matrix_N = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 3)
    # Generate matrix for fluid phase: temperature inclusive calculation of product of densities and specific heat capacities
    matrix_T, rho_Cp_T = generate_matrix_temperature_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, k_f, rho, Cp, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 4)
    # Generate matrices for solid phase: concentration and temperature
    matrix_A_solid = generate_matrix_concentration_solid_phase(d_particle, delta_r, spacesteps_solid, delta_t, D_s, 0)
    matrix_B_solid = generate_matrix_concentration_solid_phase(d_particle, delta_r, spacesteps_solid, delta_t, D_s, 1)
    matrix_C_solid = generate_matrix_concentration_solid_phase(d_particle, delta_r, spacesteps_solid, delta_t, D_s, 2)
    matrix_T_solid = generate_matrix_temperature_solid_phase(d_particle, delta_r, spacesteps_solid, delta_t, k_s, rho_Cp_solid)
    
    # calculate matrix inverse
    mat_A_sol_inv = np.linalg.inv(matrix_A_solid)
    mat_B_sol_inv = np.linalg.inv(matrix_B_solid)
    mat_C_sol_inv = np.linalg.inv(matrix_C_solid)
    mat_T_sol_inv = np.linalg.inv(matrix_T_solid)
    
    # Generate right-hand side vector for fluid phase: concentrations of reactants A, B and N
    vector_A = np.copy(conc_A_current)
    vector_B = np.copy(conc_B_current)
    vector_C = np.copy(conc_C_current)
    vector_N = np.copy(conc_N_current)
    # Generate right-hand side vector for fluid phase: temperature
    vector_T = np.copy(T_current)
    # Generate right-hand side vector for solid phase: concentration and temperature
    vector_A_solid = np.copy(conc_A_solid_current)
    vector_B_solid = np.copy(conc_B_solid_current)
    vector_C_solid = np.copy(conc_C_solid_current)
    vector_T_solid = np.copy(T_solid_current)
    
    # Initialization of solid phase profiles
    A_solid = np.zeros((spacesteps+1,spacesteps_solid))
    B_solid = np.zeros((spacesteps+1,spacesteps_solid))
    C_solid = np.zeros((spacesteps+1,spacesteps_solid))
    T_solid = np.ones((spacesteps+1,spacesteps_solid))*T_wall
    
    # Specific area
    rp = (d_particle / 2.0)
    Ap = 3 * (rp**2)
    Vp = rp**3 - (rp-delta_r)**3
    AdivV = Ap / Vp
    
    hwall_store = np.zeros(spacesteps)
    velocity_store = np.zeros(spacesteps)
    
    # Process the reaction, heat-transfer, coupling between concentration/temperature and coupling between solid/fluid phase within the right-hand side vector
    for i in range(1,spacesteps):
            # Calculate the velocity at the defined position
            velocity_current = velocity(velocity_inlet, L_reactor, conc_A_current[i], conc_B_current[i], conc_C_current[i], conc_N_current[i], T_current[i], ingoing)
            # Calculate the Reynolds number
            Re = (velocity_current * d_particle)/D_f[4]
            # Calculate the density at the defined position for fluid phase depending on the molfractions of the reactants
            molfraction = [0,0,0,0]
            molfraction[0] = conc_A_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
            molfraction[1] = conc_B_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
            molfraction[2] = conc_B_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
            molfraction[3] = conc_N_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
            rho_i = rho[4] #((molfraction[0]*rho[0])+(molfraction[1]*rho[1])+(molfraction[2]*rho[2])+(molfraction[3]*rho[3]))
            # Calculate the mass tranfer coefficient
            k_mass_transfer_A = mass_transfer_coefficient(d_particle, velocity_current, 0, D_f, mu, rho, e)
            k_mass_transfer_B = mass_transfer_coefficient(d_particle, velocity_current, 1, D_f, mu, rho, e)
            k_mass_transfer_C = mass_transfer_coefficient(d_particle, velocity_current, 2, D_f, mu, rho, e)
            # Calculate the Prandlt number 
            Pr = (Cp[4]*mu)/k_f
            # Calculate the heat transfer coefficient
            k_heat_transfer = (k_f*Pr) / d_particle
            # Calculate the wall-to-bed heat transfer coefficient for fluid phase heat transfer term
            if Re < 40:
                Nu_wall = 0.6 * (Re**0.5)
            else: 
                Nu_wall = 0.2 * (Re**0.8)
            h_wall = (Nu_wall * k_f) / d_particle
            hwall_store[i] = h_wall
            velocity_store[i] = velocity_current
            # Define the heat transfer term for fluid phase
            heat_transfer_fluid_phase = (h_wall * a * (vector_T[i] - T_wall) * delta_t ) / (rho_Cp_T[i])
            
            # Calculate the concentration- and temperature profile of the solid phase at the defined position
            for j in range(0,spacesteps_solid):
                # Calculate the reaction rate constant at the defined position
                k_reaction = (k0 * np.exp(-Ea/(R*vector_T_solid[i,j])))
                
                # Calculate the reaction term of the concentration profile of the solid phase
                reaction_solid = k_reaction * vector_A_solid[i,j] * vector_B_solid[i,j]
                
                # Define the reaction term of the concentration profile of the solid phase
                vector_A_solid[i,j] = vector_A_solid[i,j] - reaction_solid
                vector_B_solid[i,j] = vector_B_solid[i,j] - reaction_solid
                vector_C_solid[i,j] = vector_C_solid[i,j] + reaction_solid
            
                # Define the produced heat term of the temperature profile of the solid phase
                vector_T_solid[i,j] = vector_T_solid[i,j] + (k_reaction * delta_H * delta_t / rho_Cp_solid) * vector_A_solid[i,j] * vector_B_solid[i,j]
            
            # Implementation of the boundary conditions for the solid phase
            vector_A_solid[i,-1] = vector_A_solid[i,-1] + (k_mass_transfer_A * delta_t * (vector_A[i]-vector_A_solid[i,-1])) * AdivV
            vector_B_solid[i,-1] = vector_B_solid[i,-1] + (k_mass_transfer_B * delta_t * (vector_B[i]-vector_B_solid[i,-1])) * AdivV
            vector_C_solid[i,-1] = vector_C_solid[i,-1] + (k_mass_transfer_C * delta_t * (vector_C[i]-vector_C_solid[i,-1])) * AdivV
            vector_T_solid[i,-1] = vector_T_solid[i,-1] + (k_heat_transfer / rho_Cp_solid * delta_t * (vector_T[i]-vector_T_solid[i,-1])) * AdivV
            
            # Solving the concentration- and temperature profile of the solid phase at the defined position
            A_solid[i,:] = mat_A_sol_inv.dot(vector_A_solid[i,:])
            B_solid[i,:] = mat_B_sol_inv.dot(vector_B_solid[i,:])
            C_solid[i,:] = mat_C_sol_inv.dot(vector_C_solid[i,:])
            T_solid[i,:] = mat_T_sol_inv.dot(vector_T_solid[i,:])
            
            # Calculate the reaction term for the reactants and temperature in the fluid phase
            reaction_mass_A = (1.0 - e) * AdivV * k_mass_transfer_A * (vector_A[i]-vector_A_solid[i,-1])
            reaction_mass_B = (1.0 - e) * AdivV * k_mass_transfer_B * (vector_B[i]-vector_B_solid[i,-1])
            reaction_mass_C = (1.0 - e) * AdivV * k_mass_transfer_C * (vector_C[i]-vector_C_solid[i,-1])
            reaction_heat = (1.0 - e) * AdivV * k_heat_transfer * (vector_T[i]-vector_T_solid[i,-1])
            vector_A[i] = vector_A[i]  - reaction_mass_A * delta_t
            vector_B[i] = vector_B[i]  - reaction_mass_B * delta_t
            vector_C[i] = vector_C[i]  - reaction_mass_C * delta_t
            vector_N[i] = vector_N[i]
            vector_T[i] = vector_T[i] - (reaction_heat * delta_t / rho_Cp_T[i]) - heat_transfer_fluid_phase
            
    # Implement the boundary condition within the right-hand side vector
    vector_A[0] = ingoing[0]
    vector_B[0] = ingoing[1]
    vector_C[0] = ingoing[2]
    vector_N[0] = ingoing[3]
    vector_T[0] = ingoing[4]
    
    vector_A[spacesteps] = 0
    vector_B[spacesteps] = 0
    vector_C[spacesteps] = 0
    vector_N[spacesteps] = 0
    vector_T[spacesteps] = 0
    
    # Solve the equation Ax=B with x being the concentration profile 
    # at the next timestep
    A = np.linalg.solve(matrix_A,vector_A)
    B = np.linalg.solve(matrix_B,vector_B)
    C = np.linalg.solve(matrix_C,vector_C)
    N = np.linalg.solve(matrix_N,vector_N)
    T = np.linalg.solve(matrix_T,vector_T)  
    
    return np.linspace(0,L_reactor,spacesteps+1), A, B, C, N, T, A_solid, B_solid, C_solid, T_solid
    
def generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, situation):
    # Define step within the reactor
    step = delta_x

    # Initialization of matrix
    matrix = np.zeros((spacesteps+1,spacesteps+1))
        
    # Loop over position in reactor
    for i in range(1,spacesteps):
        # Calculating the current velocity at defined position
        velocity_current = velocity(velocity_inlet, L_reactor, conc_A_current[i], conc_B_current[i], conc_C_current[i], conc_N_current[i], T_current[i], ingoing)
        # Calculate the axial mass dispersion coefficient by Edwards and Richardson (1968)
        D_ax = 0.73 * D_f[situation] + (0.5*velocity_current*d_particle)/(1+((9.7*D_f[situation])/(velocity_current*d_particle)))
        # Combining the variables
        alpha = -(delta_t*D_ax)/(step**2)
        beta = (velocity_current*delta_t)/(step)
        # Generating the matrix at defined position
        matrix[i,i-1] = alpha - beta
        matrix[i,i+1] = alpha 
        matrix[i,i] = 1.0 - 2.0*alpha + beta 
        
    # Boundary conditions
    matrix[0,0] = 1.0
    matrix[spacesteps,spacesteps-3] = -2.0 / (6.0*step)
    matrix[spacesteps,spacesteps-2] = 9.0 / (6.0*step)
    matrix[spacesteps,spacesteps-1] = -18.0 / (6.0*step)
    matrix[spacesteps,spacesteps] = 11.0 / (6.0*step)

    return matrix

def generate_matrix_temperature_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, k_f, rho, Cp, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, situation):
    # Define step within the reactor
    step = delta_x 

    # Initialization of matrix
    matrix = np.zeros((spacesteps+1,spacesteps+1))
    rho_Cp_T = np.zeros(spacesteps)

    # Loop over position in reactor
    for i in range(1,spacesteps):
        molfraction = [0,0,0,0]
        # Determing average density and specific heat capacity depending on the molfraction of the reactant
        molfraction = [0,0,0,0]
        molfraction[0] = conc_A_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        molfraction[1] = conc_B_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        molfraction[2] = conc_B_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        molfraction[3] = conc_N_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        rho_Cp_T[i] = ((rho[0]*Cp[0])+(rho[1]*Cp[1])+(rho[2]*Cp[2])+(rho[3]*Cp[3]))/4
        # Calculating the current velocity at defined position
        velocity_current = velocity(velocity_inlet, L_reactor, conc_A_current[i], conc_B_current[i], conc_C_current[i], conc_N_current[i], T_current[i], ingoing)
        # Calculate the axial mass dispersion coefficient by Edwards and Richardson (1968)
        D_ax = 0.73 * D_f[situation] + (0.5*velocity_current*d_particle)/(1+((9.7*D_f[situation])/(velocity_current*d_particle)))
        # Calculate the axial thermal dispersion coefficient by Edwards and Richardson (1968)
        k_ax = (D_ax*k_f)/(D_f[situation])
        # Combining the variables
        alpha = -(delta_t * k_ax)/(rho_Cp_T[i]*(step**2))
        beta = (velocity_current*delta_t)/(step) 
        # Generating the matrix at defined position
        matrix[i,i-1] = alpha - beta
        matrix[i,i+1] = alpha
        matrix[i,i] = (1 - 2*alpha + beta)
        
    # Boundary conditions
    matrix[0,0] = 1.0
    matrix[spacesteps,spacesteps-3] = -2.0 / (6.0*step)
    matrix[spacesteps,spacesteps-2] = 9.0 / (6.0*step)
    matrix[spacesteps,spacesteps-1] = -18.0 / (6.0*step)
    matrix[spacesteps,spacesteps] = 11.0 / (6.0*step)

    return matrix, rho_Cp_T
 
def generate_matrix_concentration_solid_phase(d_particle, delta_r, spacesteps_solid, delta_t, D_s, situation): 
    # Define step within the catalytic particle
    step = delta_r
    
    # Generate vector over the catalytic particle 
    r = np.linspace((step/2),((d_particle/2)-(step/2)),spacesteps_solid)

    # Initialization of matrix
    matrix = np.zeros((spacesteps_solid,spacesteps_solid))
        
    # Loop over position in catalytic particle
    for i in range(1,spacesteps_solid-1):
        # Combining the variables
        gamma = (D_s[situation] * delta_t)/(step**2)
        
        # Generating the matrix at defined position
        matrix[i,i-1] = gamma*(-1.0 + (step / r[i]))
        matrix[i,i+1] = gamma*(-1.0 - (step / r[i]))
        matrix[i,i] = 1- matrix[i,i-1] - matrix[i,i+1]
    
    # Boundary conditions
    matrix[0,1]  = (gamma)*(-1-(step / r[0]))
    matrix[0,0] = 1.0 - matrix[0,1]
    matrix[spacesteps_solid-1,spacesteps_solid-2] = (gamma)*(-1+(step / r[-1]))
    matrix[spacesteps_solid-1,spacesteps_solid-1] = 1.0 - matrix[spacesteps_solid-1,spacesteps_solid-2]

    return matrix
 
def generate_matrix_temperature_solid_phase(d_particle, delta_r, spacesteps_solid, delta_t, k_s, rho_Cp_solid):
    # Define step within the catalytic particle
    step = delta_r
    
    # Generate vector over the catalytic particle 
    r = np.linspace((step/2),((d_particle/2)-(step/2)),spacesteps_solid)
    
    # Initialization of matrix
    matrix = np.zeros((spacesteps_solid,spacesteps_solid))
    
    # Loop over position in catalytic particle
    for i in range(1,spacesteps_solid-1):
        # Combining the variables
        gamma = (k_s * delta_t)/((step**2)*rho_Cp_solid)
        
        # Generating the matrix at defined position
        matrix[i,i-1] = gamma*(-1.0 + (step / r[i]))
        matrix[i,i+1] = gamma*(-1.0 - (step / r[i]))
        matrix[i,i] = (1- matrix[i,i-1] - matrix[i,i+1])
        
    # Boundary conditions
    matrix[0,1]  = (gamma)*(-1-(step / r[0]))
    matrix[0,0] = 1.0 - matrix[0,1]
    matrix[spacesteps_solid-1,spacesteps_solid-2] = (gamma)*(-1+(step / r[-1]))
    matrix[spacesteps_solid-1,spacesteps_solid-1] = 1.0 - matrix[spacesteps_solid-1,spacesteps_solid-2]
    
    return matrix

def velocity(velocity_inlet, L_reactor, conc_A, conc_B, conc_C, conc_N, temp, ingoing):
    
    # Calculating the current velocity in the reactor
    velocity_current = velocity_inlet #* (ingoing[0]+ingoing[1]+ingoing[2]+ingoing[3]) / (conc_A+conc_B+conc_C+conc_N) * (temp) / (ingoing[4])
    
    return velocity_current

def mass_transfer_coefficient(d_particle, velocity_current, situation, D_f, mu, rho, e):
    # Calculate the Reynolds number
    Re = (velocity_current * d_particle)/D_f[situation]
    # Calcalute the Schmidt number
    Sc = mu / (rho[situation] * D_f[situation])
    # Calculate the Sherwood number according to Gunn (1958)
    Sh = (7 - 10*e + 5*(e**2))*(1 + 0.7*(Re**0.2)*(Sc**(1/3))) + (1.3 - 2.4*e + 1.2*(e**2))*(Re**0.7)*(Sc**(1/3))
    # Calculate the mass transfer coefficient
    k_mass_transfer = (D_f[situation]*Sh) / (d_particle)

    return k_mass_transfer
    

if __name__ == '__main__':
    main()
