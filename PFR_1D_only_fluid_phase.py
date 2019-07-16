"""
In this function, called PFR_1D_only_fluid_phase.py,
the concentration and temperature profile of the 1D Plug Flow Reactor 
are solved and coupled.
The system consist of convection, diffusion and reaction.
The reaction term is defined as A + B --> C since it is a two component system.

Acknowledgements: this script is inspired on the work of I.A.W. Filot 
and M.P.C. van Etten
"""

# Importing the package for computations in Python
import numpy as np
# Importing the package for plotting in Python
import matplotlib.pyplot as plt

def main():
    """
    1. Defining the following types of variables:
        - Reactor variables
        - Variables for concentration profile
        - Variables for temperature profile
        - Computational stencil parameters
    """
    # Reactor variables
    L_reactor = 1.0     	# Length of reactor (m)
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
    
    # Variables for temperature profile
    rho_A = 5.0             # Density of the reactant A (kg/m3)
    rho_B = 5.0             # Density of the reactant B (kg/m3)
    rho_C = 5.0             # Density of the reactant B (kg/m3)
    rho_N = 5.0             # Density of the reactant N (kg/m3)
    rho_total = (rho_A+rho_B+rho_C+rho_N)/4       # Density of gas mixture (kg / m3)
    
    Cp_A = 1.0e3            # Specific heat capacity of the reactant A (Joule/(Kg.K))
    Cp_B = 1.0e3            # Specific heat capacity of the reactant B (Joule/(Kg.K))
    Cp_C = 1.0e3            # Specific heat capacity of the reactant B (Joule/(Kg.K))
    Cp_N = 1.0e3            # Specific heat capacity of the reactant C (Joule/(Kg.K))
    Cp_total = (Cp_A+Cp_B+Cp_C+Cp_N)/4            # Specific heat capacity of the gas mixture (Joule/(Kg.K))
    
    T_in = 500.0            # Inlet temperature (K)
    
    k_f = 0.14              # Thermal conductivity of gas mixture (Watt/(m.K))
    
    T_wall = 373.15         # Temperature of the reactor wall (K) (condensing steam)
    U = 5.0e-4              # Heat transfer coefficient (W/(m*K))
    a = 2/R_cycl

    # Kinetic parameters
    delta_H = -5.0e2        # Reaction ethalpy (J/mol)
    Ea = 8.0e4              # Activation energy (J/mol)
    R = 8.314               # Gas constant (J/(K*mol))
    k0 = 3.98e9             # Arrhenius pre-factor (1/s)
    
    # Combining the variables
    ingoing = [c_A_in, c_B_in, c_C_in, c_N_in, T_in]
    rho = [rho_A, rho_B, rho_C, rho_N, rho_total]  
    Cp = [Cp_A, Cp_B, Cp_C, Cp_N, Cp_total]
    D_f = [D_f_A, D_f_B, D_f_C, D_f_N]
    
    # Set grid points
    N_grid_reactor = 30
    
    # Computational stencil parameters
    delta_t = 1.0e-3                              # Time-step value (s)
    delta_x = L_reactor / N_grid_reactor          # Grid size for reactor transport (m)
    
    """
    2. Discretization of space and determination of the total time duration
    """
    time = 2.0                                # Total time duration
    
    spacesteps = N_grid_reactor                     # Number of steps in reactor
    x = np.linspace(0,L_reactor,spacesteps+1)       # Vector with the steps in reactor
    
    """
    3. Calculating the numerical solution
    """
    # Initial concentration profile (at t=0, only inert gas is present in the reactor)
    conc_A_current = np.zeros(spacesteps+1)
    conc_B_current = np.zeros(spacesteps+1)
    conc_C_current = np.zeros(spacesteps+1)
    conc_N_current = np.ones(spacesteps+1)*418.12
    
    # Initial temperature profile (at t=0, overall temperature in reactor is equal to T_wall)
    T_current = np.ones(spacesteps+1)*T_wall
   
    current_time = 0
    
    while current_time < time:
        y, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current =  solver(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, rho, Cp, delta_H, k0, Ea, R, U, a, T_wall)
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
    fig = plt.subplots(2,1)
    fig = plt.subplots_adjust(hspace=1.0)
    plt.suptitle('PFR 1D Fluid Phase')
    # Concentration profile
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(x,conc_A_current,label='Numerical solution of concentration A in bulk')
    plt.plot(x,conc_B_current,label='Numerical solution of concentration B')
    plt.plot(x,conc_C_current,label='Numerical solution of concentration C')
    plt.plot(x,conc_N_current,label='Numerical solution of concentration N')
    plt.xlabel('Position in reactor [m]')
    plt.ylabel('Concentration [mol/m3]')
    plt.title('Concentration of reactants in reactor')
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))
    plt.subplot(2,1,2)
    plt.grid()
    plt.plot(x,T_current,label='Numerical solution of temperature')
    plt.xlabel('Position in reactor [m]')
    plt.ylabel('Temperature [K]')
    plt.title('Temperature in the reactor')
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))
    
    """
    6. Functions needed for solving the concentration profile numerical
    """
    
def solver(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, rho, Cp, delta_H, k0, Ea, R, U, a, T_wall):
    # Generate matrices
    matrix_A = generate_matrix_concentration(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, 0)
    matrix_B = generate_matrix_concentration(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, 1)
    matrix_C = generate_matrix_concentration(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, 2)
    matrix_N = generate_matrix_concentration(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, 3)
    matrix_T, rho_Cp_T = generate_matrix_temperature(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, rho, Cp, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, 4)
    
    # Generate right-hand side vector
    vector_A = np.copy(conc_A_current)
    vector_B = np.copy(conc_B_current)
    vector_C = np.copy(conc_C_current)
    vector_N = np.copy(conc_N_current)
    vector_T = np.copy(T_current)
        
    # Process the reaction term within the right-hand side vector
    for i in range(1,spacesteps):
            heat_transfer = - (U * a * (vector_T[i]-T_wall) * delta_t )/(rho_Cp_T[i])
            reaction = (vector_A[i]*vector_B[i]*delta_t) * -(k0 * np.exp(-Ea/(R*vector_T[i])))
            vector_A[i] = vector_A[i] + reaction
            vector_B[i] = vector_B[i] + reaction
            vector_C[i] = vector_C[i] - reaction
            vector_N[i] = vector_N[i]
            vector_T[i] = vector_T[i] - (reaction*delta_H*(delta_t/(rho[4]*Cp[4]))) + heat_transfer
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
    
    return np.linspace(0,L_reactor,spacesteps+1), A, B, C, N, T
    
def generate_matrix_concentration(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, situation):
    # Define step
    step = L_reactor/(spacesteps+1)

    # Initialization of matrix
    matrix = np.zeros((spacesteps+1,spacesteps+1))
        
    # Loop over position
    for i in range(1,spacesteps):
        # Calculating the current velocity
        velocity_current = velocity(velocity_inlet, L_reactor, conc_A_current[i], conc_B_current[i], conc_C_current[i], conc_N_current[i], T_current[i], ingoing)
        # Combining the variables
        alpha = -(delta_t*D_f[situation])/(step**2)
        beta = (velocity_current*delta_t)/(step)
        # Generating the matrix
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

def generate_matrix_temperature(L_reactor, velocity_inlet, spacesteps, delta_t, D_f, k_f, rho, Cp, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, situation):
    # Define step
    step = L_reactor/(spacesteps+1)

    # Initialization of matrix
    matrix = np.zeros((spacesteps+1,spacesteps+1))
    rho_Cp_T = np.zeros(spacesteps)

    # Loop over position
    for i in range(1,spacesteps):
        molfraction = [0,0,0,0]
        # Determing average density and specific heat capacity depending on the molfraction of the reactant
        molfraction[0] = conc_A_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        molfraction[1] = conc_B_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        molfraction[2] = conc_C_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        molfraction[3] = conc_N_current[i]/(conc_A_current[i] + conc_B_current[i] + conc_C_current[i] + conc_N_current[i])
        rho_Cp_T[i] = ((rho[0]*Cp[0])+(rho[1]*Cp[1])+(rho[2]*Cp[2])+(rho[3]*Cp[3]))/4
        # Calculating the current velocity
        velocity_current = velocity(velocity_inlet, L_reactor, conc_A_current[i], conc_B_current[i], conc_C_current[i], conc_N_current[i], T_current[i], ingoing)
        # Combining the variables
        alpha = -(delta_t * k_f)/(rho_Cp_T[i]*step**2)
        beta = (velocity_current*delta_t)/(step) 
        # Generating the matrix
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
 

def velocity(velocity_inlet, L_reactor, conc_A, conc_B, conc_C, conc_N, temp, ingoing):
    
    # Calculating the current velocity in the reactor
    velocity_current = velocity_inlet #* (ingoing[0]+ingoing[1]+ingoing[2]+ingoing[3]) / (conc_A+conc_B+conc_C+conc_N) * (temp) / (ingoing[4])
    
    return velocity_current
 
if __name__ == '__main__':
    main()