"""
In this function, called PFR_1D_with_solid_phase_homogeneous_two_components.py,
the concentration and temperature profile of the 1D Plug Flow Reactor 
are solved and coupled for both fluid and solid phase.
The solid phase is assumed to be homogeneous. 
The system consist of convection, diffusion and reaction.
The reaction term is defined as A + B --> C.

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
    R_cycl = 0.10           # Radius of the reactor (m)
    
    # Variables for concentration profile
    c_A_in = 209.06         # Inlet feed concentration of A (mol/m3)
    c_B_in = 209.06         # Inlet feed concentration of B (mol/m3)
    c_C_in = 0.0            # Inlet feed concentration of C (mol/m3)
    c_N_in = 0.0            # Inlet feed concentration of N (mol/m3)
    
    D_f_A = 1e-5            # Diffusion coefficient of reactant A in gas (m2/s)
    D_f_B = 1e-5            # Diffusion coefficient of reactant B in gas (m2/s)
    D_f_C = 1e-5            # Diffusion coefficient of reactant B in gas (m2/s)
    D_f_N = 1e-5            # Diffusion coefficient of reactant N in gas (m2/s)
    D_f_total = (D_f_A+D_f_B+D_f_C+D_f_N)/4         # Diffusion coefficient of gas mixture (m2/s)
    
    mu = 1e-5               # Dynamic viscocity of gas mixture (m2/s)
    
    # Variables for temperature profile
    rho_A = 1.0             # Density of the reactant A (kg/m3)
    rho_B = 1.0             # Density of the reactant B (kg/m3)
    rho_C = 1.0             # Density of the reactant B (kg/m3)
    rho_N = 1.0             # Density of the reactant N (kg/m3)
    rho_total = (rho_A+rho_B+rho_C+rho_N)/4       # Density of gas mixture (kg / m3)
    
    Cp_A = 1e3              # Specific heat capacity of the reactant A (Joule/(Kg.K))
    Cp_B = 1e3              # Specific heat capacity of the reactant B (Joule/(Kg.K))
    Cp_C = 1e3              # Specific heat capacity of the reactant B (Joule/(Kg.K))
    Cp_N = 1e3              # Specific heat capacity of the reactant C (Joule/(Kg.K))
    Cp_total = (Cp_A+Cp_B+Cp_C+Cp_N)/4            # Specific heat capacity of the gas mixture (Joule/(Kg.K))
    
    T_in = 500.0            # Inlet temperature (K)
    
    k_f = 50e-3             # Thermal conductivity of gas mixture (Watt/(m.K))
    
    T_wall = 373.15         # Temperature of the reactor wall (K) (condensing steam)
    U = 0.0005              # Heat transfer coefficient (W/(m*K))
    a = 2/R_cycl
    
    # Variables for solid phase 
    e = 0.5                 # Void fraction present in the reactor (-)
    d_particle = 0.01       # Diameter of particle (m)

    # Kinetic parameters
    delta_H = -5e2          # Reaction ethalpy (J/mol)
    Ea = 80e3               # Activation energy (J/mol)
    R = 8.314               # Gas constant (J/(K*mol))
    k0 = 3.98e9             # Arrhenius pre-factor (1/s)
    
    # Combining the variables
    ingoing = [c_A_in, c_B_in, c_C_in, c_N_in, T_in]
    rho = [rho_A, rho_B, rho_C, rho_N, rho_total]  
    Cp = [Cp_A, Cp_B, Cp_C, Cp_N, Cp_total]
    D_f = [D_f_A, D_f_B, D_f_C, D_f_N, D_f_total]
    
    # Set grid points
    N_grid_reactor = 30
    
    # Computational stencil parameters
    delta_t = 1e-3                                # Time-step value (s)
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
    conc_A_surface = np.zeros(spacesteps+1)
    
    # Initial temperature profile (at t=0, overall temperature in reactor is equal to T_wall)
    T_current = np.ones(spacesteps+1)*T_wall
   
    current_time = 0
    
    while current_time < time:
        y, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, conc_A_current_surface =  solver(L_reactor, d_particle, velocity_inlet, spacesteps, delta_x, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, conc_A_surface, ingoing, rho, Cp, delta_H, k0, Ea, R, a, T_wall, e, mu)
        current_time = current_time + delta_t
        
    """
    4. Calculating the molfractions of the reactants
    """
    molfraction_A = conc_A_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current + conc_A_current_surface)
    molfraction_B = conc_B_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current + conc_A_current_surface)
    molfraction_C = conc_C_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current + conc_A_current_surface)
    molfraction_N = conc_N_current / (conc_A_current + conc_B_current + conc_C_current + conc_N_current + conc_A_current_surface)
    molfraction_A_surface = conc_A_current_surface / (conc_A_current + conc_B_current + + conc_C_current + conc_N_current + conc_A_current_surface)

    """
    5. Plotting the numerical and analytical solution
    """
    fig = plt.subplots(2,1)
    fig = plt.subplots_adjust(hspace=1.0)
    plt.suptitle('PFR 1D Fluid Phase with homogeneous solid catalytic particle')
    # Concentration profile
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(x,conc_A_current,label='Numerical solution of concentration A in bulk')
    plt.plot(x,conc_B_current,label='Numerical solution of concentration B')
    plt.plot(x,conc_C_current,label='Numerical solution of concentration C')
    plt.plot(x,conc_N_current,label='Numerical solution of concentration N')
    plt.plot(x,conc_A_current_surface, label='Numerical solution of concentration A at surface of solid phase')
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
    
def solver(L_reactor, d_particle, velocity_inlet, spacesteps, delta_x, delta_t, D_f, k_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, conc_A_surface, ingoing, rho, Cp, delta_H, k0, Ea, R, a, T_wall, e, mu):
    # Generate matrices
    matrix_A = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 0)
    matrix_B = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 1)
    matrix_C = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 2)
    matrix_N = generate_matrix_concentration_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 3)
    matrix_T, rho_Cp_T = generate_matrix_temperature_fluid_phase(L_reactor, velocity_inlet, spacesteps, delta_x, delta_t, D_f, k_f, rho, Cp, conc_A_current, conc_B_current, conc_C_current, conc_N_current, T_current, ingoing, d_particle, 4)
    
    # Generate right-hand side vector
    vector_A = np.copy(conc_A_current)
    vector_B = np.copy(conc_B_current)
    vector_C = np.copy(conc_C_current)
    vector_N = np.copy(conc_N_current)
    vector_T = np.copy(T_current)
    vector_A_surface = np.copy(conc_A_surface)
        
    # Process the reaction term within the right-hand side vector
    for i in range(1,spacesteps):
            # Calculate the reaction rate constant at the defined position
            k_reaction = 7.0 #(k0 * np.exp(-Ea/(R*vector_T[i])))
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
            k_mass_transfer = mass_transfer_coefficient(d_particle, velocity_current, 0, D_f, mu, rho_i, e)
            # Calculate the wall-to-bed heat transfer coefficient for fluid phase heat transfer term
            if Re < 40:
                Nu_wall = 0.6 * (Re**0.5)
            else: 
                Nu_wall = 0.2 * (Re**0.8)
            h_wall = (Nu_wall * k_f) / d_particle
            # Define the heat transfer term for fluid phase
            heat_transfer = (h_wall * a * (vector_T[i] - T_wall) * delta_t ) / (rho[4]*Cp[4])
            # Calculate the reaction term
            reaction = (k_mass_transfer*k_reaction*vector_A[i]*vector_B[i])/(k_mass_transfer+k_reaction)
            vector_A[i] = vector_A[i] - reaction
            vector_B[i] = vector_B[i] - reaction
            vector_C[i] = vector_C[i] + reaction
            vector_N[i] = vector_N[i]
            vector_T[i] = vector_T[i] - (reaction*delta_H*(delta_t/(rho[4]*Cp[4]))) - heat_transfer
            vector_A_surface[i] = (k_mass_transfer * vector_A[i] * vector_B[i])/(k_reaction + k_mass_transfer)
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
    
    return np.linspace(0,L_reactor,spacesteps+1), A, B, C, N, T, vector_A_surface
    
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

def velocity(velocity_inlet, L_reactor, conc_A, conc_B, conc_C, conc_N, temp, ingoing):
    
    # Calculating the current velocity in the reactor
    velocity_current = velocity_inlet #* (ingoing[0]+ingoing[1]+ingoing[2]+ingoing[3]) / (conc_A+conc_B+conc_C+conc_N) * (temp) / (ingoing[4])
    
    return velocity_current

def mass_transfer_coefficient(d_particle, velocity_current, situation, D_f, mu, rho, e):
    # Calculate the Reynolds number
    Re = (velocity_current * d_particle)/D_f[situation]
    # Calcalute the Schmidt number
    Sc = mu / (rho * D_f[situation])
    # Calculate the Sherwood number according to Gunn (1958)
    Sh = (7 - 10*e + 5*(e**2))*(1 + 0.7*(Re**0.2)*(Sc**(1/3))) + (1.3 - 2.4*e + 1.2*(e**2))*(Re**0.7)*(Sc**(1/3))
    # Calculate the mass transfer coefficient
    k_mass_transfer = (D_f[situation]*Sh) / (d_particle)

    return k_mass_transfer
 
if __name__ == '__main__':
    main()