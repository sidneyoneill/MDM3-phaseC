import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import matplotlib.dates as mdates
from datetime import datetime, timedelta


""" 
Libraries and study spaces (compartments)



N_t : vector consisting of the number of students in each compartment at time t
inflow : vector consisting of the rate of change in the number of students going into compartment i from all other compartments j, given by : sum(beta_ji * (N_j(1 - N_i/capacity_i)))
outflow : vector consisting of the rate of change in the number of students going from compartment i to all other compartments j, given by : sum(beta_ij * N_i)
dNdt : vector consisting of the ODEs describing the rate of change in occupancy for each compartment, given by dNdt = inflow - outflow
WHEN A COMPARTMENT REACHES ITS CLOSING TIME, THE EQUATION FOR dNdt CHANGES SO THAT dNdt = -outflow, I.E. THE INFLOW GOES TO ZERO

beta_ij : preference value dictating the proportion of students that move from compartment i to compartment j

N_0 : vector consisting of the number of students in each compartment at time t = 0 (INITIAL CONDITION)

"""



# N_t = vector consisting of the number of students in each compartment at time t

def load_compartment_data(file_path):
    compartment_data = pd.read_csv(file_path, usecols=['ID','LibraryName ','Capacity'])
    return compartment_data


def get_beta_matrix():
    # the beta values relating to home are all set to one for initial testing
    distance_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 6, 13, 4, 1, 6, 10,  8, 3, 6, 2], 
                            [1, 6, 0,  7, 2, 7, 2,  5, 11, 8, 7, 7],
                            [1, 13, 7, 0, 9, 10, 7, 2, 7, 8, 5, 9],
                            [1, 4, 2, 9, 0, 5, 3, 7, 12, 7, 8, 6],
                            [1, 1, 7, 10, 5, 0, 5, 10, 7, 2, 5, 1],
                            [1, 6, 2, 7, 3, 5, 0, 5, 9, 6, 5, 5],
                            [1, 10, 5, 2, 7, 10, 5, 0, 6, 7, 4, 8],
                            [1, 8, 11, 7, 12, 7, 9, 6, 0, 5, 4, 6],
                            [1, 3, 8, 8, 7, 2, 6, 7, 5, 0, 3, 1],
                            [1, 6, 7, 5, 8, 5, 5, 4, 4, 3, 0, 4],
                            [1, 2, 7, 9, 6, 1, 5, 8, 6, 1, 4, 0]
                            ])
    

    # Initial beta values for other compartments will be inversely proportional to the product of time and distance between compartments, i.e. 1/(distance * t)
    beta_matrix = 1. / distance_matrix
    np.fill_diagonal(beta_matrix,0)
    return beta_matrix


def ODE_inflow(t, beta_matrix, N_t, capacity, num_compartments, opening_times, closing_times):
    inflow = np.zeros(num_compartments)

    # Calculate how the fullness of each compartment affects the inflow rate
    capacity_factor = np.ones(num_compartments) - np.divide(N_t,capacity)
    
    # Calculate inflow rate for each compartment
    for i in range(num_compartments):
        inflow[i] = capacity_factor[i] * np.sum(beta_matrix[:,i] * N_t)

    # Determine which libraries are open at current t
    open_libraries = (t >= opening_times) & (t < closing_times)

    # Set inflow rate to zero for compartments that are not open 
    inflow = open_libraries * inflow

    return inflow


def ODE_outflow(t, beta_matrix, N_t):
    return np.sum(beta_matrix, axis = 1) * N_t


def ODE_system(t, N_t, beta_matrix, capacity, num_compartments, opening_times, closing_times):
    beta_matrix = beta_matrix / t
    return ODE_inflow(t, beta_matrix, N_t, capacity, num_compartments, opening_times, closing_times) - ODE_outflow(t, beta_matrix, N_t)


def solve_ODE():
    """
    Library : refers to each of the 11 libraries/study spaces
    Compartments : refers to home AND all 11 libraries/study spaces

    """
    # Load data containing library names and capacities
    file_path = r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\library_locations.csv"
    compartment_data = load_compartment_data(file_path)

    # Get capacity of each library
    capacity = compartment_data['Capacity'].to_numpy()

    # Set capacity of home
    student_count = np.array([3000]) # total number of students in model
    capacity = np.concat((student_count,capacity)) # Index 0 represents home


    # opening time and closing time for compartments that are open 24/7 are set to 0 and 25 respectively 
    # entry in closing_times set to 25 so that time_test < closing_times always returns True
    opening_times = np.array([0,0,9,9,9,9,9,9,8,9,8,8])
    closing_times = np.array([25,25,19,19,22,19,22,22,22,22,22,22])

    # Get total number of compartments
    num_compartments = capacity.shape[0]

    # Get preference values for each compartment (proportion of students moving from compartment i to compartment j)
    beta_matrix = get_beta_matrix()

    # Initial conditions
    N_0 = np.array([student_count[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    # Generate time points from 08:00 (8 hours) to 20:00 (20 hours) in steps of 15 minutes (0.25 hours)
    t_eval = np.arange(8, 20.25, 0.25)  # 20.25 ensures the last point is included
    t_span = (t_eval[0], t_eval[-1])  # Define start and end times


    # Solve system of ODEs
    sol = solve_ivp(ODE_system, t_span, N_0, t_eval=t_eval, args=(beta_matrix, capacity, num_compartments, opening_times, closing_times))
    

    # Plot solution to ODE

    # Create time labels in HH:MM format
    time_labels = [datetime(2024, 4, 2, int(h), int((h % 1) * 60)) for h in t_eval]  # Date is arbitrary, chose Phase C deadline
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_labels, sol.y[0], label="Home")
    plt.plot(time_labels, sol.y[1], label = compartment_data['LibraryName '][0])
    plt.plot(time_labels, sol.y[2], label = compartment_data['LibraryName '][1])
    plt.plot(time_labels, sol.y[3], label = compartment_data['LibraryName '][2])
    plt.plot(time_labels, sol.y[4], label = compartment_data['LibraryName '][3])
    plt.plot(time_labels, sol.y[5], label = compartment_data['LibraryName '][4])
    plt.plot(time_labels, sol.y[6], label = compartment_data['LibraryName '][5])
    plt.plot(time_labels, sol.y[7], label = compartment_data['LibraryName '][6])
    plt.plot(time_labels, sol.y[8], label = compartment_data['LibraryName '][7])
    plt.plot(time_labels, sol.y[9], label = compartment_data['LibraryName '][8])
    plt.plot(time_labels, sol.y[10], label = compartment_data['LibraryName '][9])
    plt.plot(time_labels, sol.y[11], label = compartment_data['LibraryName '][10])

    # Format x-axis as time (HH:MM)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Show every hour
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  

    # Show every half-hour
    # # Explicitly set major ticks at whole hours (08:00, 09:00, ...)
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(8, 21)))  

    # # Add minor ticks at 30-minute intervals (08:30, 09:30, ...)
    # plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))  

  
    plt.xticks(rotation=45) # Rotate the x-axis labels for better readability
    plt.xlabel('Time of Day')
    plt.ylabel('Number of students')
    plt.legend()
    plt.title('Number of students in each compartment against time')
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # # opening time and closing time for compartments that are open 24/7 are set to 0 and 25 respectively 
    # # entry in closing_times set to 25 so that time_test < closing_times always returns True
    # opening_times = np.array([0,0,9,9,9,9,9,9,8,9,8,8])
    # closing_times = np.array([25,25,19,19,22,19,22,22,22,22,22,22])
 
    
    # time_test = 8.5

    # open_test = (time_test >= opening_times) & (time_test < closing_times)

    # test = np.ones(12)

    # final = open_test * test

    solve_ODE()

    print("DONE!")
