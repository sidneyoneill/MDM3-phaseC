# MDM3-phaseC

This repository contains different scripts for various agent-based modelling simulations. 

To execute the student tracking simulation over a single day, the run_student_tracking_simulation.py needs to be run. From within this, it uses daily_student_data.py to generate schedules for student agents, and also uses the student_tracking_simulation.py file which is the main body of code for making the agent based model. 

To execute run the weekly simulation, the run_weekly_simulation.py needs to be run. From within this, it uses weekly_student_data.py to generate schedules for student agents, and also uses the weekly_library_simulation.py file which is the main body of code for making the weekly agent based model.

The run_weekly_library_simulation_knowledge.py and weekly_library_simulation_knowledge.py files work like the above case. However, this allows for varying levels of agents to have knowledge of the live occupancies of each library in the network, which is used to influence their movements. 

The weekly_library_metrics.py is a script that outputs results of numerous metrics about the agent based model. This should be after the weekly simulation with knowledge script is executed - it analyses the results after changing the proportion of students that have knowledge of the occupancies of each library. 

The bar_chart_weekly_simulation.py is a script that produces an animated bar chart of the library occupancies from the weekly simulation. The run_weekly_library_simulation.py has a line of code that writes the occupancy counts to a csv file. Once this weekly simulation is executed and the csv file generated, the bar chart scipt can be run to generate this animation. 


These simulations and outputs rely on the library_locations.csv and library_times.csv excel files. These contain capacities, latitude and longitude values for each library (used to get an accurate network layout), and the times to walk between each library (which is used for edge weightings in the network). 

This repository also contains a Library EDA folder - this makes plots to analyse the occupancy levels that the current university system has captured (via manual counting) over the last years. 

There is also a folder for a simplified ODE model in this repository.
