from sim import InfectSim
import matplotlib.pyplot as plt
import json

sim_name = "test"
mapfile = "map.png"

params = {"mapfile": mapfile}

# number of days to simulate
params["sim_days"] = 10

# inhabitants in the world
params["num_inhabitants"] = 400

# number of infected inhabitants at the beginning of the simulation
params["initial_infected"] = 1

# length of each day in frames
params["day_length"] = 500

# ratio of inhabitants who are active workers
params["worker_ratio"] = 0.6

# the ratio of the days that each person spends working (approx.)
params["work_length_factor"] = 0.3

# chance for a person at work to stop by a commonarea on the way home
params["workend_common_chance"] = 0.03

# chance (each frame) that a person who is not an active worker and is at home will go to a commonarea
params["home_common_chance"] = 0.005

# expected chance of a person infecting someone if they spend one entire day in the same area together (without any infection modifiers)
# the actual chance is per frame: infection_chance/day_length
params["infection_chance"] = 0.7

# how many days the infection lasts on average
params["infection_length"] = 3

# ratio of inhabitants that must be infected for lockdown to be put into effect
params["lockdown_ratio"] = 1

# chance for any inhabitant to go into lockdown
params["lockdown_chance"] = 0

# chance for any inhabitant in lockdown to return to normal behavior on any day
# params["lockdown_break_chance"] = 0.1 TODO: Implement

# this is multiplied with the infection chance when a person is in this type of object
object_infection_modifiers = {}
object_infection_modifiers["park"] = 1
object_infection_modifiers["road"] = 3.5
object_infection_modifiers["house"] = 2
object_infection_modifiers["work"] = 1.3
object_infection_modifiers["common"] = 1.4
params["object_infection_modifiers"] = object_infection_modifiers

with open(f"sim_params/{sim_name}.json", "w") as outfile:
    json.dump(params, outfile, indent = 4)

simulation = InfectSim(mapfile, params, sim_name)
simulation.calculate_R0(iterations = 10)
simulation.run_sim()
simulation.make_infection_heatmap()
simulation.plot_infection_heatmap()
simulation.plot_SIR_graph()
simulation.plot_computation_time()
anim = simulation.animation(skipframes = 2, plot_width = 10, save_anim = False)
plt.show()