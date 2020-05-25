from sim import InfectSim
import matplotlib.pyplot as plt

sim_name = "fullmixed_lockdown"
mapfile = "mixedmap.png"

params = {}

# number of days to simulate
params["sim_days"] = 50

# inhabitants in the world
params["num_inhabitants"] = 2000

# number of infected inhabitants at the beginning of the simulation
params["initial_infected"] = 10

# length of each day in frames
params["day_length"] = 700

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
params["infection_chance"] = 0.13

# how many days the infection lasts
params["infection_length"] = 6

# ratio of inhabitants that must be infected for lockdown to be put into effect
params["lockdown_ratio"] = 0.3

# chance for any inhabitant to agree with the lockdown
params["lockdown_chance"] = 0.8

# this is multiplied with the infection chance when a person is in this type of object
object_infection_modifiers = {}
object_infection_modifiers["park"] = 1
object_infection_modifiers["road"] = 4
object_infection_modifiers["house"] = 3
object_infection_modifiers["work"] = 1.2
object_infection_modifiers["common"] = 1.5
params["object_infection_modifiers"] = object_infection_modifiers

simulation = InfectSim(mapfile, params, sim_name)
simulation.run_sim()
simulation.make_infection_heatmap()
simulation.plot_infection_heatmap()
simulation.plot_SIR_graph()
simulation.plot_computation_time()
anim = simulation.animation(skipframes = 2, plot_width = 8, save_anim = True)
plt.show()