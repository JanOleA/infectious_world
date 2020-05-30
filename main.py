from sim import InfectSim
import matplotlib.pyplot as plt
import json

sim_name = "test"
mapfile = "map.png"

params = {"mapfile": mapfile}

# number of days to simulate
params["sim_days"] = 50

# inhabitants in the world
params["num_inhabitants"] = 400

# number of infected inhabitants at the beginning of the simulation
params["initial_infected"] = 10

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
params["infection_chance"] = 0.4

# how many days the infection lasts on average
params["infection_length"] = 5

# ratio of inhabitants that must be infected for lockdown to be put into effect
params["lockdown_ratio"] = 1

# chance for any inhabitant to go into lockdown when it is put into effect
params["lockdown_chance"] = 0

# chance for any inhabitant in lockdown to return to normal behavior on any day
# params["lockdown_break_chance"] = 0.1 TODO: Implement

# how much the disease will reduce the health of an infected person
params["disease_health_impact"] = 3.84557

# whether or not people can die natural deaths
params["allow_natural_deaths"] = True

# approximately how long people are expected to live if neatural deaths is on
params["life_expectancy"] = 20

# chance of a person to be reborn each day
# this is multiplied with the current fraction of the population that is dead
params["rebirth_chance"] = 0.9

# whether or not people can be reborn
params["allow_rebirths"] = True

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

if __name__ == "__main__":
    simulation = InfectSim(mapfile, params, sim_name)
    simulation.calculate_R0(iterations = 2)
    simulation.run_sim()
    simulation.make_infection_heatmap()
    simulation.plot_infection_heatmap()
    simulation.plot_SIR_graph()
    simulation.plot_death_rate()
    simulation.plot_computation_time()
    anim = simulation.animation(skipframes = 2, plot_width = 10, save_anim = False)
    plt.show()