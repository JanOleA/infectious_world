import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from PIL import Image
import time
from world import World
from calc_deathrates import death_rate
import seaborn as sns
import os


class InfectSim:
    def __init__(self, mapfile, params, sim_name):
        """ Initialize the simulation.
        
        Keyword arguments:
        mapfile -- path to an image file containing a map
        params -- dictionary containing simulation parameters
        sim_name -- title for the current simulation
        """
        self.num_inhabitants = params["num_inhabitants"]
        self.day_length = params["day_length"]
        self.max_frames = int(params["sim_days"]*self.day_length)
        self.worker_ratio = params["worker_ratio"]
        self.work_length_factor = params["work_length_factor"]
        self.workend_common_chance = params["workend_common_chance"]
        self.home_common_chance = params["home_common_chance"]
        self.infection_chance = params["infection_chance"]
        self.initial_infected = params["initial_infected"]
        self.infection_length = params["infection_length"]
        self.object_infection_modifiers = params["object_infection_modifiers"]
        self.lockdown_ratio = params["lockdown_ratio"]
        self.lockdown_chance = params["lockdown_chance"]
        self.disease_health_impact = params["disease_health_impact"]
        self.allow_natural_deaths = params["allow_natural_deaths"]
        self.life_expectancy = params["life_expectancy"]
        self.rebirth_chance = params["rebirth_chance"]
        self.allow_rebirths = params["allow_rebirths"]

        self.expected_death_rate = death_rate(5 - self.disease_health_impact)*self.infection_length

        print(f"You have provided a disease health impact of {self.disease_health_impact} and ", end = "")
        print(f"an infection length of {self.infection_length}. With a normal health of 5 this gives ", end = "")
        print(f"the disease an expected death rate of {self.expected_death_rate:.4f}.")

        self.im = Image.open(mapfile)
        self.mapfile = mapfile
        self.map_array = np.array(self.im)

        self.sim_name = sim_name
        
        self.world = World(self.map_array,
                           num_inhabitants = self.num_inhabitants,
                           worker_ratio = self.worker_ratio,
                           day_length = self.day_length,
                           work_length_factor = self.work_length_factor,
                           workend_common_chance = self.workend_common_chance,
                           home_common_chance = self.home_common_chance,
                           infection_chance = self.infection_chance,
                           initial_infected = self.initial_infected,
                           infection_length = self.infection_length,
                           object_infection_modifiers = self.object_infection_modifiers,
                           disease_health_impact = self.disease_health_impact,
                           allow_natural_deaths = self.allow_natural_deaths,
                           life_expectancy = self.life_expectancy,
                           rebirth_chance = self.rebirth_chance, 
                           allow_rebirths = self.allow_rebirths)

        max_frames = self.max_frames
        day_length = self.day_length
        num_inhabitants = self.num_inhabitants
        self.day_array = np.arange(max_frames + 1)/day_length

        self.position_history = np.zeros((max_frames + 1, num_inhabitants, 2))
        self.position_history[0] = self.world.get_actor_plotpositions()

        self.state_history = np.zeros((max_frames + 1, 5))
        self.color_history = np.empty((max_frames + 1, num_inhabitants), dtype=str)

        self.state_history[0], self.color_history[0] = self.world.get_states_and_colors()

        self.R_history = np.full(max_frames + 1, np.nan)
        self.mult_history = np.full(max_frames + 1, np.nan)

        self.frame_time = np.zeros(max_frames)
        self._has_simulated = False
        self.infection_heatmap = None
        self.recovered_stats = None
    

    def initiate_lockdown(self):
        self.world.set_behaviors("stay_home", self.lockdown_chance)

    
    def deactivate_lockdown(self):
        self.world.set_behaviors("normal")


    def run_sim(self, max_frames = None):
        if max_frames is None:
            max_frames = self.max_frames
        eta = 0
        R0 = np.nan
        s = ""

        R_eval_time = int(self.infection_length*self.day_length*1.1)
        R0_max_time = int(self.infection_length*self.day_length*1.2)
        gfactor_interval = int(self.day_length)

        current_infected = self.initial_infected
        lockdown_initiated = False

        time_begin = time.time()
        print("Running sim...")

        """ MAIN SIMULATION LOOP """
        for i in range(max_frames):
            frame_time_init = time.time()
            self.state_history[i + 1], self.color_history[i + 1] = self.world.frame_forward()
            self.position_history[i + 1] = self.world.get_actor_plotpositions()
            self.recovered_stats = self.world.get_recovered_stats()

            current_infected = self.state_history[i + 1][1]
            if (current_infected / self.num_inhabitants > self.lockdown_ratio
                    and not lockdown_initiated):
                self.initiate_lockdown()
                lockdown_initiated = True

            if self.world.global_time > R_eval_time:
                recovered_recently = self.recovered_stats[:,0][(self.world.global_time - self.recovered_stats[:,1]) < R_eval_time]
                if len(recovered_recently) > 0:
                    self.R_history[i + 1] = np.average(recovered_recently)
                    if self.world.global_time < R0_max_time:
                        R0 = np.average(self.R_history[R_eval_time + 1: min(i + 2, R0_max_time)])
                    
            if self.world.global_time > gfactor_interval:
                if self.state_history[int(i - gfactor_interval + 1)][1] != 0:
                    self.mult_history[i + 1] = current_infected/self.state_history[i - gfactor_interval + 1][1]

            if i % 10 == 0:
                minutes = int(eta)
                seconds = eta%1*60
                s = f"{i/max_frames*100:3.1f}% | ETA = {minutes:02d}:{int(seconds):02d} | Current infected = {current_infected} | R0 (this run) = {R0:3.3f}       "
                print(s, end = "\r")

            time_now = time.time()
            self.frame_time[i] = time_now - frame_time_init
            total_elapsed_time = time_now - time_begin 
            eta = (((total_elapsed_time)/(i + 1))*(max_frames - i)//6)/10


        """ CLEANING UP AND SAVING DATA """

        print(" "*len(s), end = "\r")
        minutes = total_elapsed_time/60
        seconds = minutes%1*60
        print(f"Simulation completed... Time taken: {int(minutes):02d}:{int(seconds):02d}")

        print(f"Final states:")
        print(f" S: {self.state_history[-1][0]}")
        print(f" I: {self.state_history[-1][1]}")
        print(f" R: {self.state_history[-1][2]}")
        print(f"DI: {self.state_history[-1][3]}")
        print(f"DN: {self.state_history[-1][4]}")

        self.map = self.world.get_map()

        if not os.path.exists(f"{os.getcwd()}/output"):
            os.mkdir(f"{os.getcwd()}/output")
        if not os.path.exists(f"{os.getcwd()}/output/{self.sim_name}"):
            os.mkdir(f"{os.getcwd()}/output/{self.sim_name}")

        self.output_dir = f"{os.getcwd()}/output/{self.sim_name}"

        output_data = [self.map,
                       self.im,
                       self.position_history,
                       self.state_history,
                       self.color_history,
                       self.day_length,
                       self.R_history]

        print("Saving data...")
        np.save(f"{self.output_dir}/data.npy", output_data)


    def calculate_R0(self, iterations = 5):
        """ Runs the beginning of the simulation _iterations_ number of times
        and returns the average R0 value.
        """
        R0 = np.nan
        s = ""

        R_eval_time = int(self.infection_length*self.day_length*1.1)
        R0_max_time = int(self.infection_length*self.day_length*1.2)

        R0_list = []

        print(f"Calculating R0 with {iterations} iterations...")
        s = ""

        """ MAIN LOOP """
        for itr in range(iterations):
            R_history = np.full(R0_max_time + 1, np.nan)
            print(f"Iteration {itr}...", end = "\r")
            for i in range(R0_max_time):
                self.world.frame_forward()
                recovered_stats = self.world.get_recovered_stats()

                if self.world.global_time > R_eval_time:
                    recovered_recently = recovered_stats[:,0][(self.world.global_time - recovered_stats[:,1]) < R_eval_time]
                    if len(recovered_recently) > 0:
                        R_history[i + 1] = np.average(recovered_recently)
                        if self.world.global_time < R0_max_time:
                            R0 = np.average(R_history[R_eval_time + 1: min(i + 2, R0_max_time)])
                        
                if i%10 == 0:
                    s = f"Iteration {itr}... {i/R0_max_time*100:2.2f}%   R0: {R0:2.3f}"
                    print(s, end = "\r")

            print(" "*len(s), end = "\r")
            print(f"Iteration {itr}: {R0:2.3f}")
            R0_list.append(R0)
            R0 = np.nan
            self.world.reset()
        print(f"Average R0: {np.mean(R0_list):2.3f}")

        return np.mean(R0_list)


    def make_infection_heatmap(self):
        map_ = self.map
        infection_heatmap = np.zeros(map_.shape)
        for i, row in enumerate(map_):
            for j, item in enumerate(row):
                infection_heatmap[i, j] = item.infection_occurences

        self.infection_heatmap = infection_heatmap


    def plot_infection_heatmap(self, save_plot = True):
        plt.figure(figsize = (8,8))
        hmap = sns.heatmap(self.infection_heatmap[::-1],
                           cmap = cm.OrRd,
                           alpha = 0.8,
                           zorder = 2)
        plt.axis("equal")

        hmap.imshow(self.map_array,
                    aspect = hmap.get_aspect(),
                    extent = hmap.get_xlim() + hmap.get_ylim(),
                    zorder = 1)

        plt.title("Infection hotspots")

        if save_plot: plt.savefig(f"{self.output_dir}/infection_heatmap.pdf", dpi = 400)


    def plot_SIR_graph(self, save_plot = True):
        fig, ax = plt.subplots(figsize = (8,8))
        state_history = self.state_history
        day_array = self.day_array

        infected = state_history[:,1]
        dead_inf = infected + state_history[:,3]
        recovered = dead_inf + state_history[:,2]
        susceptible = recovered + state_history[:,0]
        dead_natural = susceptible + state_history[:,4]
        l1 = [ax.fill_between(day_array, infected, label = "infected", color = "red", alpha = 0.3)]
        l2 = [ax.fill_between(day_array, infected, dead_inf, label = "dead (from infection)", color = "black", alpha = 0.3)]
        l3 = [ax.fill_between(day_array, dead_inf, recovered, label = "recovered", color = "green", alpha = 0.3)]
        l4 = [ax.fill_between(day_array, recovered, susceptible, label = "susceptible", color = "blue", alpha = 0.3)]
        if np.sum(state_history[:,4]) >= 1:
            l5 = [ax.fill_between(day_array, susceptible, dead_natural, label = "dead (natural)", color = "purple", alpha = 0.3)]

        ax.set_ylabel("Inhabitants")

        ax2 = ax.twinx()

        # shift R history
        R_history = self.R_history[int(self.day_length*self.infection_length):]
        R_plot = np.zeros(len(day_array))
        R_plot[:len(R_history)] = R_history

        R0_est = np.nanmean(R_plot/state_history[:,0]*state_history[1,0]*(day_array[-1] - day_array))/np.mean(day_array)

        l6 = ax2.plot(day_array, R_plot, "--", color = "orange", label = "R value")
        l7 = ax2.plot(day_array, self.mult_history, "--", color = "grey", label = "growth factor", linewidth = 0.5)

        ax2.set_ylabel("R value / growth factor", color = "orange")
        ax2.axhline(1, day_array[0], day_array[-1], color = "orange", linestyle = "--")
        ax2.tick_params(axis='y', colors = "orange")
        ax2.set_ylim(0, 5)

        plt.xlabel("Day")
        
        lns = l1 + l2 + l3 + l4
        if np.sum(state_history[:,4]) >= 1:
            lns = lns + l5
        lns = lns + l6 + l7
        
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc = 2)

        plt.title("SIR plot")

        print(f"R0 estimate: {R0_est}")

        if save_plot: plt.savefig(f"{self.output_dir}/infection_development.pdf", dpi = 400)


    def plot_death_rate(self, save_plot = True):
        plt.figure(figsize = (8,8))
        dr_recovered = np.full(self.state_history[:,3].shape, np.inf)
        dr_recovered[self.state_history[:,2] != 0] = (self.state_history[:,3])[self.state_history[:,2] != 0]/((self.state_history[:,2])[self.state_history[:,2] != 0])
        dr_cumulative = self.state_history[:,3]/(self.state_history[:,2] + self.state_history[:,1] + self.state_history[:,4])

        plt.plot(self.day_array, dr_recovered, color = "red", label = "Death rate (vs. recovered)")
        plt.plot(self.day_array, dr_cumulative, color = "green", label = "Death rate (vs. cumulative num. of infections)")
        plt.hlines(self.expected_death_rate, self.day_array[0], self.day_array[-1], color = "black", linestyle = "--", label = "Expected rate")
        plt.xlabel("Day")
        plt.ylabel("Death rate")
        plt.ylim(0, 1)
        plt.legend()

        plt.title("Death rate (only deaths while infected)")

        if save_plot: plt.savefig(f"{self.output_dir}/death_rate.pdf", dpi = 400)


    def plot_computation_time(self, save_plot = True):
        plt.figure(figsize = (8,8))
        day_comptimes = []
        start = 0
        end = self.day_length
        while start < self.max_frames:
            day_comptimes.append(np.sum(self.frame_time[start:end]))
            start += self.day_length
            end += self.day_length

        plt.plot(day_comptimes)
        plt.xlabel("Day")
        plt.ylabel("Computation time")
        plt.title("Computation time")
        if save_plot: plt.savefig(f"{self.output_dir}/comp_time.pdf", dpi = 400)

    
    def animation(self, skipframes = 1, fps = 30, plot_width = 13, max_frames = None, save_anim = False):
        map_ = self.map
        if max_frames is None:
            max_frames = self.max_frames

        anim_size = np.array(map_.T.shape + np.array([1,0]))/len(map_[1])*plot_width
        fig, ax = plt.subplots(figsize = anim_size.astype(int))

        self.world.plot_world(ax = ax)
        ax.set_xlim(left = -2)

        initial_positions = self.position_history[0]

        d = ax.scatter(initial_positions[:,0],
                       initial_positions[:,1],
                       c = self.color_history[0], s = 5, zorder = 4)

        day_length = self.day_length

        print("Animating...")
        def animate(i):
            index = i*skipframes + 1
            positions = self.position_history[index]
            d.set_offsets(positions)
            d.set_color(self.color_history[index])
            day = index//day_length
            plt.title(f"Frame {index}, day {day}, day progress {((index)/day_length)%1:1.2f}, infected = {self.state_history[index][1]}")
            return d,

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

        anim = animation.FuncAnimation(fig, animate, frames=max_frames//skipframes, interval=20)
        if save_anim:
            print("Saving animation...")
            anim.save(f"{self.output_dir}/movie.mp4", writer=writer)
        return anim