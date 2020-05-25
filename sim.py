import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from PIL import Image
import time
from world import World
import seaborn as sns
import time

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
        self.infection_length = params["infection_length"]
        self.object_infection_modifiers = params["object_infection_modifiers"]

        self.im = Image.open(mapfile)
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
                           infection_length = self.infection_length,
                           object_infection_modifiers = self.object_infection_modifiers)

        max_frames = self.max_frames
        day_length = self.day_length
        num_inhabitants = self.num_inhabitants
        self.day_array = np.arange(max_frames + 1)/day_length

        self.position_history = np.zeros((max_frames + 1, num_inhabitants, 2))
        self.position_history[0] = self.world.get_actor_plotpositions()

        self.state_history = np.zeros((max_frames + 1, 3))
        self.color_history = np.empty((max_frames + 1, num_inhabitants), dtype=str)

        self.state_history[0], self.color_history[0] = self.world.get_states_and_colors()

        self.frame_time = np.zeros(max_frames)
        self._has_simulated = False
        self.infection_heatmap = None
    

    def run_sim(self, max_frames = None):
        if max_frames is None:
            max_frames = self.max_frames
        eta = 0
        s = ""

        time_begin = time.time()
        print("Running sim...")
        for i in range(max_frames):
            frame_time_init = time.time()
            if i % 10 == 0:
                print(" "*len(s), end = "\r")
                minutes = int(eta)
                seconds = eta%1*60
                s = f"{i/max_frames*100:3.1f}% | ETA = {minutes:02d}:{int(seconds):02d}"
                print(s, end = "\r")
            self.world.frame_forward()
            self.position_history[i + 1] = self.world.get_actor_plotpositions()
            self.state_history[i + 1], self.color_history[i + 1] = self.world.get_states_and_colors()
            time_now = time.time()
            self.frame_time[i] = time_now - frame_time_init
            total_elapsed_time = time_now - time_begin 
            eta = (((total_elapsed_time)/(i + 1))*(max_frames - i)//6)/10

        print(" "*len(s), end = "\r")
        minutes = total_elapsed_time/60
        seconds = minutes%1*60
        print(f"Simulation completed... Time taken: {int(minutes)}:{int(seconds)}")
        self.map = self.world.get_map()


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

        if save_plot: plt.savefig(f"plots/infection_heatmap_{self.sim_name}.pdf", dpi = 400)


    def plot_SIR_graph(self, save_plot = True):
        fig, ax = plt.subplots(figsize = (8,8))
        state_history = self.state_history
        day_array = self.day_array
        ax.plot(day_array, state_history[:,0], label = "susceptible", color = "blue")
        ax.plot(day_array, state_history[:,1], label = "infected", color = "red")
        ax.plot(day_array, state_history[:,2], label = "recovered", color = "green")
        plt.legend()
        plt.xlabel("Day")
        plt.ylabel("Inhabitants")
        if save_plot: plt.savefig(f"plots/infection_development_{self.sim_name}.pdf", dpi = 400)


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
        if save_plot: plt.savefig(f"plots/comp_time_{self.sim_name}.pdf", dpi = 400)

    
    def animation(self, skipframes = 1, fps = 30, plot_width = 13, max_frames = None, save_anim = False):
        map_ = self.map
        if max_frames is None:
            max_frames = self.max_frames

        anim_size = np.array(map_.T.shape)/len(map_[1])*plot_width
        fig, ax = plt.subplots(figsize = anim_size.astype(int))

        self.world.plot_world(ax = ax)

        initial_positions = self.position_history[0]

        d = ax.scatter(initial_positions[:,0],
                       initial_positions[:,1],
                       c = self.color_history[0], s = 5, zorder = 4)

        day_length = self.world.day_length

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
            anim.save(f"movies/movie_{self.sim_name}.mp4", writer=writer)
        return anim