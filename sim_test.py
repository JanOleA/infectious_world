import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from PIL import Image
import random
import time
import names
from world import World
import seaborn as sns
import time

fps = 30
max_frames = 35000
num_inhabitants = 2000
day_length = 700
worker_ratio = 0.5
work_length_factor = 0.3
workend_common_chance = 0.03
home_common_chance = 0.005
infection_chance = 0.15
infection_length = 7
save_anim = True
anim_skipframes = 5

sim_name = "fullmixed"

im = Image.open("mixedmap.png")

object_infection_modifiers = {}
object_infection_modifiers["park"] = 1
object_infection_modifiers["road"] = 4
object_infection_modifiers["house"] = 3
object_infection_modifiers["work"] = 1.2
object_infection_modifiers["common"] = 1.5

world = World(np.array(im),
              num_inhabitants = num_inhabitants,
              worker_ratio = 0.5,
              day_length = day_length,
              work_length_factor = work_length_factor,
              workend_common_chance = workend_common_chance,
              home_common_chance = home_common_chance,
              infection_chance = infection_chance,
              infection_length = infection_length,
              object_infection_modifiers = object_infection_modifiers)

day_array = np.arange(max_frames + 1)/day_length

position_history = np.zeros((max_frames + 1, num_inhabitants, 2))
position_history[0] = world.get_actor_plotpositions()

state_history = np.zeros((max_frames + 1, 3))
color_history = np.empty((max_frames + 1, num_inhabitants), dtype=str)

state_history[0], color_history[0] = world.get_states_and_colors()

frame_time = np.zeros(max_frames)

params_history = [world.get_actor_params()]

eta = 0
s = ""

print("Running sim...")
for i in range(max_frames):
    time_init = time.time()
    if i % 10 == 0:
        print(" "*len(s), end = "\r")
        minutes = int(eta)
        seconds = eta%1*60
        s = f"{i/max_frames*100:3.1f}% | ETA = {minutes:02d}:{int(seconds):02d}"
        print(s, end = "\r")
    world.frame_forward()
    position_history[i + 1] = world.get_actor_plotpositions()
    state_history[i + 1], color_history[i + 1] = world.get_states_and_colors()
    frame_time[i] = time.time() - time_init
    if i > 0:
        avg_time = np.mean(frame_time[:i])
        eta = (avg_time*(max_frames - i)//6)/10

print(" "*len(s), end = "\r")
print("Simulation completed...")

map_ = world.get_map()
infection_rates = np.zeros(map_.shape)

for i, row in enumerate(map_):
    for j, item in enumerate(row):
        infection_rates[i, j] = item.infection_occurences

plt.figure(figsize = (8,8))
hmap = sns.heatmap(infection_rates[::-1],
                   cmap = cm.OrRd,
                   alpha = 0.8,
                   zorder = 2)
plt.axis("equal")

hmap.imshow(np.array(im),
            aspect = hmap.get_aspect(),
            extent = hmap.get_xlim() + hmap.get_ylim(),
            zorder = 1)

plt.savefig(f"plots/infection_heatmap_{sim_name}.pdf", dpi = 400)

fig, ax = plt.subplots(figsize = (8,8))
plt.plot(day_array, state_history[:,0], label = "susceptible", color = "blue")
plt.plot(day_array, state_history[:,1], label = "infected", color = "red")
plt.plot(day_array, state_history[:,2], label = "recovered", color = "green")
plt.legend()
plt.xlabel("Day")
plt.ylabel("Inhabitants")
plt.savefig(f"plots/infection_development_{sim_name}.pdf", dpi = 400)

plt.figure(figsize = (8,8))
day_comptimes = []
start = 0
end = day_length
while start < max_frames:
    day_comptimes.append(np.sum(frame_time[start:end]))
    start += day_length
    end += day_length

plt.plot(day_comptimes)
plt.xlabel("Day")
plt.ylabel("Computation time")
plt.savefig(f"plots/comp_time_{sim_name}.pdf", dpi = 400)

anim_size = np.array(map_.T.shape)/len(map_[1])*13
fig, ax = plt.subplots(figsize = anim_size.astype(int))

print("Plotting map...")
world.plot_world(ax = ax)

initial_positions = position_history[0]

d = ax.scatter(initial_positions[:,0],
               initial_positions[:,1],
               c = color_history[0], s = 5, zorder = 4)

day_length = world.day_length

print("Animating...")
def animate(i):
    index = i*anim_skipframes + 1
    positions = position_history[index]
    d.set_offsets(positions)
    d.set_color(color_history[index])
    day = index//day_length
    plt.title(f"Frame {index}, day {day}, day progress {((index)/day_length)%1:1.2f}, infected = {state_history[index][1]}")
    return d,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

anim = animation.FuncAnimation(fig, animate, frames=max_frames//anim_skipframes, interval=20)
if save_anim:
    print("Saving animation...")
    anim.save(f"movies/movie_{sim_name}.mp4", writer=writer)
plt.show()