import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from PIL import Image
import random
import time
import names
from world import World
import seaborn as sns

fps = 30
max_frames = 10000
num_inhabitants = 400
save_anim = False

im = Image.open("map.png")

object_infection_modifiers = {}
object_infection_modifiers["park"] = 1
object_infection_modifiers["road"] = 6
object_infection_modifiers["house"] = 2
object_infection_modifiers["work"] = 1.5
object_infection_modifiers["common"] = 2

world = World(np.array(im),
              num_inhabitants = num_inhabitants,
              worker_ratio = 0.5,
              day_length = 500,
              work_length_factor = 0.3,
              workend_common_chance = 0.05,
              home_common_chance = 0.005,
              infection_chance = 0.3,
              infection_length = 3,
              object_infection_modifiers = object_infection_modifiers)

position_history = np.zeros((max_frames + 1, num_inhabitants, 2))
position_history[0] = world.get_actor_plotpositions()

state_history = np.zeros((max_frames + 1, 3))
state_history[0] = world.get_actor_states_num()

params_history = [world.get_actor_params()]

print("Running sim...")
for i in range(max_frames):
    if i % 10 == 0:
        print(f"{i/max_frames*100:3.1f}%", end = "\r")
    world.frame_forward()
    position_history[i + 1] = world.get_actor_plotpositions()
    state_history[i + 1] = world.get_actor_states_num()
    params_history.append(world.get_actor_params())

print(f"100%   ")

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

hmap.imshow(np.array(im)[::-1],
            aspect = hmap.get_aspect(),
            extent = hmap.get_xlim() + hmap.get_ylim(),
            zorder = 1)


fig, ax = plt.subplots(figsize = (8,8))
plt.plot(state_history[:,0], label = "susceptible", color = "blue")
plt.plot(state_history[:,1], label = "infected", color = "red")
plt.plot(state_history[:,2], label = "recovered", color = "green")
plt.legend()
plt.xlabel("Frame")
plt.ylabel("Inhabitants")

fig, ax = plt.subplots(figsize = (8,8))

print("Plotting map...")
world.plot_world(ax = ax)

initial_positions = position_history[0]

initial_params = params_history[0]
colors = []
for actor in initial_params:
    colors.append(actor["color"])

d = ax.scatter(initial_positions[:,0],
               initial_positions[:,1],
               c = colors, s = 5, zorder = 4)

day_length = world.day_length

print("Animating...")
def animate(i):
    index = i + 1
    positions = position_history[index]
    params = params_history[index]
    colors = []
    for actor in params:
        colors.append(actor["color"])
    d.set_offsets(positions)
    d.set_color(colors)
    day = index//day_length
    plt.title(f"Frame {index}, day {day}, day progress {(index)/day_length:1.2f}, infected = {state_history[index][1]}")
    return d,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

anim = animation.FuncAnimation(fig, animate, frames=max_frames, interval=20)
if save_anim:
    print("Saving animation...")
    anim.save("test.mp4", writer=writer)
plt.show()