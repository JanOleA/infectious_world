import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from PIL import Image
import random
import time
import names
from world import World

fps = 30
max_frames = 2100
save_anim = True

im = Image.open("mixedmap_half.png")

world = World(np.array(im), day_length=700)
position_history = [world.get_actor_plotpositions()]
params_history = [world.get_actor_params()]

print("Running sim...")
for i in range(max_frames):
    if i % 10 == 0:
        print(f"{i/max_frames*100:1.3f}%", end = "\r")
    world.frame_forward()
    position_history.append(world.get_actor_plotpositions())
    params_history.append(world.get_actor_params())

print("")

fig, ax = plt.subplots(figsize=(8,8))

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
    positions = position_history[i]
    params = params_history[i]
    colors = []
    for actor in params:
        colors.append(actor["color"])
    d.set_offsets(positions)
    d.set_color(colors)
    day = i//day_length
    plt.title(f"Frame {i}, day {day}, day progress {(i%day_length)/day_length:1.2f}")
    return d,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

anim = animation.FuncAnimation(fig, animate, frames=max_frames, interval=20)
if save_anim:
    print("Saving animation...")
    anim.save("test.mp4", writer=writer)
plt.show()