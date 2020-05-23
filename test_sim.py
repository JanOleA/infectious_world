import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from PIL import Image
import random
import time
import names
from world import World

fps = 15
max_frames = 300

im = Image.open("map.png")

world = World(np.array(im))
world.send_N_to_work(10)
position_history = [world.get_actor_positions()]

print("Running sim...")
for i in range(max_frames):
    world.frame_forward()
    position_history.append(world.get_actor_positions())

fig, ax = plt.subplots(figsize=(8,8))

print("Plotting map...")
world.plot_world(ax = ax)

initial_positions = position_history[0]

d = ax.scatter(initial_positions[:,0],
               initial_positions[:,1],
               c = "white", s = 5, zorder = 4)

print("Animating...")
def animate(i):
    positions = position_history[i]
    d.set_offsets(positions)
    plt.title(f"Frame {i}")
    return d,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

anim = animation.FuncAnimation(fig, animate, frames=max_frames, interval=20)
print("Saving animation...")
anim.save("test.mp4", writer=writer)
plt.show()