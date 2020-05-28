import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from PIL import Image
import os, sys
import json
import time

if len(sys.argv) < 6:
    print("Usage:")
    print("python load_anim.py [sim_name] [plot_width] [skipframes] [fps] [save_anim]")
    sys.exit(1)

try:
    sim_name = str(sys.argv[1])
    plot_width = int(sys.argv[2])
    skipframes = int(sys.argv[3])
    fps = int(sys.argv[4])
    if sys.argv[5].lower() == "true":
        save_anim = True
    else:
        save_anim = False
except Exception as e:
    print(f"Couldn't parse arguments: {e}")
    sys.exit(1)

try:
    with open(f"sim_params/{sim_name}.json") as infile:
        params = json.load(infile)
except Exception as e:
    print(f"Couldn't load sim params: {e}")

load = np.load(f"output/{sim_name}/data.npy", allow_pickle = True)
map_ = load[0]
im = load[1]
position_history = load[2]
state_history = load[3]
color_history = load[4]
day_length = load[5]
if len(load) > 6:
    R_history = load[6]

max_frames = position_history.shape[0] - 1


def plot_SIR_graph(state_history, day_length, max_frames, R_history, infection_length):
    day_array = np.arange(max_frames + 1)/day_length

    fig, ax = plt.subplots(figsize = (8,8))
    l1 = ax.plot(day_array, state_history[:,0], label = "susceptible", color = "blue")
    l2 = ax.plot(day_array, state_history[:,1], label = "infected", color = "red")
    l3 = ax.plot(day_array, state_history[:,2], label = "recovered", color = "green")
    ax.set_ylabel("Inhabitants")

    ax2 = ax.twinx()

    # shift R history
    R_history = R_history[int(day_length*infection_length):]
    R_plot = np.zeros(len(day_array))
    R_plot[:len(R_history)] = R_history

    R0_est = np.nanmean(R_plot/state_history[:,0]*state_history[1,0]*(day_array[-1] - day_array))/np.mean(day_array)

    l4 = ax2.plot(day_array, R_plot, "--", color = "orange", label = "R value")
    l5 = [ax2.axhline(R0_est, day_array[0], day_array[-1], color = "black", linestyle = "--", label = "R0 estimate")]

    ax2.set_ylabel("R value / growth factor", color = "orange")
    ax2.axhline(1, day_array[0], day_array[-1], color = "orange", linestyle = "--")
    ax2.tick_params(axis='y', colors = "orange")
    ax2.set_ylim(0, 5)

    plt.xlabel("Day")
    
    lns = l1 + l2 + l3 + l4 + l5
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc = 1)


plot_SIR_graph(state_history, day_length, max_frames, R_history, params["infection_length"])

anim_size = np.array(map_.T.shape)/len(map_[1])*plot_width
fig, ax = plt.subplots(figsize = anim_size.astype(int))

plt.imshow(np.array(im)[::-1])
        
ax.set_ylim(-1, map_.shape[0])
ax.set_xlim(-1, map_.shape[1])

plt.axis("equal")

initial_positions = position_history[0]
time_start = time.time()

d = ax.scatter(initial_positions[:,0],
               initial_positions[:,1],
               c = color_history[0], s = 5, zorder = 4)
s = ""
print("Animating...")
def animate(i):
    time_elapsed = time.time() - time_start
    index = i*skipframes + 1
    s = f"{index/max_frames * 100:2.3f}%  {time_elapsed:.2f}  "
    print(s, end = "\r")
    positions = position_history[index]
    d.set_offsets(positions)
    d.set_color(color_history[index])
    day = index//day_length
    plt.title(f"Frame {index}, day {day}, day progress {((index)/day_length)%1:1.2f}, infected = {state_history[index][1]}")
    return d,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

anim = animation.FuncAnimation(fig, animate, frames=max_frames//skipframes, interval=20)
if save_anim:
    print("Saving animation...")
    anim.save(f"output/{sim_name}/movie.mp4", writer=writer)
    print("")

plt.show()