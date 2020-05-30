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
    R_history = R_history[int(day_length*infection_length):]
    R_plot = np.zeros(len(day_array))
    R_plot[:len(R_history)] = R_history

    l6 = ax2.plot(day_array, R_plot, "--", color = "orange", label = "R value", alpha = 0.5)

    ax2.set_ylabel("R value / growth factor", color = "orange")
    ax2.axhline(1, day_array[0], day_array[-1], color = "orange", linestyle = "--")
    ax2.tick_params(axis='y', colors = "orange")
    ax2.set_ylim(0, 5)

    plt.xlabel("Day")

    lns = l1 + l2 + l3 + l4
    if np.sum(state_history[:,4]) >= 1:
        lns = lns + l5
    lns = lns + l6

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc = 2)


plot_SIR_graph(state_history, day_length, max_frames, R_history, params["infection_length"])

anim_size = np.array(map_.T.shape*np.array([1,0.5]) + np.array([1,0]))/len(map_[1])*plot_width
fig, axs = plt.subplots(1, 2, figsize = anim_size.astype(int))

ax = axs[0]

ax.imshow(np.array(im)[::-1])
        
ax.set_ylim(-1, map_.shape[0])
ax.axis("equal")
ax.set_xlim(-2, map_.shape[1])

initial_positions = position_history[0]
time_start = time.time()

d = ax.scatter(initial_positions[:,0],
               initial_positions[:,1],
               c = color_history[0], s = 5, zorder = 4)

def draw_betweens(state_history, day_length, max_frames, infection_length, end_index, ax, ax2, plot_arrays):
    day_array, infected, dead_inf, recovered, susceptible, dead_natural = plot_arrays
    l1 = [ax.fill_between(day_array[:end_index], infected[:end_index], label = "infected", color = "red", alpha = 0.3)]
    l2 = [ax.fill_between(day_array[:end_index], infected[:end_index], dead_inf[:end_index], label = "dead (from infection)", color = "black", alpha = 0.3)]
    l3 = [ax.fill_between(day_array[:end_index], dead_inf[:end_index], recovered[:end_index], label = "recovered", color = "green", alpha = 0.3)]
    l4 = [ax.fill_between(day_array[:end_index], recovered[:end_index], susceptible[:end_index], label = "susceptible", color = "blue", alpha = 0.3)]
    if np.sum(state_history[:,4]) >= 1:
        l5 = [ax.fill_between(day_array[:end_index], susceptible[:end_index], dead_natural[:end_index], label = "dead (natural)", color = "purple", alpha = 0.3)]
    
    ax.set_xlim(day_array[0], day_array[end_index])
    ax2.set_xlim(day_array[0], day_array[end_index])

    lns = l1 + l2 + l3 + l4
    if np.sum(state_history[:,4]) >= 1:
        lns = lns + l5
    lns = lns

    return lns

day_array = np.arange(max_frames + 1)/day_length

# shift R history
R_history = R_history[int(day_length*params["infection_length"]):]
R_plot = np.zeros(len(day_array))
R_plot[:len(R_history)] = R_history

infected = state_history[:,1]
dead_inf = infected + state_history[:,3]
recovered = dead_inf + state_history[:,2]
susceptible = recovered + state_history[:,0]
dead_natural = susceptible + state_history[:,4]
plot_arrays = [day_array, infected, dead_inf, recovered, susceptible, dead_natural]
axs[1].set_ylabel("Inhabitants")
axs[1].set_xlim(day_array[0], day_array[1])
axs[1].set_xlabel("Day")

ax2 = axs[1].twinx()
l6, = ax2.plot(day_array[:1], R_plot[:1], "--", color = "orange", label = "R value", alpha = 0.5)
ax2.set_ylim(0, 5)
ax2.axhline(1, day_array[0], day_array[-1], color = "orange", linestyle = "--")
ax2.set_xlim(day_array[0], day_array[1])
ax2.set_ylabel("R value / growth factor", color = "orange")
ax2.tick_params(axis='y', colors = "orange")
lns = draw_betweens(state_history, day_length, max_frames, params["infection_length"], 1, axs[1], ax2, plot_arrays)

lns = lns + [l6]

labs = [l.get_label() for l in lns]
axs[1].legend(lns, labs, loc = 2)

s = ""
print("Animating...")
def animate(i):
    time_elapsed = time.time() - time_start
    index = i*skipframes + 1
    eta = (((time_elapsed)/(index))*(max_frames - index)//6)/10
    minutes = int(eta)
    seconds = eta%1*60
    s = f"{index/max_frames * 100:2.3f}% | ETA = {minutes:02d}:{int(seconds):02d}     "
    print(s, end = "\r")
    positions = position_history[index]
    d.set_offsets(positions)
    d.set_color(color_history[index])
    l6.set_data(day_array[:index], R_plot[:index])
    day = index//day_length
    ax.set_title(f"Frame {index}, day {day}, day progress {((index)/day_length)%1:1.2f}, infected = {state_history[index][1]}")
    axs[1].collections.clear()
    ax2.collections.clear()
    draw_betweens(state_history, day_length, max_frames, params["infection_length"], index, axs[1], ax2, plot_arrays)
    return d, l6

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

anim = animation.FuncAnimation(fig, animate, frames=max_frames//skipframes, interval=20)
if save_anim:
    print("Saving animation...")
    anim.save(f"output/{sim_name}/movie.mp4", writer=writer)
    print("")

print(" "*len(s))
print("Done...")

plt.show()