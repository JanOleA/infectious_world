import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import random


class World:
    def __init__(self, in_map):
        self.generate_map(in_map)

    def generate_map(self, in_map):
        self._map = []
        self._world_size = [len(in_map), len(in_map[0])]
        self._house_list = []
        self._workplace_list = []
        self._person_list = []

        for i, line in enumerate(in_map):
            self._map.append([])
            for j, char in enumerate(line):
                if char == "r":
                    new_object = Road()
                elif char == "h":
                    new_object = House()
                elif char == "w":
                    new_object = Work()
                elif char == "p":
                    new_object = Park()
                elif char == "c":
                    new_object = CommonArea()
                else:
                    try:
                        val = int(char)
                        new_object = House(val)
                    except ValueError as e:
                        raise ValueError(f"Unknown map object: {e}")

                self._map[i].append(new_object)

    def plot_world(self):
        fig, ax = plt.subplots(figsize = (14,9))
        for i, line in enumerate(self._map):
            if len(line) > self._world_size[1]:
                self._world_size[1] = len(line)

            for j, item in enumerate(line):
                params = item.plot_params
                y = ((self._world_size[0] - 1) - i) - params["y-size"]/2
                x = j - params["x-size"]/2
                rect = patches.Rectangle((x, y),
                                         params["x-size"],
                                         params["y-size"],
                                         linewidth = 1,
                                         facecolor = params["color"],
                                         edgecolor = params["edgecolor"])

                ax.add_patch(rect)
        
        ax.set_ylim(-1, self._world_size[0])
        ax.set_xlim(-1, self._world_size[1])
                
        plt.show()
            

class MapObject:
    """ Superclass for objects on the map """
    def __init__(self):
        self._params = {}
        self._params["x-size"] = 1
        self._params["y-size"] = 1


class House(MapObject):
    def __init__(self, N = 1):
        super().__init__()
        self._N = N

    def __str__(self):
        return "H"

    @property
    def plot_params(self):
        self._params["color"] = "peru"
        self._params["edgecolor"] = "black"

        return self._params

    @property
    def N(self):
        return self._N


class Road(MapObject):
    def __str__(self):
        return "R"

    @property
    def plot_params(self):
        self._params["color"] = "grey"
        self._params["edgecolor"] = None

        return self._params


class Work(MapObject):
    def __str__(self):
        return "W"

    @property
    def plot_params(self):
        self._params["color"] = "blue"
        self._params["edgecolor"] = "black"

        return self._params


class Park(MapObject):
    def __str__(self):
        return "P"

    @property
    def plot_params(self):
        self._params["color"] = "green"
        self._params["edgecolor"] = None

        return self._params


class CommonArea(MapObject):
    def __str__(self):
        return "C"

    @property
    def plot_params(self):
        self._params["color"] = "purple"
        self._params["edgecolor"] = "black"

        return self._params


if __name__ == "__main__":
    with open("map.txt", "r") as infile:
        in_map = infile.read().split("\n")

    world = World(in_map)
    world.plot_world()