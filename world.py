import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import random


class World:
    def __init__(self, in_map):
        self.generate_map(in_map)


    def generate_map(self, in_map):
        print("Generating map...")
        self._map = []
        self._world_size = [len(in_map), len(in_map[0])]
        self._road_list = []
        self._house_list = []
        self._workplace_list = []
        self._park_list = []
        self._commonarea_list = []

        for i, line in enumerate(in_map):
            self._map.append([])
            if len(line) > self._world_size[1]:
                self._world_size[1] = len(line)

            for j, char in enumerate(line):
                y = i
                x = j

                if char == "r":
                    new_object = Road(x, y)
                    self._road_list.append(new_object)
                elif char == "h":
                    new_object = House(x, y)
                    self._house_list.append(new_object)
                elif char == "w":
                    new_object = Work(x, y)
                    self._workplace_list.append(new_object)
                elif char == "p":
                    new_object = Park(x, y)
                    self._park_list.append(new_object)
                elif char == "c":
                    new_object = CommonArea(x, y)
                    self._commonarea_list.append(new_object)
                else:
                    try:
                        val = int(char)
                        new_object = House(x, y, N = val)
                    except ValueError as e:
                        raise ValueError(f"Unknown map object: {e}")

                self._map[i].append(new_object)

        self._map = np.array(self._map)

        for row in self._map:
            for item in row:
                item.set_roadconnections(self._map, self._world_size)


    def plot_world(self):
        fig, ax = plt.subplots(figsize = (14,9))
        for i, line in enumerate(self._map):
            for j, item in enumerate(line):
                params = item.plot_params
                roadconnections = item.roadconnections
                y = i - params["y-size"]/2
                x = j - params["x-size"]/2
                rect = patches.Rectangle((x, y),
                                         params["x-size"],
                                         params["y-size"],
                                         linewidth = 1,
                                         facecolor = params["color"],
                                         edgecolor = params["edgecolor"],
                                         zorder = 1)

                ax.add_patch(rect)

                for roadconnection in roadconnections:
                    plt.arrow(j, i, roadconnection[1]/2.5, roadconnection[0]/2.5, width = 0.04, zorder = 4,
                              color = "black")
        
        ax.set_ylim(-1, self._world_size[0])
        ax.set_xlim(-1, self._world_size[1])
                
        plt.show()
            

class MapObject:
    """ Superclass for objects on the map """
    def __init__(self, x, y):
        self._params = {}
        self._params["x-size"] = 1
        self._params["y-size"] = 1
        self._roadconnections = [] # list of tuples that point to adjacent roads
        self._x = x
        self._y = y


    def set_roadconnections(self, map_, world_size):
        """ finds and stores all roadconnections from this object """
        if self._x > 0:
            other = map_[self._y, self._x - 1]
            if isinstance(other, Road):
                self.add_roadconnection((0, -1))
                other.add_roadconnection((0, 1))

        if self._x < world_size[1] - 1:
            other = map_[self._y, self._x + 1]
            if isinstance(other, Road):
                self.add_roadconnection((0, 1))
                other.add_roadconnection((0, -1))

        if self._y > 0:
            other = map_[self._y - 1, self._x]
            if isinstance(other, Road):
                self.add_roadconnection((-1, 0))
                other.add_roadconnection((1, 0))
        if self._y < world_size[0] - 1:
            other = map_[self._y + 1, self._x]
            if isinstance(other, Road):
                self.add_roadconnection((1, 0))
                other.add_roadconnection((-1, 0))


    def add_roadconnection(self, direction):
        if not self.has_roadconnection(direction):
            self._roadconnections.append(direction)


    def has_roadconnection(self, direction):
        """ True/False whether or not the object has the given road connection
        """
        if direction in self._roadconnections:
            return True
        else:
            return False


    @property
    def roadconnections(self):
        return self._roadconnections


class House(MapObject):
    def __init__(self, x, y, N = 1):
        super().__init__(x, y)
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