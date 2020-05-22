import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from PIL import Image
import random
import time

class World:
    def __init__(self, in_map):
        in_map = in_map[::-1]
        self._map_define_types = ["p", "r", "h", "w", "c"]
        self._map_define_colors = np.array([[28,167,0],[121,121,121],[177,115,18],[72,94,255],[149,0,167]])

        self.generate_map(in_map)
        self._num_actors = len(self._actors_list)
        self.initialize_inhabitans()


    def generate_map(self, in_map):
        print("Generating map...")
        self._map = []
        self._world_size = [len(in_map), len(in_map[0])]
        self._road_list = []
        self._house_list = []
        self._workplace_list = []
        self._park_list = []
        self._commonarea_list = []
        self._actors_list = []

        for i, line in enumerate(in_map):
            self._map.append([])
            if len(line) > self._world_size[1]:
                self._world_size[1] = len(line)

            for j, color in enumerate(line):
                y = i
                x = j

                dists = self._map_define_colors - color
                dists_norm = np.zeros(len(dists))
                for k, dist in enumerate(dists):
                    length = np.linalg.norm(dist)
                    dists_norm[k] = length

                argmin = dists_norm.argmin()
                char = self._map_define_types[argmin]

                if char == "r":
                    new_object = Road(x, y)
                    self._road_list.append(new_object)
                elif char == "h":
                    new_object = House(x, y, (1, 3))
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
                    raise ValueError(f"Unknown map object")

                self._map[i].append(new_object)

        self._map = np.array(self._map)

        for house in self._house_list:
            dwellers = house.get_dwellers()
            for dweller in dwellers:
                self._actors_list.append(dweller)

        print("Calculating road connections...")
        for row in self._map:
            for item in row:
                item.set_roadconnections(self._map, self._world_size)

            if isinstance(item, House) or isinstance(item, Work):
                if len(item.roadconnections) == 0:
                    print(f"House/workplace at position {item.position} has no road connections")


    def initialize_inhabitans(self):
        print(f"Inhabitants = {self._num_actors}")
        print("Initializing actors...")
        
        s = ""
        for i, actor in enumerate(self._actors_list):
            print(" "*len(s), end = "\r")
            s = f"{i/self._num_actors*100:3.1f}%"
            print(s, end = "\r")
            workplace = np.random.choice(self._workplace_list)
            actor.set_workplace(self._map, workplace)
        print(" "*len(s), end = "\r")
        print("100%")


    def plot_world(self):
        fig, ax = plt.subplots(figsize = (11,11))
        for i, line in enumerate(self._map):
            for j, item in enumerate(line):
                params = item.plot_params
                y = i - params["y-size"]/2
                x = j - params["x-size"]/2
                rect = patches.Rectangle((x, y),
                                         params["x-size"],
                                         params["y-size"],
                                         linewidth = 1,
                                         facecolor = params["color"],
                                         edgecolor = params["edgecolor"])

                ax.add_patch(rect)


        for actor in self._actors_list:
            pos = actor.position
            params = actor.plot_params
            xrand = (np.random.random() - 0.5)/2.5
            yrand = (np.random.random() - 0.5)/2.5
            plt.plot(pos[0] + xrand, pos[1] + yrand,
                     marker = params["shape"],
                     color = params["color"],
                     markersize = params["markersize"])
        
        ax.set_ylim(-1, self._world_size[0])
        ax.set_xlim(-1, self._world_size[1])
        
        plt.axis("equal")

    
    def find_random_path(self):
        """ For testing pathfinding. Chooses a random person and a random destination and tries to find a path """
        start = np.random.choice(self._actors_list)
        destination = np.random.choice(self._workplace_list)

        path = start.find_path(self._map, destination)
        return start, destination, path
        
    
    def plot_random_path(self):
        start, destination, path = self.find_random_path()
        if not path:
            plt.plot(start.position[0], start.position[1], marker = "x", color = "red", markersize = 15)
            plt.plot(destination.position[0], destination.position[1], marker = "x", color = "blue", markersize = 15)
            return False
        else:
            plt.plot(start.position[0], start.position[1], marker = "x", color = "red")
            plt.plot(destination.position[0], destination.position[1], marker = "x", color = "blue")
            
            cur_pos = start.position
            for node in path[1:]:
                next_pos = node.position
                plt.plot([cur_pos[0], next_pos[0]], [cur_pos[1], next_pos[1]], color = "black")
                cur_pos = next_pos


class MapObject:
    """ Superclass for objects on the map """
    def __init__(self, x, y):
        self._params = {}
        self._params["x-size"] = 1
        self._params["y-size"] = 1
        self._roadconnections = [] # list of tuples that point to adjacent roads in (x, y) format
        self._x = x
        self._y = y

        self.camefrom = None # used for pathfinding
        self.gscore = np.inf # used for pathfinding
        self.fscore = np.inf # used for pathfinding

    
    @property
    def position(self):
        """ return position as (x, y) format """
        return (self._x, self._y)


    def set_roadconnections(self, map_, world_size):
        """ finds and stores all roadconnections from this object in (x,y) format """
        if self._x > 0:
            other = map_[self._y, self._x - 1]
            if isinstance(other, Road):
                self.add_roadconnection((-1, 0))
                other.add_roadconnection((1, 0))
            if self._y > 0:
                other = map_[self._y - 1, self._x - 1]
                if isinstance(other, Road):
                    self.add_roadconnection((-1, -1))
                    other.add_roadconnection((1, 1))
            if self._y < world_size[0] - 1:
                other = map_[self._y + 1, self._x - 1]
                if isinstance(other, Road):
                    self.add_roadconnection((-1, 1))
                    other.add_roadconnection((1, -1))

        if self._x < world_size[1] - 1:
            other = map_[self._y, self._x + 1]
            if isinstance(other, Road):
                self.add_roadconnection((1, 0))
                other.add_roadconnection((-1, 0))
            if self._y > 0:
                other = map_[self._y - 1, self._x + 1]
                if isinstance(other, Road):
                    self.add_roadconnection((1, -1))
                    other.add_roadconnection((-1, 1))
            if self._y < world_size[0] - 1:
                other = map_[self._y + 1, self._x + 1]
                if isinstance(other, Road):
                    self.add_roadconnection((1, 1))
                    other.add_roadconnection((-1, -1))

        if self._y > 0:
            other = map_[self._y - 1, self._x]
            if isinstance(other, Road):
                self.add_roadconnection((0, -1))
                other.add_roadconnection((0, 1))
        if self._y < world_size[0] - 1:
            other = map_[self._y + 1, self._x]
            if isinstance(other, Road):
                self.add_roadconnection((0, 1))
                other.add_roadconnection((0, -1))


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

        self._dwellers = []


        if isinstance(N, tuple):
            N = np.random.randint(N[0], N[1] + 1)

        for i in range(N):
            self._dwellers.append(Person(x, y, self))


    def get_dwellers(self):
        return self._dwellers


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
        self._params["color"] = "red"
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


class Actor:
    """ Superclass for map actors """
    def __init__(self, xhome, yhome, homeplace):
        self._homeplace = homeplace
        self._xhome = xhome
        self._yhome = yhome
        self._workplace = None
        self._workpath = None
        self._xwork = None
        self._ywork = None
        self._params = {}
        self._x = self._xhome
        self._y = self._yhome

        self._status = "idle"

        self._init_params()

    
    def set_workplace(self, map_, workplace):
        self._xwork, self._ywork = workplace.position
        self._workplace = workplace

        workpath = self.find_path(map_, workplace)
        if workpath != False:
            self._workpath = workpath
        else:
            print("Could not find workpath for", self)


    def set_color(self, color):
        self._params["color"] = color
    

    def find_path(self, map_, target):
        """ A* algorithm for pathfinding """
        target_position = target.position
        array_position = np.array([self._x, self._y])
        array_target_position = np.array(target_position)

        i, j = self._y, self._x

        start = map_[i,j]

        start.camefrom = None
        start.gscore = 0
        start.fscore = self._h(array_position, array_target_position)
        open_set = [start]
        open_set_fscores = [start.fscore]
        closed_set = []

        while len(open_set) > 0:
            ind = np.argmin(open_set_fscores)
            current = open_set[ind]
            if current == target:
                for item in open_set + closed_set:
                    item.fscore = np.inf
                    item.gscore = np.inf

                return self._reconstruct_path(target)

            ind = open_set.index(current)
            open_set.pop(ind)
            open_set_fscores.pop(ind)

            closed_set.append(current)

            for connection in current.roadconnections:
                neighbor_position = np.array(current.position) + np.array(connection)
                neighbor = map_[neighbor_position[1], neighbor_position[0]]

                tentative_gscore = current.gscore + np.linalg.norm(np.array(connection))

                if not isinstance(neighbor, Road) and neighbor != target:
                    tentative_gscore += 1000000

                if tentative_gscore < neighbor.gscore:
                    neighbor.camefrom = current
                    neighbor.gscore = tentative_gscore
                    neighbor.fscore = neighbor.gscore + self._h(neighbor_position, array_target_position)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
                        open_set_fscores.append(neighbor.fscore)

        return False


    def _init_params(self):
        self._params["color"] = "black"


    def _h(self, position, target_position):
        """ Heuristic function for A* algorithm. Estimates the shortest
        possible distance to the target position.
        position and target_position must be Numpy arrays
        """
        return np.linalg.norm(target_position - position)


    def _reconstruct_path(self, target):
        """ Finalize the path for the A* algorithm 
        returns the finished path as a list of road objects
        """
        path = [target]
        current = target
        while current.camefrom is not None:
            path.append(current.camefrom)
            current = current.camefrom

        return path[::-1]


    @property
    def position(self):
        """ Return (x, y) posititon of actor """
        return (self._x, self._y)

    
    @property
    def status(self):
        return self._status


class Person(Actor):
    def _init_params(self):
        self._params["color"] = "cyan"
        self._params["shape"] = "^"
        self._params["markersize"] = 1


    @property
    def plot_params(self):
        return self._params


if __name__ == "__main__":
    im = Image.open("map.png")

    world = World(np.array(im))
    print("Plotting...")
    world.plot_world()
    plt.show()

