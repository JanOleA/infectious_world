import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from PIL import Image
import random
import time
import names


class World:
    def __init__(self, in_map,
                 num_inhabitants = 500,
                 worker_ratio = 0.5,
                 day_length = 500,
                 work_length_factor = 0.3,
                 workend_common_chance = 0.05,
                 home_common_chance = 0.005):
        in_map = in_map[::-1]
        self._image_map = in_map

        self._worker_ratio = worker_ratio
        self._global_time = 0
        self._day_length = day_length
        self._work_length = int(work_length_factor*day_length)
        self._workend_common_chance = workend_common_chance
        self._home_common_chance = home_common_chance

        self._map_define_types = ["p", "r", "h", "w", "c"]
        self._map_define_colors = np.array([[28,167,0], # park, green
                                            [121,121,121], # road, grey
                                            [177,115,18], # home, brown
                                            [255,81,65], # work, tomatored
                                            [149,0,167]]) # common area, purple

        self.generate_map(in_map)
        self._num_actors = num_inhabitants
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

        s = ""
        for i, line in enumerate(in_map):
            self._map.append([])
            if len(line) > self._world_size[1]:
                self._world_size[1] = len(line)

            for j, color in enumerate(line):
                progress = (i*self._world_size[1] + j)/(in_map.size/3)*100
                print(" "*len(s), end = "\r")
                s = f"{progress:3.1f}%"
                print(s, end = "\r")
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
                    raise ValueError(f"Unknown map object")

                self._map[i].append(new_object)

        self._map = np.array(self._map)
        
        print(" "*len(s), end = "\r")
        print("100%")

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

        self._actor_positions = []
        self._actor_plotpositions = []
        self._actor_params = []

        work_rolls = np.random.random(size = self._num_actors)
        
        s = ""
        for i in range(self._num_actors):
            if i%10 == 0:
                print(" "*len(s), end = "\r")
                s = f"{i/self._num_actors*100:3.1f}%"
                print(s, end = "\r")

            if i < len(self._house_list) and self._num_actors >= len(self._house_list):
                house = self._house_list[i]
            else:
                house = np.random.choice(self._house_list)

            new_person = Person(house.position[0], house.position[1], house, self._map)
            workplace = np.random.choice(self._workplace_list)
            new_person.set_workplace(self._map, workplace)
            if work_rolls[i] < self._worker_ratio:
                new_person.set_param("active_worker", True)
            house.add_dweller(new_person)
            self._actors_list.append(new_person)
            self._actor_positions.append(new_person.position)
            self._actor_plotpositions.append(new_person.plotpos)
            self._actor_params.append(new_person.params)

        print(" "*len(s), end = "\r")
        print("100%")

        self._actor_positions = np.array(self._actor_positions, dtype=np.float64)
        self._actor_plotpositions = np.array(self._actor_plotpositions, dtype=np.float64)
        self._actor_params = np.array(self._actor_params)


    def frame_forward(self):
        """ Step one frame forward in the simulation """
        day_time = self._global_time%self._day_length
        rolls = np.random.random(size=self._num_actors)

        for i, actor in enumerate(self._actors_list):
            actor_pos = actor.frame_forward()
            self._actor_positions[i] = actor_pos
            self._actor_plotpositions[i] = actor.plotpos
            self._actor_params[i] = actor.params

            roll = rolls[i]

            if actor.params["active_worker"]:
                if actor.status == "idle":
                    if (day_time >= self._day_length*0.08
                            and day_time < self._day_length*0.15):

                        if roll < 0.1:
                            actor.set_motion(self._map, actor.workplace)
                    if (day_time == int(self._day_length*0.15)
                          and isinstance(actor.current_container, House)):
                        actor.set_motion(self._map, actor.workplace)

                    if (day_time >= self._day_length*0.08+self._work_length
                            and day_time < self._day_length*0.15+self._work_length):

                        if roll < 0.3:
                            if roll < self._workend_common_chance:
                                commonarea = np.random.choice(self._commonarea_list)
                                actor.set_motion(self._map, commonarea)
                            else:
                                actor.set_motion(self._map, actor.homeplace)

                    if (day_time == int(self._day_length*0.15+self._work_length)
                            and isinstance(actor.current_container, Work)):
                        if roll < self._workend_common_chance:
                            actor.set_motion(self._map, actor.homeplace)
                        else:
                            actor.set_motion(self._map, actor.homeplace)
            else:
                if actor.status == "idle" and actor.is_home:
                    if (roll < self._home_common_chance
                            and day_time > self._day_length*0.1
                            and day_time < self._day_length*0.5):
                        commonarea = np.random.choice(self._commonarea_list)
                        actor.set_motion(self._map, commonarea)


            if day_time >= self._day_length*0.8:
                if actor.status == "idle":
                    if actor.current_container != actor.homeplace:
                        actor.set_motion(self._map, actor.homeplace)




        self._global_time += 1


    def send_N_to_work(self, N):
        inds = np.random.randint(self._num_actors, size = N)
        for ind in inds:
            actor = self._actors_list[ind]
            actor.set_motion(self._map, actor.workplace)

    
    def get_actor_positions(self):
        return self._actor_positions.copy()

    
    def get_actor_plotpositions(self):
        return self._actor_plotpositions.copy()


    def get_actor_params(self):
        return self._actor_params.copy()


    def get_actors(self):
        return self._actors_list


    def plot_world(self, ax = None):
        """ Plot the world in its current state """
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,8))
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
        """
        plt.imshow(self._image_map)
        
        ax.set_ylim(-1, self._world_size[0])
        ax.set_xlim(-1, self._world_size[1])
        
        plt.axis("equal")

    @property
    def global_time(self):
        return self._global_time

    @property
    def day_length(self):
        return self._day_length


class MapObject:
    """ Superclass for objects on the map """
    def __init__(self, x, y):
        self._params = {} # plot params
        self._params["x-size"] = 1
        self._params["y-size"] = 1
        self._roadconnections = [] # list of tuples that point to adjacent roads in (x, y) format
        self._x = x
        self._y = y

        self.camefrom = None # used for pathfinding
        self.gscore = np.inf # used for pathfinding
        self.fscore = np.inf # used for pathfinding

        self._contained_actors = []

    
    def add_actor(self, actor):
        """ Add an actor to the ones within this object """
        if actor not in self._contained_actors:
            self._contained_actors.append(actor)


    def remove_actor(self, actor):
        """ Remove an actor from this object, if it is in it """
        while actor in self._contained_actors:
            self._contained_actors.remove(actor)


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
    def position(self):
        """ return position as (x, y) format """
        return np.array((self._x, self._y))

    @property
    def roadconnections(self):
        return self._roadconnections

    @property
    def contained_actors(self):
        return self._contained_actors.copy()


class House(MapObject):
    def __init__(self, x, y):
        super().__init__(x, y)
        self._N = 0
        self._dwellers = []


    def add_dweller(self, actor):
        self._dwellers.append(actor)
        self._N += 1


    def get_dwellers(self):
        return self._dwellers


    def __str__(self):
        return f"H {self.position}"


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
        return f"R {self.position}"


    @property
    def plot_params(self):
        self._params["color"] = "grey"
        self._params["edgecolor"] = None

        return self._params


class Work(MapObject):
    def __str__(self):
        return f"W {self.position}"


    @property
    def plot_params(self):
        self._params["color"] = "tomato"
        self._params["edgecolor"] = "black"

        return self._params


class Park(MapObject):
    def __str__(self):
        return f"P {self.position}"


    @property
    def plot_params(self):
        self._params["color"] = "green"
        self._params["edgecolor"] = None

        return self._params


class CommonArea(MapObject):
    def __str__(self):
        return f"C {self.position}"


    @property
    def plot_params(self):
        self._params["color"] = "purple"
        self._params["edgecolor"] = "black"

        return self._params


class Actor:
    """ Superclass for map actors """
    def __init__(self, xhome, yhome, homeplace, map_, image_map = None):
        self._homeplace = homeplace
        self._xhome = xhome
        self._yhome = yhome
        self._workplace = None
        self._workpath = None
        self._xwork = None
        self._ywork = None
        self._params = {} # parameters
        self._x = self._xhome
        self._y = self._yhome
        self._current_container = homeplace
        self._name = None
        self._map = map_
        self._common_timer_max = 100
        self._common_timer = 0

        self._status = "idle"
        self._image_map = image_map

        self._plotpos_modifier = (np.random.random(size=2)-0.5)/3

        self._init_params()

    
    def set_workplace(self, map_, workplace):
        self._xwork, self._ywork = workplace.position
        self._workplace = workplace

        workpath = self.find_path(map_, workplace, self.homeplace)
        if workpath != False:
            self._workpath = workpath
        else:
            print("Could not find workpath for", self, f"living at {self.position}")


    def set_motion(self, map_, target):
        self._status = "moving"
        self._plotpos_modifier = np.zeros(2)
        self._x, self._y = self.current_container.position
        current_path = None
        if target == self._workplace:
            if self._current_container in self._workpath:
                current_path = self._workpath
                path_ind = current_path.index(self._current_container)
        elif target == self._homeplace:
            if self._current_container in self._workpath:
                current_path = self._workpath[::-1]
                path_ind = current_path.index(self._current_container)
        
        if current_path is None:
            current_path = self.find_path(map_, target, start = self._current_container)
            path_ind = 0

        if self._current_container == target:
            """ already at target """
            if not self in self._current_container.contained_actors:
                self._current_container.add_actor(self)
            self._x, self._y = target.position
            self._status = "idle"
            self._target_node = None
            self._path_ind = None
            self._current_path = None
            return

        try:    
            self._target_node = current_path[path_ind + 1]
        except Exception as e:
            print("Couldn't update target node:", e)
            print(self._target_node)
            s = [str(x) for x in current_path]
            print("Path:", s)
            print("Current loc:", self.position)
            print("Target was:", target)
        
        self._path_ind = path_ind
        self._current_path = current_path


    def step_values(self):
        print(f"Position = {self.position}")
        print(f"Target = {self._current_path[-1]}")
        print(f"Path ind = {self._path_ind}")
        print(f"Target node = {self._target_node}")
        plt.figure(figsize=(8,8))
        self.plot_world()
        self.plot_path()
        plt.show()


    def frame_forward(self):
        """ move the actor along the current path if status is 'moving'"""
        if isinstance(self._current_container, CommonArea):
            self._common_timer += 1
            if self._common_timer > self._common_timer_max:
                self._common_timer = 0
                self.set_motion(self._map, self.homeplace)
        else:
            self._common_timer = 0

        if self._status == "moving":
            target_vector = self._target_node.position - self.position
            target_dist = np.linalg.norm(target_vector)
            direction = target_vector/np.linalg.norm(target_dist)

            new_pos = self.position + direction*self.params["basic_speed"]
            current_node_dist = np.linalg.norm(self.position - self._current_container.position)

            if target_dist < current_node_dist:
                if self._target_node == self._current_path[-1]:
                    new_pos = self.position + direction*0.15
                    if target_dist < 0.3:
                        """ target reached """
                        self._plotpos_modifier = (np.random.random(size=2)-0.5)/3
                        self._x, self._y = self._target_node.position
                        self._current_container.remove_actor(self)
                        self._current_container = self._target_node
                        self._current_container.add_actor(self)
                        self._status = "idle"
                        self._target_node = None
                        self._path_ind = None
                        self._current_path = None
                    else:
                        self._x, self._y = new_pos[0], new_pos[1]
                    return self.position

                self._path_ind += 1
                self._current_container.remove_actor(self)
                self._current_container = self._target_node
                self._current_container.add_actor(self)
                try:
                    self._target_node = self._current_path[self._path_ind + 1]
                except Exception as e:
                    print("Couldn't update target node:", e)
                    print(self._target_node)
                    s = [str(x) for x in self._current_path]
                    print("Path:", s)
                    print("Current loc:", self.position)

            self._x, self._y = new_pos
            return self.position
        else:
            return self.position


    def set_color(self, color):
        self._params["color"] = color
    

    def find_path(self, map_, target, start = None):
        """ A* algorithm for pathfinding """
        target_position = target.position
        if start is None:
            start_position = self.position
            i, j = int(round(self._y)), int(round(self._x))
            start = map_[i,j]
        else:
            start_position = start.position

        start.camefrom = None
        start.gscore = 0
        start.fscore = self._h(start_position, target_position)
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
                    neighbor.fscore = neighbor.gscore + self._h(neighbor_position, target_position)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
                        open_set_fscores.append(neighbor.fscore)

        return False # if no path was found


    def _init_params(self):
        self._params["color"] = "black"
        self._params["basic_speed"] = 0.2


    def set_param(self, param, value):
        if param in self._params:
            self._params[param] = value
        else:
            print(f"Attempted to set param {param} for object {self} with value {value}, but key does not exist...")


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


    def plot_world(self):
        plt.imshow(self._image_map)
        wz = self._image_map.shape
        plt.ylim(-1, wz[0])
        plt.xlim(-1, wz[1])
        plt.axis("equal")


    def plot_workpath(self):
        """ Plots the path from home to work for the this actor """
        plt.plot(self.homeplace.position[0], self.homeplace.position[1], marker = "x", color = "red")
        plt.plot(self.workplace.position[0], self.workplace.position[1], marker = "x", color = "blue")
        
        path = self.workpath

        cur_pos = self.homeplace.position
        for node in path[1:]:
            next_pos = node.position
            plt.plot([cur_pos[0], next_pos[0]], [cur_pos[1], next_pos[1]], color = "black")
            cur_pos = next_pos

    
    def plot_path(self):
        """ Plots the current path for this actor """
        plt.plot(self._current_path[0].position[0], self._current_path[0].position[1], marker = "x", color = "red")
        plt.plot(self._current_path[-1].position[0], self._current_path[-1].position[1], marker = "x", color = "blue")
        
        path = self._current_path

        cur_pos = self.position
        for node in path[1:]:
            next_pos = node.position
            plt.plot([cur_pos[0], next_pos[0]], [cur_pos[1], next_pos[1]], color = "black")
            cur_pos = next_pos


    @property
    def is_home(self):
        if self._current_container == self.homeplace:
            return True
        else:
            return False

    @property
    def current_path_goal(self):
        return self._current_path[-1]

    @property
    def position(self):
        """ Return (x, y) posititon of actor """
        return np.array((self._x, self._y), dtype = np.float64)

    @property
    def plotpos(self):
        """ Return a position slightly offset to differentiate persons in animation """
        return self.position + self._plotpos_modifier

    @property
    def status(self):
        return self._status

    @property
    def homeplace(self):
        return self._homeplace

    @property
    def workplace(self):
        return self._workplace

    @property
    def workpath(self):
        return self._workpath

    @property
    def params(self):
        return self._params.copy()

    @property
    def current_container(self):
        return self._current_container


class Person(Actor):
    def __init__(self, xhome, yhome, homeplace, map_, image_map = None):
        super().__init__(xhome, yhome, homeplace, map_, image_map)
        self._name = names.get_full_name()


    def _init_params(self):
        self._params["color"] = "cyan"
        self._params["shape"] = "^"
        self._params["markersize"] = 1
        # infection status values:
        # 0 = susceptible
        # 1 = infected
        # 2 = recovered
        self._params["infection_status"] = 0
        self._params["infection_chance_modifier"] = 1 # multiplied with the infection chance of the current area the person is in
                                                      # applies for infecting _other_ people
        self._params["active_worker"] = False
        self._params["basic_speed"] = 0.45

    
    def __str__(self):
        return self._name


