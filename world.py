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
                 home_common_chance = 0.005,
                 infection_chance = 0.1,
                 initial_infected = 10,
                 infection_length = 2,
                 object_infection_modifiers = None):

        """ Initialize the world and all parameters, generate the map and
        initialize the inhabitants
        """

        in_map = in_map[::-1]
        self._image_map = in_map

        self._worker_ratio = worker_ratio
        self._global_time = 0
        self._day_length = day_length
        self._work_length = int(work_length_factor*day_length)
        self._workend_common_chance = workend_common_chance
        self._home_common_chance = home_common_chance
        self._infection_chance = infection_chance
        self._infection_chance_per_frame = infection_chance/day_length
        self._initial_infected = initial_infected
        self._infection_length_days = infection_length
        self._infection_length_frames = infection_length*day_length

        if object_infection_modifiers is None:
            object_infection_modifiers = {}
            object_infection_modifiers["park"] = 1
            object_infection_modifiers["road"] = 1
            object_infection_modifiers["house"] = 1
            object_infection_modifiers["work"] = 1
            object_infection_modifiers["common"] = 1
        self._object_infection_modifiers = object_infection_modifiers

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
        """ Load the map and create all the map objects """
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
                    inf_mod = self._object_infection_modifiers["road"]
                    new_object = Road(x, y, infection_chance_modifier = inf_mod)
                    self._road_list.append(new_object)
                elif char == "h":
                    inf_mod = self._object_infection_modifiers["house"]
                    new_object = House(x, y, infection_chance_modifier = inf_mod)
                    self._house_list.append(new_object)
                elif char == "w":
                    inf_mod = self._object_infection_modifiers["work"]
                    new_object = Work(x, y, infection_chance_modifier = inf_mod)
                    self._workplace_list.append(new_object)
                elif char == "p":
                    inf_mod = self._object_infection_modifiers["park"]
                    new_object = Park(x, y, infection_chance_modifier = inf_mod)
                    self._park_list.append(new_object)
                elif char == "c":
                    inf_mod = self._object_infection_modifiers["common"]
                    new_object = CommonArea(x, y, infection_chance_modifier = inf_mod)
                    self._commonarea_list.append(new_object)
                else:
                    raise ValueError(f"Unknown map object")

                self._map[i].append(new_object)

        self._map = np.array(self._map)

        print(f"Number of houses = {len(self._house_list)}")
        
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
        """ Generate all the inhabitants, and set them to workers with the 
        chance as defined in world initialization
        """
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
            new_person.set_preferred_commonarea(self._commonarea_list)
            house.add_dweller(new_person)
            self._actors_list.append(new_person)
            self._actor_positions.append(new_person.position)
            self._actor_plotpositions.append(new_person.plotpos)
            self._actor_params.append(new_person.params)

        initial_infect_inds = np.random.choice(np.arange(self._num_actors), self._initial_infected)

        for i in initial_infect_inds:
            actor = self._actors_list[i]
            actor.set_infected(infection_length = self._infection_length_frames)
            self._actor_params[i] = actor.params

        print(" "*len(s), end = "\r")
        print("100%")

        self._actor_positions = np.array(self._actor_positions, dtype=np.float64)
        self._actor_plotpositions = np.array(self._actor_plotpositions, dtype=np.float64)
        self._actor_params = np.array(self._actor_params)
        self._recovered_stats = np.zeros((self._num_actors, 2))


    def frame_forward(self):
        """ Step one frame forward in the simulation """
        day_time = self._global_time%self._day_length
        rolls = np.random.random(size=self._num_actors)

        self.infect()

        ### this loop defines the main behaviour of persons in this world
        for i, actor in enumerate(self._actors_list):
            actor_pos = actor.frame_forward(self._global_time)
            self._actor_positions[i] = actor_pos
            self._actor_plotpositions[i] = actor.plotpos
            actor_params = actor.params
            self._actor_params[i] = actor_params

            if actor_params["infection_status"] == 2:
                self._recovered_stats[i] = (actor_params["infected_others"], actor_params["became_immune"])

            roll = rolls[i]

            behavior = actor.params["behavior"]

            if behavior == "normal":
                self._normal_actor_behaviour(actor, roll, day_time)
            elif behavior == "stay_home":
                self._stay_home_behaviour(actor, roll, day_time)

        self._global_time += 1


    def _normal_actor_behaviour(self, actor, roll, day_time):
        if actor.params["active_worker"]:
            if actor.status == "idle":
                ### Leave home for work in the morning
                if (day_time >= self._day_length*0.08
                        and day_time < self._day_length*0.15):
                    if roll < 0.1:
                        actor.set_motion(actor.workplace)
                if (day_time == int(self._day_length*0.15)
                      and isinstance(actor.current_container, House)):
                    ### if the person still at home by day_length*0.15, go to work
                    actor.set_motion(actor.workplace)

                ### leave work and go either home or to a commonarea
                if (day_time >= self._day_length*0.08+self._work_length
                        and day_time < self._day_length*0.15+self._work_length):
                    if roll < 0.3:
                        if roll < self._workend_common_chance*0.3:
                            if np.random.random() < 0.9:
                                commonarea = actor.preffered_commonarea
                            else:
                                commonarea = np.random.choice(self._commonarea_list)
                            actor.set_motion(commonarea)
                        else:
                            actor.set_motion(actor.homeplace)

                ### if still at work by day_length*0.16 + work_length, go home or to a commonarea
                if (day_time == int(self._day_length*0.15+self._work_length)
                        and isinstance(actor.current_container, Work)):
                    if roll < self._workend_common_chance:
                        if np.random.random() < 0.9:
                            commonarea = actor.preffered_commonarea
                        else:
                            commonarea = np.random.choice(self._commonarea_list)
                        actor.set_motion(commonarea)
                    else:
                        actor.set_motion(actor.homeplace)
        else:
            ### for homestayers, every frame has a chance of sending them to a commonarea
            if actor.status == "idle" and actor.is_home:
                if (roll < self._home_common_chance
                        and day_time > self._day_length*0.1
                        and day_time < self._day_length*0.5):
                    if np.random.random() < 0.9:
                        commonarea = actor.preffered_commonarea
                    else:
                        commonarea = np.random.choice(self._commonarea_list)
                    actor.set_motion(commonarea)

        ### at day_length*0.8 or more, send the actor home if it is not at home (and not moving somewhere right now)
        if day_time >= self._day_length*0.8:
            if actor.status == "idle":
                if actor.current_container != actor.homeplace:
                    actor.set_motion(actor.homeplace)


    def _stay_home_behaviour(self, actor, roll, day_time):
        if not actor.is_home:
            if not actor.current_path_goal == actor.homeplace:
                actor.set_motion(actor.homeplace)


    def set_behaviors(self, new_behavior, chance):
        """ reset all actors to normal behavior, then set the new behavior with
        the given chance
        """
        rolls = np.random.random(size=self._num_actors)
        for roll, actor in zip(rolls, self._actors_list):
            if roll < chance:
                actor.set_param("behavior", new_behavior)
            else:
                actor.set_param("behavior", "normal")

    
    def set_behaviors_conditional(self, new_behavior, old_behavior, chance = 1):
        """ set the behavior to the new behavior with the given chance, provided
        the old behavior is correct
        """
        rolls = np.random.random(size=self._num_actors)
        for roll, actor in zip(rolls, self._actors_list):
            if roll < chance and actor.params["behavior"] == old_behavior:
                actor.set_param("behavior", new_behavior)


    def infect(self):
        """ Checks all the objects in the map and spreads disease according
        to the world parameters to other actors in boxes with infected actors
        in them.
        """
        for row in self._map:
            for item in row:
                location_modifier = item.infection_chance_modifier
                location_chance = location_modifier*self._infection_chance_per_frame
                num_in_current = len(item.contained_actors)
                for i in range(num_in_current):
                    for j in range(i + 1, num_in_current):
                        roll = np.random.random()
                        one = item.contained_actors[i]
                        other = item.contained_actors[j]
                        one_params = one.params
                        other_params = other.params

                        if one_params["infection_status"] == 1:
                            if other_params["infection_status"] == 0:
                                if roll < location_chance:
                                    one.increase_param("infected_others")
                                    other.set_infected(infection_length = self._infection_length_frames)
                                    item.infection_occurences += 1

                        elif other_params["infection_status"] == 1:
                            if one_params["infection_status"] == 0:
                                if roll < location_chance:
                                    other.increase_param("infected_others")
                                    one.set_infected(infection_length = self._infection_length_frames)
                                    item.infection_occurences += 1


    def send_N_to_work(self, N):
        inds = np.random.randint(self._num_actors, size = N)
        for ind in inds:
            actor = self._actors_list[ind]
            actor.set_motion(actor.workplace)

    
    def get_actor_positions(self):
        return self._actor_positions.copy()

    
    def get_actor_plotpositions(self):
        return self._actor_plotpositions.copy()


    def get_actor_params(self):
        return self._actor_params.copy()


    def get_actors(self):
        return self._actors_list

    
    def get_map(self):
        return self._map

    
    def get_actor_states_num(self):
        states = np.zeros(3)
        for actor in self._actors_list:
            state = actor.params["infection_status"]
            states[int(state)] += 1

        return states

    
    def get_states_and_colors(self):
        states = np.zeros(3)
        colors = []
        for actor in self._actors_list:
            params = actor.params
            state = params["infection_status"]
            states[int(state)] += 1
            colors.append(params["color"])

        return states, colors


    def get_recovered_stats(self):
        return self._recovered_stats[self._recovered_stats[:,1] != 0]


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
    def __init__(self, x, y, infection_chance_modifier = 1):
        self._params = {} # plot params
        self._params["x-size"] = 1
        self._params["y-size"] = 1
        self._roadconnections = [] # list of tuples that point to adjacent roads in (x, y) format
        self._x = x
        self._y = y
        self._infection_chance_modifier = infection_chance_modifier

        self.camefrom = None # used for pathfinding
        self.gscore = np.inf # used for pathfinding
        self.fscore = np.inf # used for pathfinding

        self._contained_actors = []
        self.infection_occurences = 0

    
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
        return self._contained_actors

    @property
    def infection_chance_modifier(self):
        return self._infection_chance_modifier


class House(MapObject):
    def __init__(self, x, y, infection_chance_modifier = 1):
        super().__init__(x, y, infection_chance_modifier)
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
        self._stored_paths = []
        self._stored_paths_usage = []
        self._preffered_commonarea = None

        self._common_timer_max = 100
        self._common_timer = 0

        self._status = "idle"
        self._image_map = image_map

        self._plotpos_modifier = (np.random.random(size=2)-0.5)/3
        homeplace.add_actor(self)

        self._init_params()

    
    def set_workplace(self, map_, workplace):
        self._xwork, self._ywork = workplace.position
        self._workplace = workplace

        if self._workpath is not None:
            if self._workpath in self._stored_paths:
                index = self._stored_paths.index(self._workpath)
                self._stored_paths.remove(self._workpath)
                self._stored_paths_usage.pop(index)

        workpath = self.find_path(map_, workplace, self.homeplace)
        if workpath != False:
            self._workpath = workpath
            self._stored_paths.append(self._workpath)
            self._stored_paths_usage.append(2000) # workpath has large weight to remain
        else:
            print("Could not find workpath for", self, f"living at {self.position}")


    def set_preferred_commonarea(self, common_areas):
        """ Find the closest common_area (in direct line) to current position
        and set it as preffered
        """
        min_dist = np.linalg.norm(common_areas[0].position - self.position)
        current_choice = common_areas[0]

        for common_area in common_areas[1:]:
            dist = np.linalg.norm(common_area.position - self.position)
            if dist < min_dist:
                min_dist = dist
                current_choice = common_area

        self._preffered_commonarea = current_choice


    def set_motion(self, target):
        map_ = self._map
        self._status = "moving"
        self._plotpos_modifier = np.zeros(2)
        self._x, self._y = self.current_container.position
        current_path = None

        # look for path in stored paths
        for check_path in self._stored_paths:
            if target == check_path[-1]:
                if self._current_container in check_path:
                    current_path = check_path
                    path_ind = current_path.index(self._current_container)
            elif target == check_path[0]:
                 if self._current_container in check_path:
                    current_path = check_path[::-1]
                    path_ind = current_path.index(self._current_container)
        
        # if a valid path could not be find, generate a new one
        if current_path is None:
            current_path = self.find_path(map_, target, start = self._current_container)

            # store new path in stored paths, but first remove the least
            # used one if there are 20 or more paths stored
            if len(self._stored_paths) >= 20:
                ind_remove = np.argmin(self._stored_paths_usage)
                self._stored_paths.pop(ind_remove)
                self._stored_paths_usage.pop(ind_remove)
            if not current_path in self._stored_paths:
                self._stored_paths.append(current_path)
                self._stored_paths_usage.append(0)
            path_ind = 0

        if current_path in self._stored_paths:
            index = self._stored_paths.index(current_path)
            self._stored_paths_usage[index] += 1

        if self._current_container == target:
            # already at the target
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


    def frame_forward(self, global_time):
        """ general behavior for all actors on each frame """
        if isinstance(self._current_container, CommonArea):
            self._common_timer += 1
            if self._common_timer > self._common_timer_max:
                self._common_timer = 0
                self.set_motion(self.homeplace)
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
        self._params["behavior"] = "normal"


    def set_param(self, param, value):
        if param in self._params:
            self._params[param] = value
        else:
            print(f"Attempted to set param {param} for object {self} with value {value}, but key does not exist...")


    def increase_param(self, param, value = 1):
        """ Increase the value of a given parameter in the params dictionary
        
        Keyword arguments:
        value -- how much to increase the parameter (default 1)
        """

        if param in self._params:
            self._params[param] += value
        else:
            print(f"Attempted to increase param {param} for object {self} with value {value}, but key does not exist...")


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
    def preffered_commonarea(self):
        return self._preffered_commonarea

    @property
    def is_home(self):
        if self._current_container == self.homeplace:
            return True
        else:
            return False

    @property
    def current_path_goal(self):
        if self._current_path is not None:
            return self._current_path[-1]
        return None

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
        # infection status values:
        # 0 = susceptible
        # 1 = infected
        # 2 = recovered
        self._params["infection_status"] = 0
        self._params["infection_chance_modifier"] = 1 # multiplied with the infection chance of the current area the person is in
                                                      # applies for infecting _other_ people
        self._params["active_worker"] = False
        self._params["basic_speed"] = 0.45

        self._params["behavior"] = "normal"
        self._params["infected_others"] = 0
        self._params["became_immune"] = 0


    def frame_forward(self, global_time):
        if self._params["infection_status"] == 1:
            self._infection_duration += 1
            if self._infection_duration > self._infection_length:
                """ become immune """
                self._params["infection_status"] = 2
                self._params["color"] = "green"
                self._params["became_immune"] = global_time
                self._infection_duration = 0
        return Actor.frame_forward(self, global_time)


    def set_infected(self, infection_length = 100):
        self._params["infection_status"] = 1
        self._params["color"] = "red"
        self._infection_duration = 0
        self._infection_length = infection_length + np.random.randint(int(infection_length/20) + 1) - int(infection_length/40)

    
    def __str__(self):
        return self._name


