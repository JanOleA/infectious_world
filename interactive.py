import pygame
from pygame.locals import *
from sim import InfectSim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os
import json
from calc_deathrates import inverse_deathrate, death_rate

matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg

matplotlib.rcParams['savefig.pad_inches'] = 0


class InteractiveSim(InfectSim):
    def __init__(self, mapfile, params, sim_name, show_death_list):
        super().__init__(mapfile, params, sim_name)
        self._running = True
        self._screen = None
        self.map_size = self.map_array.shape[:2]
        self.height = 800
        self.map_scale = self.height/self.map_size[0]

        self.width = int(self.map_size[1]*self.map_scale)
        self.size = (self.width, self.height) # size of the map
        self._i = 0
        self.show_death_list = show_death_list

        self._rightshift = 200
        if show_death_list:
            self._rightextend = 300
        else:
            self._rightextend = 0
        self._downshift = 0
        self._max_len_deathlist = int((self.height + self._downshift)/20) - 2
        self._deathlist = []

        self.states = None
        self.colors = None
        self.positions = None
        self.recovered_stats = None

        self.states_prev2days = np.zeros((self.day_length*2, self.state_history.shape[1]))

        initial_states = self.world.get_actor_states_num()
        for i in range(self.day_length*2):
            self.states_prev2days[i] = initial_states

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.DARKREDPURPLE = (144, 0, 34)
        self.LIGHTGREY = (200, 200, 200)
        self.UI_BG = (100, 100, 100)


    def animated_SIR_plot(self):
        self.history_ax.collections.clear()
        states_prev2days = self.states_prev2days

        day_array = np.arange(self.day_length*2)

        infected = states_prev2days[:,1]
        dead_inf = infected + states_prev2days[:,3]
        recovered = dead_inf + states_prev2days[:,2]
        susceptible = recovered + states_prev2days[:,0]
        dead_natural = susceptible + states_prev2days[:,4]
        self.history_ax.fill_between(day_array, infected, label = "infected", color = "red", alpha = 0.3)
        self.history_ax.fill_between(day_array, infected, dead_inf, label = "dead (from infection)", color = "black", alpha = 0.3)
        self.history_ax.fill_between(day_array, dead_inf, recovered, label = "recovered", color = "green", alpha = 0.3)
        self.history_ax.fill_between(day_array, recovered, susceptible, label = "susceptible", color = "blue", alpha = 0.3)
        if np.sum(states_prev2days[:,4]) >= 1:
            self.history_ax.fill_between(day_array, susceptible, dead_natural, label = "dead (natural)", color = "purple", alpha = 0.3)

        self.canvas = agg.FigureCanvasAgg(self.history_fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data_historyplot = self.renderer.tostring_rgb()


    def set_health_impact_from_rate(self, rate):
        health_impact = 5 - inverse_deathrate(rate/self.infection_length)
        self.world.set_infection_health_impact(health_impact)


    def reset(self):
        self._deathlist = []
        self.world.reset()


    def on_init(self):
        pygame.init()
        pygame.display.set_caption("Infection spread simulator")
        self._screen = pygame.display.set_mode((self.size[0] + self._rightshift + self._rightextend, self.size[1] + self._downshift), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True

        self._clock = pygame.time.Clock()

        self._background = pygame.image.load(self.mapfile).convert()
        self._background = pygame.transform.scale(self._background, self.size) # map background
        self._sprites = []

        person_images = ["person.png",
                         "person_sick.png",
                         "person_recovered.png",
                         "person_dead.png"]

        for i in range(len(person_images)):
            self._sprites.append(pygame.image.load(f"{os.getcwd()}/graphics/{person_images[i]}").convert())
            self._sprites[i].set_colorkey((255,0,255))
            self._sprites[i] = pygame.transform.scale(self._sprites[i], (20, 20))

        
        self.current_infected = self.initial_infected
        self.lockdown_initiated = False
        self.ui_rect = pygame.Rect(0,0,self._rightshift + self.width + self._rightextend, 800)

        pygame.font.init()
        self.font = pygame.font.SysFont('Calibri', 17)
        self.font_small = pygame.font.SysFont('Calibri', 14)
        self.font_big = pygame.font.SysFont('Calibri', 20)

        self.text_day = self.font_big.render(f"Day: {0}", True, self.WHITE)

        self.text_susceptible = self.font.render(f"Susceptible: {self.num_inhabitants - self.initial_infected}", True, self.WHITE)
        self.text_numinfected = self.font.render(f"Infected: {self.current_infected}", True, self.WHITE)
        self.text_recovered = self.font.render(f"Recovered: {0}", True, self.WHITE)
        self.text_dead = self.font.render(f"Dead: {0}", True, self.WHITE)

        self.text_dead_title = self.font.render(f"Most recent deaths:", True, self.WHITE)
        
        self.sliders = []

        self.text_infection_chance = self.font.render(f"Infection chance: {0}", True, self.WHITE)
        self.sliders.append(Slider(5, 170, 160, 20,
                                   lval = self.infection_chance/100, cval = self.infection_chance, rval = self.infection_chance*4, func="square",
                                   mod_func=self.world.set_infection_chance))

        self.text_death_chance = self.font.render(f"Exp. death rate: {0}", True, self.WHITE)
        self.sliders.append(Slider(5, 220, 160, 20,
                                   lval = 0, cval = self.expected_death_rate, rval = 3, func="linear",
                                   mod_func=self.set_health_impact_from_rate))

        self.text_stay_home_chance = self.font.render(f"Stay home chance: {0}", True, self.WHITE)
        self.sliders.append(Slider(5, 270, 160, 20,
                                   lval = 1e-15, cval = self.infected_stay_home_chance, rval = 1, func="linear",
                                   mod_func=self.world.set_infected_stay_home_chance))

        self.buttons = []
        self.buttons.append(Button(5, self.height - 40, 160, 30, "Reset", self.WHITE, self.RED, self.DARKREDPURPLE, self.reset))
        self.buttons.append(Button(5, self.height - 260, 160, 30, "Initiate lockdown", self.BLACK, self.WHITE, self.LIGHTGREY, self.initiate_lockdown))
        self.buttons.append(Button(5, self.height - 300, 160, 30, "Deactivate lockdown", self.BLACK, self.WHITE, self.LIGHTGREY, self.deactivate_lockdown))

        self.history_fig = plt.figure(figsize = (1.6, 1.6), dpi = 100)
        self.history_ax = plt.axes([0,0,1,1], frameon=False)
        self.history_ax.get_xaxis().set_visible(False)
        self.history_ax.get_yaxis().set_visible(False)
        plt.autoscale(tight = True)

        self.animated_SIR_plot()
        plot_size = self.canvas.get_width_height()
        self.plot_surf = pygame.image.fromstring(self.raw_data_historyplot, plot_size, "RGB")


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                for button in self.buttons:
                    button.pressed = False


    def loop(self):
        store_index = (self.world.global_time + 1)%self.max_frames
        self.states, self.colors = self.world.frame_forward()
        self.positions = self.world.get_actor_plotpositions()
        self.recovered_stats = self.world.get_recovered_stats()

        self.state_history[store_index] = self.states
        self.color_history[store_index] = self.colors
        self.position_history[store_index] = self.positions

        self.states_prev2days = np.roll(self.states_prev2days, -1, axis = 0)
        self.states_prev2days[-1] = self.states

        self.keys = pygame.key.get_pressed()

        if pygame.mouse.get_pressed()[0] == True:
            pos = pygame.mouse.get_pos()
            for slider in self.sliders:
                if slider.is_inside(pos):
                    mod_func = slider.set_button_x(pos[0], self.keys)
                    mod_func(slider.get_value())

            for button in self.buttons:
                if button.is_inside(pos):
                    if not button.pressed:
                        act_func = button.act_func
                        button.pressed = True
                        if act_func is not None:
                            act_func()

        current_infected = self.states[1]
        if (current_infected / self.num_inhabitants > self.lockdown_ratio
                and not lockdown_initiated):
            self.world.set_behaviors("stay_home", self.lockdown_chance)
            lockdown_initiated = True

        day = self.world.global_time/self.day_length
        self.text_day = self.font_big.render(f"Day: {day:2.2f}", True, self.WHITE)

        self.text_susceptible = self.font.render(f"Susceptible: {int(self.states[0])}", True, self.WHITE)
        self.text_numinfected = self.font.render(f"Infected: {int(self.states[1])}", True, self.WHITE)
        self.text_recovered = self.font.render(f"Recovered: {int(self.states[2])}", True, self.WHITE)
        self.text_dead = self.font.render(f"Dead: {int(self.states[3] + self.states[4])}", True, self.WHITE)

        self.text_infection_chance = self.font.render(f"Infection chance: {self.world.infection_chance:3.3g}", True, self.WHITE)
        expected_death_rate = death_rate(5 - self.world.disease_health_impact)*self.infection_length
        if expected_death_rate < 1e-3:
            rate_string = f"{expected_death_rate:1.2e}"
        else:
            rate_string = f"{expected_death_rate:1.3f}"
        self.text_death_chance = self.font.render(f"Exp. death rate: {rate_string}", True, self.WHITE)

        self.text_stay_home_chance = self.font.render(f"Stay home chance: {self.world.infected_stay_home_chance:3.3g}", True, self.WHITE)

        self._clock.tick_busy_loop(30)
        self.fps = self._clock.get_fps()

        self.text_fps = self.font.render(f"fps = {self.fps:3.1f}", True, self.WHITE)

        self._i += 1
                    

    def render(self):
        pygame.draw.rect(self._screen, self.UI_BG, self.ui_rect)
        self._screen.blit(self._background, (self._rightshift, self._downshift))

        self._screen.blit(self.text_day, (5, 10))
        self._screen.blit(self.text_susceptible, (5, 40))
        self._screen.blit(self.text_numinfected, (5, 60))
        self._screen.blit(self.text_recovered, (5, 80))
        self._screen.blit(self.text_dead, (5, 100))
        self._screen.blit(self.text_infection_chance, (5, 150))
        self._screen.blit(self.text_death_chance, (5, 200))
        self._screen.blit(self.text_stay_home_chance, (5, 250))
        self._screen.blit(self.text_fps, (self.size[0] + self._rightshift - self.text_fps.get_rect().width - 10, 10))

        for slider in self.sliders:
            bg, button = slider()
            pygame.draw.rect(self._screen, self.BLACK, bg)
            pygame.draw.rect(self._screen, self.WHITE, button)

        for button in self.buttons:
            bg, text = button()
            trect = text.get_rect()
            text_x = bg.centerx - trect.width/2
            text_y = bg.centery - trect.height/2

            pygame.draw.rect(self._screen, button.color, bg)
            self._screen.blit(text, (text_x, text_y))

        for pos, color, actor in zip(self.positions, self.colors, self.world.get_actors()):
            x = pos[0]*self.map_scale - 10 + self._rightshift + self.map_scale/2
            y = self.height - (pos[1] + 1)*self.map_scale - 10 + self._downshift + self.map_scale/2
            if color == "c":
                img = self._sprites[0]
            elif color == "r":
                img = self._sprites[1]
            elif color == "g":
                img = self._sprites[2]
            elif color == "k":
                img = self._sprites[3]

            if actor.params["infection_status"] == 3 or actor.params["infection_status"] == 4:
                if not actor in self._deathlist:
                    self._deathlist.append(actor)

            self._screen.blit(img, (x,y))

        
        if self.show_death_list:
            x = self._rightshift + self.width + 5
            self._screen.blit(self.text_dead_title, (x, 10))
            y = 35
            for person in self._deathlist[-self._max_len_deathlist:]:
                if person.params["infection_status"] == 3:
                    infection = "yes"
                else:
                    infection = "no"
                text_dead = self.font_small.render(f"{str(person)}, age: {person.params['age']:2.1f}, infected: {infection}", True, self.WHITE)
                self._screen.blit(text_dead, (x, y))
                y += 20
                

        if self._i%10 == 0:
            self.animated_SIR_plot()
            plot_size = self.canvas.get_width_height()
            self.plot_surf = pygame.image.fromstring(self.raw_data_historyplot, plot_size, "RGB")
        self._screen.blit(self.plot_surf, (5,590))

        pygame.display.flip()


    def cleanup(self):
        pygame.quit()

        self.map = self.world.get_map()

        if not os.path.exists(f"{os.getcwd()}/output"):
            os.mkdir(f"{os.getcwd()}/output")
        if not os.path.exists(f"{os.getcwd()}/output/{self.sim_name}"):
            os.mkdir(f"{os.getcwd()}/output/{self.sim_name}")

        self.output_dir = f"{os.getcwd()}/output/{self.sim_name}"

        output_data = [self.map,
                       self.im,
                       self.position_history,
                       self.state_history,
                       self.color_history,
                       self.day_length,
                       self.R_history]

        print("Saving data...")
        np.save(f"{self.output_dir}/data.npy", output_data)
 

    def execute(self):
        if self.on_init() == False:
            self._running = False
 
        while(self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.loop()
            self.render()
        self.cleanup()


class Slider:
    def __init__(self,
                 left,
                 top,
                 width,
                 height,
                 lval = 0,
                 cval = 50,
                 rval = 100,
                 func = "linear",
                 bwidth = 10,
                 mod_func = None):

        self.width = width
        self.height = height
        self.center = (left + width/2, top + height/2)
        self.left = left
        self.top = top
        self.bgrect = pygame.Rect(left, top, width, height)
        self.button = pygame.Rect(left, top, bwidth, height*0.8)
        self.bwidth = bwidth

        self.lval = lval
        self.cval = cval
        self.rval = rval

        self.leftmost_x = left + bwidth*1.1
        self.center_x = self.center[0]
        self.rightmost_x = left + width - bwidth*1.1

        self.mod_func = mod_func

        self.set_button_pos(self.center)

        if func == "square":
            self.c = lval
            self.b = 4*cval - 3*lval - rval
            self.a = rval - self.b - lval
            self.retfunc = self.squarefunc
        else:
            self.retfunc = self.linearfunc


    def linearfunc(self, x):
        val = x - self.leftmost_x
        val /= (self.rightmost_x - self.leftmost_x)
        if val < 0.5:
            return (val*2)*self.cval + self.lval
        else:
            return (val - 0.5)*(self.rval - self.cval)*2 + self.cval

    
    def squarefunc(self, x):
        val = x - self.leftmost_x
        val /= (self.rightmost_x - self.leftmost_x)
        return self.a*val**2 + self.b*val + self.c

    
    def set_button_pos(self, x, y = None):
        if y is None:
            x, y = x
        self.button.center = (x, y)


    def set_button_x(self, x, keys):
        if not keys[306] and not keys[305]:
            if abs(x - self.center[0]) < self.bwidth/3:
                x = self.center[0]
        self.button.centerx = max(min(x, self.rightmost_x), self.leftmost_x)

        return self.mod_func


    def is_inside(self, x, y = None):
        if y is None:
            x, y = x
        if self.left < x < self.left + self.width:
            if self.top < y < self.top + self.height:
                return True
        return False

    
    def get_value(self):
        return self.retfunc(self.button.centerx)


    def __call__(self):
        return self.bgrect, self.button


class Button:
    def __init__(self, left, top, width, height, text, text_color, bg_color, bg_pressed_color, act_func):
        self.width = width
        self.height = height
        self.center = (left + width/2, top + height/2)
        self.left = left
        self.top = top
        self.bgrect = pygame.Rect(left, top, width, height)
        self.act_func = act_func
        self.pressed = False
        font = pygame.font.SysFont('Calibri', 17)
        self.text = font.render(text, True, text_color)
        self.bg_color = bg_color
        self.bg_pressed_color = bg_pressed_color


    def is_inside(self, x, y = None):
        if y is None:
            x, y = x
        if self.left < x < self.left + self.width:
            if self.top < y < self.top + self.height:
                return True
        return False
    

    @property
    def color(self):
        if self.pressed:
            return self.bg_pressed_color
        else:
            return self.bg_color

    def __call__(self):
        return self.bgrect, self.text


if __name__ == "__main__":
    sim_name = "interactive_efficient_nodeath"
    show_death_list = False

    with open(f"{os.getcwd()}/sim_params/{sim_name}.json", "r") as infile:
        params = json.load(infile)

    mapfile = params["mapfile"]
    
    interactive_sim = InteractiveSim(mapfile, params, sim_name, show_death_list)
    interactive_sim.execute()