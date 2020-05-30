import pygame
from pygame.locals import *
from sim import InfectSim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json

class InteractiveSim(InfectSim):
    def __init__(self, mapfile, params, sim_name):
        super().__init__(mapfile, params, sim_name)
        self._running = True
        self._display_surf = None
        self.map_size = self.map_array.shape[:2]
        self.height = 800
        self.map_scale = self.height/self.map_size[0]

        self.width = int(self.map_size[1]*self.map_scale)
        self.size = (self.width, self.height) # size of the map
        self._i = 0

        self._rightshift = 200
        self._downshift = 0

        self.states = None
        self.colors = None
        self.positions = None
        self.recovered_stats = None

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.UI_BG = (100, 100, 100)

        pygame.font.init()
        self.font = pygame.font.SysFont('Calibri', 15)

        self.text_susceptible = self.font.render(f"Susceptible: {0}", True, self.WHITE)
        self.text_numinfected = self.font.render(f"Infected: {0}", True, self.WHITE)
        self.text_recovered = self.font.render(f"Recovered: {0}", True, self.WHITE)
        self.text_dead = self.font.render(f"Dead: {0}", True, self.WHITE)


    def on_init(self):
        pygame.init()
        pygame.display.set_caption("Infection spread simulator")
        self._screen = pygame.display.set_mode((self.size[0] + self._rightshift, self.size[1] + self._downshift), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        self._background = pygame.image.load(self.mapfile).convert()
        self._background = pygame.transform.scale(self._background, self.size) # map background
        self._sprites = []

        person_images = ["person.png",
                         "person_sick.png",
                         "person_recovered.png",
                         "person_dead.png"]

        for i in range(len(person_images)):
            self._sprites.append(pygame.image.load(f"graphics/{person_images[i]}").convert())
            self._sprites[i].set_colorkey((255,0,255))
            self._sprites[i] = pygame.transform.scale(self._sprites[i], (9, 15))

        self.current_infected = self.initial_infected
        self.lockdown_initiated = False
        self.ui_rect = pygame.Rect(0,0,self._rightshift,800)
 

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False


    def on_loop(self):
        i = self._i
        self.states, self.colors = self.world.frame_forward()
        self.positions = self.world.get_actor_plotpositions()
        self.recovered_stats = self.world.get_recovered_stats()

        current_infected = self.states[1]
        if (current_infected / self.num_inhabitants > self.lockdown_ratio
                and not lockdown_initiated):
            self.world.set_behaviors("stay_home", self.lockdown_chance)
            lockdown_initiated = True

        self.text_susceptible = self.font.render(f"Susceptible: {self.states[0]}", True, self.WHITE)
        self.text_numinfected = self.font.render(f"Infected: {self.states[1]}", True, self.WHITE)
        self.text_recovered = self.font.render(f"Recovered: {self.states[2]}", True, self.WHITE)
        self.text_dead = self.font.render(f"Dead: {self.states[3] + self.states[4]}", True, self.WHITE)
        self._i += 1
                    

    def on_render(self):
        pygame.draw.rect(self._screen, self.UI_BG, self.ui_rect)
        self._screen.blit(self._background, (self._rightshift, self._downshift))

        self._screen.blit(self.text_susceptible, (5, 10))
        self._screen.blit(self.text_numinfected, (5, 30))
        self._screen.blit(self.text_recovered, (5, 50))
        self._screen.blit(self.text_dead, (5, 70))

        for pos, color in zip(self.positions, self.colors):
            x = pos[0]*self.map_scale - 4 + self._rightshift + self.map_scale/2
            y = self.height - (pos[1] + 1)*self.map_scale - 8 + self._downshift + self.map_scale/2
            if color == "c":
                img = self._sprites[0]
            elif color == "r":
                img = self._sprites[1]
            elif color == "g":
                img = self._sprites[2]
            elif color == "k":
                img = self._sprites[3]

            self._screen.blit(img, (x,y))
        pygame.display.flip()


    def on_cleanup(self):
        pygame.quit()
 

    def execute(self):
        if self.on_init() == False:
            self._running = False
 
        while(self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()
 

if __name__ == "__main__":
    sim_name = "test"
    mapfile = "map.png"

    with open(f"sim_params/{sim_name}.json", "r") as infile:
        params = json.load(infile)
    
    interactive_sim = InteractiveSim(mapfile, params, sim_name)
    interactive_sim.execute()