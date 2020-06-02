# Infection spread simulator
#### Requried packages:
- pygame (only if running the interactive version)
- numpy
- matplotlib
- json
- PIL (pillow)
- names
- seaborn

## Various premade modes for the interactive version:
In the top of the interactive.py file there are two settings:
- sim\_name
- show\_death\_list

These can be edited to get different behavior.  
Possible choices for "sim\_name":
- sim_name = "interactive"

standard mode, has 500 people, people can die

- sim_name = "interactive\_efficient"

fewer people so the game runs faster, has 300 people, people can die

- sim_name = "interactive\_efficient\_nodeath"

has 300 people like the previous option, but here people can't die
(but they still can if you drag the death rate slider to the right)

The possible choices for "show\_death\_list" is `True` or `False`. If it is `True` there will be shown a list of the most recent deaths on the right side of the screen. (Each person in the simulation is given a randomly generated name.)

### Note
The R0 calculation can be inaccurate, especially if one or more of the following is true:
- The disease length is short
- The simulation is small
- The real R0 is very high