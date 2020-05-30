import numpy as np
import matplotlib.pyplot as plt

expected_lifespan = 21

def age_health(age, five_divisor = 10, zero_point = 21):
    health = ((age/zero_point)**3 - 1)*5*five_divisor/(1-five_divisor)
    return np.maximum(np.minimum(20, health), -0.5)/5

def death_rate(health, modifier = 0.1, inf_rate = -1):
    if isinstance(health, np.ndarray):
        rate = 1/(np.maximum(health, inf_rate+1e-3) - inf_rate)**3*modifier
        rate[health < inf_rate+1e-3] = np.inf
        return rate
        
    if health < inf_rate+1e-3:
        return np.inf
    else:
        return 1/(np.maximum(health, inf_rate+1e-3) - inf_rate)**3*modifier

fig, axs = plt.subplots(1, 2, figsize = (14,8))

for div in range(-5, 12):
    if div == 1:
        continue
    ages = np.linspace(0, expected_lifespan * 1.5, 1000)
    age_healths = age_health(ages, five_divisor = div, zero_point = expected_lifespan)
    death_rates = death_rate(age_healths*5)

    axs[0].plot(ages, death_rates, label = f"death rate | div = {div}")
    axs[0].set_xlabel("age")

    axs[1].plot(ages, age_healths, label = f"health multiplier | div = {div}")
    axs[1].set_xlabel("age")

axs[0].legend()
axs[1].legend()


class Person:
    def __init__(self):
        self.age = 0
        self.dead = False

    def age_one(self):
        roll = np.random.random()
        if roll < death_rate(age_health(self.age, five_divisor = 20, zero_point = expected_lifespan)*5)/4:
            self.dead = True
            return self.age
        else:
            self.age += 0.25
            return None

N = 1000
people = []

for i in range(N):
    people.append(Person())

death_ages = []

for i in range(int(expected_lifespan*8)):
    for person in people:
        if not person.dead:
            result = person.age_one()
            if result is not None:
                death_ages.append(result)

print("Mean death age:", np.mean(death_ages))

plt.figure()
plt.hist(death_ages, bins = int(expected_lifespan*2))
plt.xlabel("age")
plt.ylabel("number of dead")
plt.show()