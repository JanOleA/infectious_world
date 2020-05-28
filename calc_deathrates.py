import numpy as np
import matplotlib.pyplot as plt

def inverse_deathrate(rate, modifier = 0.1, inf_rate = -1):
    """ Returns what health is required for a given chance of dying (per day) """
    health = (modifier/rate)**(1/3) + inf_rate
    return health

def death_rate(health, modifier = 0.1, inf_rate = -1):
    if health < inf_rate+1e-3:
        return np.inf
    else:
        return 1/(np.maximum(health, inf_rate+1e-3) - inf_rate)**3*modifier

if __name__ == "__main__":
    wanted_rate = float(input("Enter a death rate (for disease duration) > "))
    disease_duration = float(input("Enter disease duration (days) > "))
    rate_perday = wanted_rate/disease_duration

    health_for_rate = inverse_deathrate(rate_perday)
    print(f"Health requried for requested death rate is {health_for_rate:.5f}.")
    print(f"Thus, if the normal health is 5, the disease reduction must be {5 - health_for_rate:.5f} health.")

    """
    rates = np.linspace(1e-4, 1, 1000)
    healths = inverse_deathrate(rates)

    plt.plot(rates, healths)
    plt.xlabel("Death rate")
    plt.ylabel("Health required")
    plt.show()"""