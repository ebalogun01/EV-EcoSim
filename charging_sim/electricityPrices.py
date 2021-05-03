"""Contains pricing structure and electricity data. Should contain attribute that spits out vector of prices depending
on whatever pricing scheme is being used..."""
import numpy as np


# load_profile is a 24x1 array with kWh consumed in each our of the day, starting at 0:00
# Rates in $/kWh based on "Residential TOU Service for Plug-In EV2"
# Peak (weekday) = 4 to 9 PM 
# Partial-peak (weekday) = 3 to 4 PM, 9 to 12 AM
# Off-peak: all other times

def electricityPricesSummer():
    TOU_rate = np.array(
        [0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611,
         0.16611, 0.16611, 0.16611, 0.16611, 0.36812, 0.47861, 0.47861, 0.47861, 0.47861, 0.47861,
         0.36812, 0.36812, 0.36812])
    return TOU_rate


def electricityPricesWinter():
    TOU_rate = np.array(
        [0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611,
         0.16611, 0.16611, 0.16611, 0.16611, 0.33480, 0.35150, 0.35150, 0.35150, 0.35150, 0.35150,
         0.33480, 0.33480, 0.33480])
    return TOU_rate
