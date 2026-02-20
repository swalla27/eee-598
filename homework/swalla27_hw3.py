# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 20 February 2026

# Homework 3 on Intermodulation Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Define constants related to the cosine waveform used as input.
FREQ = 12e9
OMEGA = FREQ*2*np.pi
PERIOD = 1 / FREQ
AMPLITUDE = 10

# Define constants related to the cubic polynomial model.
ALPHA1 = 28.2
ALPHA2 = 1.20
ALPHA3 = -0.179

def polynomial_model(t: float):

    dc_term = (ALPHA2 * AMPLITUDE**2) / 2
    lin_term = (ALPHA1*AMPLITUDE + 3*ALPHA3*AMPLITUDE**3/4)*cosine_function(t)
    quad_term = (ALPHA2 * AMPLITUDE**2 / 2) * cosine_function(2*t)
    cub_term = (ALPHA3 * AMPLITUDE**3 / 4) * cosine_function(3*t)

    return dc_term + lin_term + quad_term + cub_term

def cosine_function(t: float):
    return AMPLITUDE*np.cos(OMEGA*t)

time_array = np.arange(0, PERIOD*10, PERIOD/5000)
output_array = polynomial_model(time_array)

plt.plot(time_array, output_array)
plt.show()