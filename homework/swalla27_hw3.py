# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 20 February 2026

# Homework 3 on Intermodulation Analysis

import numpy as np
import pandas as pd
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.fft import fft, fftfreq


PROJ_FOLDER = '/home/steven-wallace/Documents/asu/eee-598/homework'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(PROJ_FOLDER, 'swalla27_hw3_graphs.pdf'))

# Define constants related to the cosine waveform used as input.
WAVE_FREQ = 12e9
OMEGA = WAVE_FREQ*2*np.pi
PERIOD = 1 / WAVE_FREQ
SAMPLE_FREQ = WAVE_FREQ*20


# Define constants related to the cubic polynomial model.
ALPHA1 = 28.2
ALPHA2 = 0.126
ALPHA3 = -0.05*10

#####################################
##### Unit Conversion Functions #####
#####################################

def watts_to_dBm(P: float):
    return 10 * np.log(P/1e-3)

def dBm_to_amplitude(Pin_dBm: float):
    Pin_watts = 1e-3 * (10**(Pin_dBm/10))
    amplitude = np.sqrt(Pin_watts*2*50)
    return amplitude

def volts_to_dBm(V: float):
    return 10 * np.log(V**2 / 50)

##############################
##### Modeling Functions #####
##############################

def polynomial_model(t: float):

    lin_term = ALPHA1*cosine_function(t)
    quad_term = ALPHA2 * cosine_function(t)**2
    cub_term = ALPHA3 * cosine_function(t)**3

    return lin_term + quad_term + cub_term

def cosine_function(t: float):
    return amplitude*np.cos(OMEGA*t)

def tangent_to_fundamental():
    pass


#############################
##### Routine Functions #####
#############################

def graphing_routine(x: np.array, y: np.array, xfield: str, yfield: str, tfield: str, show = False):
    fig = plt.figure()

    plt.plot(x, y)
    plt.xlabel(xfield)
    plt.ylabel(yfield)
    plt.title(tfield)
    plt.grid(True)

    pdf.savefig(fig)

    if show:
        plt.show()

    plt.close()

def harmonics_from_fft(fft_dBm: np.array):
    fundamental = fft_dBm[5]
    second_harm = fft_dBm[10]
    third_harm = fft_dBm[15]
    return fundamental, second_harm, third_harm

###################################################
##### Time Domain Plot for Single Input Power #####
###################################################

Pin_dBm = 30
amplitude = dBm_to_amplitude(Pin_dBm)

num_periods = 5
samples_per_period = int(SAMPLE_FREQ / WAVE_FREQ)
sample_spacing = 1 / SAMPLE_FREQ
N = num_periods * samples_per_period

input_times = np.arange(0, PERIOD*num_periods, sample_spacing)
output_voltages = polynomial_model(input_times)

graphing_routine(x=input_times, y=output_voltages, xfield='Time (s)', yfield='Output Voltage (V)', 
                 tfield=f'Time Domain When Pin = {Pin_dBm} dBm', show=False)

##############################################
##### Make an FFT for Single Input Power #####
##############################################

fft_frequencies = fftfreq(N, sample_spacing)[:N//2]
fft_raw = fft(output_voltages)
fft_volts = 2.0/N * np.abs(fft_raw[0:N//2])
fft_dBm = volts_to_dBm(fft_volts)

graphing_routine(x=fft_frequencies/1e9, y=fft_dBm, xfield='Freq (GHz)', yfield='FFT Magnitude', 
                 tfield=f'Fourier Transform When Pin = {Pin_dBm}', show=False)


##################################################
##### Sweep Input Power, Plot Harmonic Power #####
##################################################

fund_powers = dict()
harm2_powers = dict()
harm3_powers = dict()

for Pin_dBm in range(-10, 41):

    amplitude = dBm_to_amplitude(Pin_dBm)
    output_voltages = polynomial_model(input_times)

    fft_frequencies = fftfreq(N, sample_spacing)[:N//2]
    fft_raw = fft(output_voltages)
    fft_volts = 2.0/N * np.abs(fft_raw[0:N//2])
    fft_dBm = volts_to_dBm(fft_volts)

    # graphing_routine(x=fft_frequencies, y=fft_dBm, xfield='Freq (Hz)', yfield='FFT Magnitude', 
    #                 tfield=f'Fourier Transform When Pin = {Pin_dBm}', show=False)
    
    fund_power, harm2_power, harm3_power = harmonics_from_fft(fft_dBm)
    fund_powers.update({Pin_dBm: fund_power})
    harm2_powers.update({Pin_dBm: harm2_power})
    harm3_powers.update({Pin_dBm: harm3_power})

x = -200
y = dict()
for key, value in fund_powers.items():
    if value < x:
        break
    x = value

    y.update({key: value})
    
plt.plot(y.keys(), y.values(), label='Fundamental Tone')
plt.plot(harm2_powers.keys(), harm2_powers.values(), label='Second Harmonic')
plt.plot(harm3_powers.keys(), harm3_powers.values(), label='Third Harmonic')
# plt.ylim([-50, 80])
plt.xlabel('Input Power (dBm)')
plt.ylabel('Output Power (dBm)')
plt.title('Graphing IIP3 and IIP2')
plt.legend()
plt.grid(True)
plt.show()


pdf.close()