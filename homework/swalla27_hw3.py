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
from scipy.signal import argrelextrema

PROJ_FOLDER = '/home/steven-wallace/Documents/asu/eee-598/homework'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(PROJ_FOLDER, 'swalla27_hw3_graphs.pdf'))

# Define several frequencies for the cosine input waveforms.
FREQ_1 = 12e9
OMEGA_1 = FREQ_1*2*np.pi
PERIOD_1 = 1 / FREQ_1

FREQ_2 = 12e9 + 100e6
OMEGA_2 = FREQ_2*2*np.pi
PERIOD_2 = 1 / FREQ_2

SAMPLE_FREQ = FREQ_1*20

# Define constants related to the cubic polynomial model.
ALPHA1 = 28.2
# ALPHA2 = 10
# ALPHA3 = -35

ALPHA2 = 0.126
ALPHA3 = 0.0473

#####################################
##### Unit Conversion Functions #####
#####################################

def watts_to_dBm(P: float):
    """A function which converts a power from W to dBm."""

    return 10 * np.log10(P/1e-3)

def dBm_to_amplitude(Pin_dBm: float):
    """A function which converts a power in dBm to a voltage amplitude, assuming the impedance seen is 50 ohms."""

    Pin_watts = 1e-3 * (10**(Pin_dBm/10))
    amplitude = np.sqrt(Pin_watts*2*50)
    return amplitude

def volts_to_dBm(V: float):
    """A function which converts a voltage to a power in dBm. This is used to convert the FFT into dBm."""

    return 10 * np.log10(V**2 / 50)

##############################
##### Modeling Functions #####
##############################

def cubic_polynomial(x: float):
    return ALPHA1*x + ALPHA2*x**2 + ALPHA3*x**3

def harmonic_model(t: float):
    """A polynomial model which includes harmonics, but not intermodulation."""
    
    x = amplitude*np.cos(OMEGA_1*t)
    return cubic_polynomial(x)

def intermod_model(t: float):
    """A polynomial model which includes both intermodulation and harmonics."""
    
    x1 = amplitude*np.cos(OMEGA_1*t)
    x2 = amplitude*np.cos(OMEGA_2*t)
    return cubic_polynomial(x1+x2)

def create_tangent_line(x: np.array, y: np.array):
    """Create a tangent line from two numpy arrays x and y.\n
       Currently hard-coded to make a tangent out of the first two points in the arrays."""

    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[1] - m*x[1]
    return m*x + b


#############################
##### Routine Functions #####
#############################

def graphing_routine(x: np.array, y: np.array, xfield: str, yfield: str, tfield: str, show=False):
    """A consistent graphing routine used throughout the program which collects all graphs into a single pdf.\n
       It includes options for the axis labels, title, and whether to show the graph while the program runs."""

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
    """This function will return the fundemental, second harmonic, and third harmonic intensities from an FFT waveform.\n
       Currently hard-coded under the assumption those values will always exist at the same indices."""

    fundamental = fft_dBm[5]
    second_harm = fft_dBm[10]
    third_harm = fft_dBm[15]
    return fundamental, second_harm, third_harm

def find_interception(x: np.array, y1: np.array, y2: np.array, tol=1):
    """This function will find the interception between two waveforms stored as numpy arrays.\n
       It will return the x and y-values of the interception point."""

    lowest_distance = 1e20
    intercept_x = 0
    intercept_y = 0

    for a, b, c in zip(x, y1, y2):
        delta = np.abs(b-c)
        if delta < lowest_distance:
            lowest_distance = delta
        else:
            intercept_x = a
            intercept_y = b
            break
    
    if lowest_distance > tol:
        return None
    else:
        return intercept_x, intercept_y
    


###################################################
##### Time Domain Plot for Single Input Power #####
###################################################

Pin_dBm = 30
amplitude = dBm_to_amplitude(Pin_dBm)

num_periods = 5
samples_per_period = int(SAMPLE_FREQ / FREQ_1)
sample_spacing = 1 / SAMPLE_FREQ
N = num_periods * samples_per_period

input_times = np.arange(0, PERIOD_1*num_periods, sample_spacing)
# output_voltages = harmonic_model(input_times)

# graphing_routine(x=input_times, y=output_voltages, xfield='Time (s)', yfield='Output Voltage (V)', 
#                  tfield=f'Time Domain; Pin = {Pin_dBm} dBm; Harmonics Only', show=False)

##############################################
##### Make an FFT for Single Input Power #####
##############################################

# fft_frequencies = fftfreq(N, sample_spacing)[:N//2]
# fft_raw = fft(output_voltages)
# fft_volts = 2.0/N * np.abs(fft_raw[0:N//2])
# fft_dBm = volts_to_dBm(fft_volts)

# graphing_routine(x=fft_frequencies/1e9, y=fft_dBm, xfield='Freq (GHz)', yfield='FFT Magnitude', 
#                  tfield=f'Fourier Transform; Pin = {Pin_dBm} dBm; Harmonics Only', show=False)


##################################################
##### Sweep Input Power, Plot Harmonic Power #####
##################################################

fund_powers = list()
harm2_powers = list()
harm3_powers = list()
POWERS_TO_TEST = np.arange(-10, 41)

for Pin_dBm in POWERS_TO_TEST:

    amplitude = dBm_to_amplitude(Pin_dBm)
    output_voltages = harmonic_model(input_times)

    fft_frequencies = fftfreq(N, sample_spacing)[:N//2]
    fft_raw = fft(output_voltages)
    fft_volts = 2.0/N * np.abs(fft_raw[0:N//2])
    fft_dBm = volts_to_dBm(fft_volts)
    
    fund_power, harm2_power, harm3_power = harmonics_from_fft(fft_dBm)
    fund_powers.append(fund_power)
    harm2_powers.append(harm2_power)
    harm3_powers.append(harm3_power)

    if Pin_dBm == 5:
        graphing_routine(x=input_times, y=output_voltages, xfield='Time (s)', yfield='Output Voltage (V)', 
                        tfield=f'Time Domain; Pin = {Pin_dBm} dBm; Harmonics Only', show=True)
        graphing_routine(x=fft_frequencies/1e9, y=fft_dBm, xfield='Freq (GHz)', yfield='FFT Magnitude', 
                        tfield=f'Frequency Domain; Pin = {Pin_dBm} dBm; Harmonics Only', show=True)

fund_tangent = create_tangent_line(POWERS_TO_TEST, fund_powers)

fund_local_min = argrelextrema(np.array(fund_powers), np.less)[0][0]

fund_powers = fund_powers[:fund_local_min]

fig = plt.figure()

plt.plot(POWERS_TO_TEST[:fund_local_min], fund_powers, label='Fundamental Tone', color='black')
plt.plot(POWERS_TO_TEST, fund_tangent, label='Tangent Line', linestyle='dashed', color='red')
plt.plot(POWERS_TO_TEST, harm2_powers, label='Second Harmonic', color='tab:blue')
plt.plot(POWERS_TO_TEST, harm3_powers, label='Third Harmonic', color='green')
plt.xlabel('Input Power (dBm)')
plt.ylabel('Output Power (dBm)')
plt.title('IIP3 and IIP2, Harmonics Only')
plt.legend()
plt.grid(True)

pdf.savefig(fig)
plt.close()

#############################################
##### Two Tone Intermodulation Analysis #####
#############################################

fund_powers = list()
harm2_powers = list()
harm3_powers = list()
POWERS_TO_TEST = np.arange(-10, 41)

for Pin_dBm in POWERS_TO_TEST:

    amplitude = dBm_to_amplitude(Pin_dBm)
    output_voltages = intermod_model(input_times)

    fft_frequencies = fftfreq(N, sample_spacing)[:N//2]
    fft_raw = fft(output_voltages)
    fft_volts = 2.0/N * np.abs(fft_raw[0:N//2])
    fft_dBm = volts_to_dBm(fft_volts)
    
    fund_power, harm2_power, harm3_power = harmonics_from_fft(fft_dBm)
    fund_powers.append(fund_power)
    harm2_powers.append(harm2_power)
    harm3_powers.append(harm3_power)

    if Pin_dBm == 5:
        graphing_routine(x=input_times, y=output_voltages, xfield='Time (s)', yfield='Output Voltage (V)', 
                        tfield=f'Time Domain; Pin = {Pin_dBm} dBm; Including Intermodulation', show=True)
        graphing_routine(x=fft_frequencies/1e9, y=fft_dBm, xfield='Freq (GHz)', yfield='FFT Magnitude', 
                        tfield=f'Frequency Domain; Pin = {Pin_dBm} dBm; Including Intermodulation', show=True)



fund_tangent = create_tangent_line(POWERS_TO_TEST, fund_powers)

fund_local_min = argrelextrema(np.array(fund_powers), np.less)[0][0]

fund_powers = fund_powers[:fund_local_min]

fig = plt.figure()

plt.plot(POWERS_TO_TEST[:fund_local_min], fund_powers, label='Fundamental Tone', color='black')
plt.plot(POWERS_TO_TEST, fund_tangent, label='Tangent Line', linestyle='dashed', color='red')
plt.plot(POWERS_TO_TEST, harm2_powers, label='Second Harmonic', color='tab:blue')
plt.plot(POWERS_TO_TEST, harm3_powers, label='Third Harmonic', color='green')
plt.xlabel('Input Power (dBm)')
plt.ylabel('Output Power (dBm)')
plt.title('Graphing IIP3 and IIP2, Including Intermodulation')
plt.legend()
plt.grid(True)
plt.show()

pdf.savefig(fig)
plt.close()









pdf.close()