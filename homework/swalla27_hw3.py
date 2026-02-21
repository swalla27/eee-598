# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 20 February 2026

# Homework 3 on Intermodulation Analysis

import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.fft import fft, fftfreq

SHOW_GRAPHS = False
PROJ_FOLDER = '/home/steven-wallace/Documents/asu/eee-598/homework'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(PROJ_FOLDER, 'swalla27_hw3_graphs.pdf'))

# Define frequencies for the cosine input waveforms.
FREQ_1 = 12e9
OMEGA_1 = FREQ_1*2*np.pi
PERIOD_1 = 1 / FREQ_1

FREQ_2 = 12e9 + 100e6
OMEGA_2 = FREQ_2*2*np.pi
PERIOD_2 = 1 / FREQ_2

SAMPLE_FREQ = FREQ_1*20

# Define constants related to the cubic polynomial model.
ALPHA1 = 28.2
ALPHA2 = 10
ALPHA3 = -15

# ALPHA2 = 0.126
# ALPHA3 = -1.33

# Define variables for how to sample the waveforms in the time domain.
NUM_PERIODS = 5
SAMPLES_PER = int(SAMPLE_FREQ / FREQ_1)
SAMPLE_SPACING = 1 / SAMPLE_FREQ
N = NUM_PERIODS * SAMPLES_PER

# The input powers which will be tested in the sweep.
POWERS_TO_TEST = np.arange(-10, 41)

# The x-axis used by all time domain waveforms in the program.
input_times = np.arange(0, PERIOD_1*NUM_PERIODS, SAMPLE_SPACING)

#####################################
##### Unit Conversion Functions #####
#####################################

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

def harmonic_model(t: float, amplitude: float):
    """A polynomial model which includes harmonics, but not intermodulation."""
    
    x = amplitude*np.cos(OMEGA_1*t)
    return cubic_polynomial(x), x

def intermod_model(t: float, amplitude: float):
    """A polynomial model which includes both intermodulation and harmonics."""
    
    x1 = amplitude*np.cos(OMEGA_1*t)
    x2 = amplitude*np.cos(OMEGA_2*t)
    return cubic_polynomial(x1+x2), x1+x2

def create_tangent_line(x: np.array, y: np.array):
    """Create a tangent line from two numpy arrays x and y.\n
       Currently hard-coded to make a tangent out of the first two points in the arrays."""

    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[1] - m*x[1]
    return m*x + b


#############################
##### Routine Functions #####
#############################

def harmonics_from_fft(fft_dBm: np.array):
    """This function will return the fundemental, second harmonic, and third harmonic intensities from an FFT waveform.\n
       Currently hard-coded under the assumption those values will always exist at the same indices."""

    fundamental = fft_dBm[5]
    second_harm = fft_dBm[10]
    third_harm = fft_dBm[15]
    return fundamental, second_harm, third_harm

def find_interception(x: np.array, y1: np.array, y2: np.array):
    """This function will find the interception between two waveforms stored as numpy arrays.\n
       It will return the x and y-values of the interception point."""

    delta_array = np.abs(y2-y1)
    intercept_idx = np.argmin(delta_array)
    intercept_x = x[intercept_idx]
    intercept_y = y1[intercept_idx]
    return intercept_x, intercept_y, intercept_idx

######################################
##### Sweep Input Power Function #####
######################################

def sweep_input_power(which_model, model_name: str):


    fund_powers = np.zeros(len(POWERS_TO_TEST))
    harm2_powers = np.zeros(len(POWERS_TO_TEST))
    harm3_powers = np.zeros(len(POWERS_TO_TEST))

    for idx, Pin_dBm in enumerate(POWERS_TO_TEST):

        amplitude = dBm_to_amplitude(Pin_dBm)
        output_voltages, input_voltages = which_model(input_times, amplitude)

        fft_frequencies = fftfreq(N, SAMPLE_SPACING)[:N//2]

        out_fft_raw = fft(output_voltages)
        out_fft_volts = 2.0/N * np.abs(out_fft_raw[0:N//2])
        out_fft_dBm = volts_to_dBm(out_fft_volts)
        out_fft_avg = sum(out_fft_dBm) - len(out_fft_dBm)

        in_fft_raw = fft(input_voltages)
        in_fft_volts = 2.0/N * np.abs(in_fft_raw[0:N//2])
        in_fft_dBm = volts_to_dBm(in_fft_volts)
        
        fund_power, harm2_power, harm3_power = harmonics_from_fft(out_fft_dBm)
        fund_powers[idx] = fund_power
        harm2_powers[idx] = harm2_power
        harm3_powers[idx] = harm3_power

        if Pin_dBm == 5:
            fig = plt.figure()
            plt.plot(input_times*1e9, output_voltages, label='Output Voltage')
            plt.plot(input_times*1e9, input_voltages, label='Input Voltage')
            plt.xlabel('Time (ns)')
            plt.ylabel('Output Voltage (V)')
            plt.title(f'Time Domain; Pin = {Pin_dBm} dBm; {model_name} Case')
            plt.legend(loc='upper right')
            plt.grid(True)
            pdf.savefig(fig)
            if SHOW_GRAPHS:
                plt.show()
            plt.close()

            fig = plt.figure()
            plt.plot(fft_frequencies/1e9, out_fft_dBm, label='Output Signal')
            plt.plot(fft_frequencies/1e9, in_fft_dBm, label='Input Signal')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('FFT Magnitude (dBm)')
            plt.title(f'Frequency Domain; Pin = {Pin_dBm} dBm; {model_name} Case')
            plt.legend()
            plt.grid(True)
            pdf.savefig(fig)
            if SHOW_GRAPHS:
                plt.show()
            plt.close()

    tanline = create_tangent_line(POWERS_TO_TEST, fund_powers)

    iip2, oip2, _ = find_interception(POWERS_TO_TEST, tanline, harm2_powers)
    iip3, *_ = find_interception(POWERS_TO_TEST, tanline, harm3_powers)
    cp_in, _, comp_idx = find_interception(POWERS_TO_TEST, tanline-1, fund_powers)

    print(f'{model_name} Case:\n\tInput Comp = {cp_in:.0f} dBm\n\tIIP3 = {iip3:.0f} dBm\n\tIIP2 = {iip2:.0f} dBm')

    fig = plt.figure()
    plt.plot(POWERS_TO_TEST[:comp_idx+5], fund_powers[:comp_idx+5], label='Fundamental Tone', color='black')
    plt.plot(POWERS_TO_TEST, tanline, label='Tangent Line', linestyle='dashed', color='red')
    plt.plot(POWERS_TO_TEST, harm2_powers, label='Second Harmonic', color='tab:blue')
    plt.plot(POWERS_TO_TEST, harm3_powers, label='Third Harmonic', color='green')
    plt.xlabel('Input Power (dBm)')
    plt.ylabel('Output Power (dBm)')
    plt.title(f'Input Power Sweep; {model_name} Case')
    plt.ylim([-10, oip2+5])
    plt.xlim([min(POWERS_TO_TEST), iip2+5])
    plt.legend()
    plt.grid(True)
    if SHOW_GRAPHS:
        plt.show()
    pdf.savefig(fig)
    plt.close()


if __name__ == "__main__":

    sweep_input_power(harmonic_model, 'Harmonic')
    sweep_input_power(intermod_model, 'Intermod')

pdf.close()