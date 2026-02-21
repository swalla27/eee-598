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
from typing import Callable

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "mathtext.fontset": "cm"
})

############################################
##### LNA Should Have These Properties #####
############################################
"""
Gain = 14.5 dB
OIP3 = 29 dBm; IIP3 = 14.5 dBm
OIP2 = 38 dBm; IIP2 = 23.5 dBm
"""

#####################
##### Constants #####
#####################

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

SAMPLE_FREQ = FREQ_1*40

# Define constants related to the cubic polynomial model.
# ALPHA1 = 2
ALPHA2 = 10
ALPHA3 = -15

ALPHA1 = 28.18
# ALPHA2 = 0.126
# ALPHA3 = -1.33

# Define variables for how to sample the waveforms in the time domain.
NUM_PERIODS = 50
SAMPLES_PER = int(SAMPLE_FREQ / FREQ_1)
SAMPLE_SPACING = 1 / SAMPLE_FREQ
N = NUM_PERIODS * SAMPLES_PER

# The input powers which will be tested in the sweep.
INPUT_POWERS = np.arange(-20, 30+1)

# The x-axis used by all time domain waveforms in the program.
INPUT_TIMES = np.arange(0, PERIOD_1*NUM_PERIODS, SAMPLE_SPACING)

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

def ratio_to_dB20(x: float):
    return 20 * np.log10(x)

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

def harmonics_from_fft(freq_array: np.array, fft_dBm: np.array):
    """This function will return the fundemental, second harmonic, and third harmonic intensities from an FFT waveform.\n
       Currently hard-coded under the assumption those values will always exist at the same indices."""
    
    # Find the index of the fundamental frequency.
    delta_array = np.abs(freq_array-FREQ_1)
    region = np.argmin(delta_array)
    window = fft_dBm[region-10:region+10]
    fund_idx = np.argmax(window) + (region-10)
    fund_value = fft_dBm[fund_idx]
    print(f'fund value = {fund_value}')

    delta_array = np.abs(freq_array-2*FREQ_1)
    region = np.argmin(delta_array)
    window = fft_dBm[region-10:region+10]
    harm2_idx = np.argmax(window) + (region-10)
    harm2_value = fft_dBm[harm2_idx]
    print(f'harm2 value = {harm2_value}')

    delta_array = np.abs(freq_array-3*FREQ_1)
    region = np.argmin(delta_array)
    window = fft_dBm[region-10:region+10]
    harm3_idx = np.argmax(window) + (region-10)
    harm3_value = fft_dBm[harm3_idx]
    print(f'harm3 value = {harm3_value}')

    return fund_value, harm2_value, harm3_value

def find_intercept(x: np.array, y1: np.array, y2: np.array):
    """This function will find the interception between two waveforms stored as numpy arrays.\n
       It will return the x and y-values of the interception point."""

    delta_array = np.abs(y2-y1)
    intercept_idx = np.argmin(delta_array)
    xintercept = x[intercept_idx]
    yintercept = y2[intercept_idx]
    return xintercept, yintercept, intercept_idx

######################################
##### Sweep Input Power Function #####
######################################

def sweep_input_power(selected_model: Callable, model_name: str):
    """The purpose of this function is to sweep the input power, collect several important metrics, and create a few graphs with each call.\n
       *****Inputs*****\n
       selected_model: I will pass either the harmonic model or the intermodulation model in this slot, which determines the condition for the sweep.\n
       model_name: The name of the model being used in this function call.\n
       *****Outputs*****\n
       This function will print information related to the intercept points to the terminal, and it will also create several graphs each time it is called.\n
       Those graphs include time domain, frequency domain, and then a single graph depicting input power vs output power."""

    # Initialize a few numpy arrays with zeros. Their purpose is to store the powers for the fundamental tone and each of the harmonics.
    fund_powers = np.zeros(len(INPUT_POWERS))
    harm2_powers = np.zeros(len(INPUT_POWERS))
    harm3_powers = np.zeros(len(INPUT_POWERS))

    # Begin to loop over the input powers specified above.
    for idx, Pin_dBm in enumerate(INPUT_POWERS):
        print(Pin_dBm)

        # First, convert the input power to an amplitude. Then, extract the input and output voltages from the chosen model with that amplitude.
        # Finally, calculate the average value of the output voltage so that it can be subtracted from the FFT.
        amplitude = dBm_to_amplitude(Pin_dBm)
        output_voltages, input_voltages = selected_model(INPUT_TIMES, amplitude)
        gain_est = ratio_to_dB20(max(output_voltages) / amplitude)
        print(f'\tVoltage Gain Estimate = {gain_est:.1f} dB')

        # Find the frequencies on the x-axis for these FFT plots.
        freq_array = fftfreq(N, SAMPLE_SPACING)[:N//2]

        # Find the FFT of the output voltage and convert this to dBm.
        out_fft_raw = fft(output_voltages)
        out_fft_volts = 2.0/N * np.abs(out_fft_raw[:N//2])
        out_fft_dBm = volts_to_dBm(out_fft_volts)

        # Find the FFT of the input voltage and convert this to dBm.
        in_fft_raw = fft(input_voltages)
        in_fft_volts = 2.0/N * np.abs(in_fft_raw[:N//2])
        in_fft_dBm = volts_to_dBm(in_fft_volts)
        
        # Extract the power of the fundamental tone and the two harmonics based on the output FFT.
        fund_power, harm2_power, harm3_power = harmonics_from_fft(freq_array, out_fft_dBm)
        fund_powers[idx] = fund_power
        harm2_powers[idx] = harm2_power
        harm3_powers[idx] = harm3_power

        # Create time and frequency domain graphs under the following condition.
        if (Pin_dBm == -10) or (Pin_dBm == 10):

            # This section will graph the time domain input and output voltages when the if condition above is met.
            fig = plt.figure()
            plt.plot(INPUT_TIMES[:N//10]*1e9, output_voltages[:N//10], label='Output Voltage', color='black')
            plt.plot(INPUT_TIMES[:N//10]*1e9, input_voltages[:N//10], label='Input Voltage', color='red')
            plt.xlabel('Time (ns)')
            plt.ylabel('Output Voltage (V)')
            plt.title(f'Time Domain; Pin = {Pin_dBm} dBm; {model_name} Case')
            plt.legend(loc='upper right', edgecolor='black')
            plt.grid(True)
            pdf.savefig(fig)
            if SHOW_GRAPHS:
                plt.show()
            plt.close()

            # This section will graph an FFT of the input and output when the if condition above is met.
            DISCARD_FIRST = 10 # Discard this many points from the beginning of the FFT graph.
            fig = plt.figure()
            plt.plot(freq_array[DISCARD_FIRST:N//10]/1e9, out_fft_dBm[DISCARD_FIRST:N//10], label='Output Signal', color='black')
            plt.plot(freq_array[DISCARD_FIRST:N//10]/1e9, in_fft_dBm[DISCARD_FIRST:N//10], label='Input Signal', color='red')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('FFT Magnitude (dBm)')
            plt.title(f'Frequency Domain; Pin = {Pin_dBm} dBm; {model_name} Case')
            plt.legend(loc='lower left', edgecolor='black')
            plt.grid(True)
            pdf.savefig(fig)
            if SHOW_GRAPHS:
                plt.show()
            plt.close()

    # Create a tangent line of the fundamental tone, and then collect several intercept points.
    tanline = create_tangent_line(INPUT_POWERS, fund_powers)
    iip2, oip2, _ = find_intercept(INPUT_POWERS, tanline, harm2_powers)
    iip3, oip3, _ = find_intercept(INPUT_POWERS, tanline, harm3_powers)
    cp_in, cp_out, comp_idx = find_intercept(INPUT_POWERS, tanline-1, fund_powers)

    # Print a summary of the intercept points to the terminal.
    print(f'{model_name} Case:')
    print(f'\tVoltage Gain Estimate = {gain_est:.1f} dB')
    print(f'\tCP In = {cp_in:.1f} dBm; CP Out = {cp_out:.1f} dBm')
    print(f'\tIIP3 = {iip3:.1f} dBm; OIP3 = {oip3:.1f} dBm')
    print(f'\tIIP2 = {iip2:.1f} dBm; OIP2 = {oip2:.1f} dBm')

    # Create the graph of input power vs output power. This will include the fundamental tone, its tangent, the 2nd harmonic, and the 3rd harmonic.
    fig = plt.figure()
    plt.plot(INPUT_POWERS[:comp_idx+6], fund_powers[:comp_idx+6], label='Fundamental Tone', color='black')
    plt.plot(INPUT_POWERS, tanline, label='Tangent Line', linestyle='dashed', color='red')
    plt.plot(INPUT_POWERS, harm2_powers, label='Second Harmonic', color='tab:blue')
    plt.plot(INPUT_POWERS, harm3_powers, label='Third Harmonic', color='green')
    plt.xlabel('Input Power (dBm)')
    plt.ylabel('Output Power (dBm)')
    plt.title(f'Input Power Sweep; {model_name} Case')
    plt.ylim([min(fund_powers)-5, oip2+5])
    plt.xlim([min(INPUT_POWERS), iip2+5])
    plt.legend(loc='upper left', edgecolor='black')
    plt.figtext(0.72, 0.15, f'Input CP = {cp_in} dBm\nIIP3 = {iip3} dBm\nIIP2 = {iip2} dBm', bbox=dict(facecolor='white', alpha=0.7))
    plt.grid(True)
    if SHOW_GRAPHS:
        plt.show()
    pdf.savefig(fig)
    plt.close()

#############################
##### Program Execution #####
#############################

if __name__ == "__main__":

    # Call the function to sweep the input power twice, once with the harmonic only model and a second time with intermodulation included.
    sweep_input_power(harmonic_model, 'Harmonic')
    sweep_input_power(intermod_model, 'Intermodulation')

    # Close the pdf object to free up memory.
    pdf.close()