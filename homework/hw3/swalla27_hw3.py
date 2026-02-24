# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 20 February 2026

# Homework 3 on Intermodulation Analysis

import numpy as np
import time
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.fft import fft, fftfreq
from typing import Callable

############################################
##### LNA Should Have These Properties #####
############################################
"""
Gain = 14.5 dB
CP Out = 15 dBm; CP In = 0.5 dBm
OIP3 = 29 dBm; IIP3 = 14.5 dBm
OIP2 = 38 dBm; IIP2 = 23.5 dBm
"""

#####################
##### Constants #####
#####################

# This is the project folder and I use that path to create a pdf object for storing the graphs.
PROJ_FOLDER = '/home/steven-wallace/Documents/asu/eee-598/homework/hw3'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(PROJ_FOLDER, 'swalla27_hw3_graphs.pdf'))

# Define frequencies for the cosine input waveforms.
FREQ_1 = 12e9
OMEGA_1 = FREQ_1*2*np.pi
PERIOD_1 = 1 / FREQ_1

FREQ_2 = 12e9 * 1.1
OMEGA_2 = FREQ_2*2*np.pi
PERIOD_2 = 1 / FREQ_2

# These are the alpha values that actually give the gain, IIP3, and IIP2 of the LNA.
ALPHA = [5.97, 2.30, -8.45]

# These are the alpha values that I calculated by hand.
# ALPHA = [5.31, 0.237, -0.89]

# Define variables for how to sample the waveforms in the time domain.
SAMPLE_FREQ = FREQ_1*40
NUM_PERIODS = 200
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

def dBm_to_Vpk(power_dBm: float):
    """A function which converts a power in dBm to a voltage Vpk, assuming the impedance seen is 50 ohms."""

    power_watts = 1e-3 * (10**(power_dBm/10))
    Vpk = np.sqrt(power_watts*2*50)
    return Vpk

def Vpk_to_dBm(Vpk: float):
    """A function which converts a voltage to a power in dBm. This is used to convert the FFT into dBm."""

    Vrms = Vpk / np.sqrt(2)
    power_watts = Vrms**2 / 50
    power_dBm = 10 * np.log10(power_watts / 1e-3)
    return power_dBm

##############################
##### Modeling Functions #####
##############################

def cubic_polynomial(x: float, ALPHA: list):
    return ALPHA[0]*x + ALPHA[1]*x**2 + ALPHA[2]*x**3

def harmonic_model(t: float, Vpk: float, ALPHA: list):
    """A polynomial model which includes harmonics, but not intermodulation."""
    
    x = Vpk*np.cos(OMEGA_1*t)
    return cubic_polynomial(x, ALPHA), x

def intermod_model(t: float, Vpk: float, ALPHA: list):
    """A polynomial model which includes both intermodulation and harmonics."""
    
    x1 = Vpk*np.cos(OMEGA_1*t)
    x2 = Vpk*np.cos(OMEGA_2*t)
    return cubic_polynomial(x1+x2, ALPHA), x1+x2

def create_tanline(x: np.array, y: np.array):
    """Create a tangent line from two numpy arrays x and y.\n
       Currently hard-coded to make a tangent out of the first two points in the arrays."""

    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[1] - m*x[1]
    return m*x + b

#############################
##### Routine Functions #####
#############################

def find_local_max(x: np.array, y: np.array, search_here: float):
    """This function will search in the region surrounding 'seach_here' for a local maximum in the y-axis array.\n
       It will return the maximum y-value in the neighborhood surrounding that search point."""

    delta = np.abs(x-search_here)
    region = np.argmin(delta)
    window = y[region-10:region+10]
    local_max = np.max(window)
    return local_max

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

def sweep_input_power(selected_model: Callable, model_name: str, ALPHA: list, verbose=True, makegraphs=True, showgraphs=False):
    """The purpose of this function is to sweep the input power, collect several important metrics, and create a few graphs with each call.\n
       *****Inputs*****\n
       selected_model: I will pass either the harmonic model or the intermodulation model in this slot, which determines the condition for the sweep.\n
       model_name: The name of the model being used in this function call.\n
       verbose: When set to True, the function will print important information to the terminal.\n
       makegraphs: When set to True, the function will make graphs and store them in the pdf object we created above.\n
       showgraphs: When set to True, the function will show those graphs as the program executes. Do not set makegraphs False and showgraphs True.\n
       *****Outputs*****\n
       This function will print information related to the intercept points to the terminal, and it will also create several graphs each time it is called.\n
       Those graphs include time domain, frequency domain, and then a single graph depicting input power vs output power."""

    # Initialize a few numpy arrays with zeros. Their purpose is to store the powers for the fundamental tone and each of the harmonics.
    fund_powers = np.zeros(len(INPUT_POWERS))
    harm2_powers = np.zeros(len(INPUT_POWERS))
    harm3_powers = np.zeros(len(INPUT_POWERS))

    # Begin to loop over the input powers specified above.
    for idx, Pin_dBm in enumerate(INPUT_POWERS):

        # First, convert the input power to an Vpk. Then, extract the input and output voltages from the chosen model with that Vpk.
        # Finally, calculate the average value of the output voltage so that it can be subtracted from the FFT.
        Vpk = dBm_to_Vpk(Pin_dBm)
        output_voltages, input_voltages = selected_model(INPUT_TIMES, Vpk, ALPHA)

        # Find the frequencies on the x-axis for these FFT plots.
        freq_array = fftfreq(N, SAMPLE_SPACING)[:N//2]

        # Find the FFT of the output voltage and convert this to dBm.
        out_fft_raw = fft(output_voltages)
        out_fft_volts = 2.0/N * np.abs(out_fft_raw[:N//2])
        out_fft_dBm = Vpk_to_dBm(out_fft_volts)

        # Find the FFT of the input voltage and convert this to dBm.
        in_fft_raw = fft(input_voltages)
        in_fft_volts = 2.0/N * np.abs(in_fft_raw[:N//2])
        in_fft_dBm = Vpk_to_dBm(in_fft_volts)
        
        # Extract the power of the fundamental tone and the two harmonics based on the output FFT.
        fund_power = find_local_max(freq_array, out_fft_dBm, search_here=FREQ_1)
        harm2_power = find_local_max(freq_array, out_fft_dBm, search_here=FREQ_1*2)
        harm3_power = find_local_max(freq_array, out_fft_dBm, search_here=FREQ_1*3)
        fund_powers[idx] = fund_power
        harm2_powers[idx] = harm2_power
        harm3_powers[idx] = harm3_power

        # Create time and frequency domain graphs under the following condition.
        if (Pin_dBm in [-20, -5, 5]) and makegraphs:

            # This section will graph the time domain input and output voltages when the if condition above is met.
            SLICE_START = 0
            SLICE_END = N//10
            fig = plt.figure()
            plt.plot(INPUT_TIMES[SLICE_START:SLICE_END]*1e9, output_voltages[SLICE_START:SLICE_END], label='Output Voltage', color='black')
            plt.plot(INPUT_TIMES[SLICE_START:SLICE_END]*1e9, input_voltages[SLICE_START:SLICE_END], label='Input Voltage', color='tab:blue')
            plt.xlabel('Time (ns)')
            plt.ylabel('Output Voltage (V)')
            plt.title(f'Time Domain; Pin = {Pin_dBm:.1f} dBm; {model_name} Case')
            plt.legend(loc='upper right', edgecolor='black')
            plt.grid(True)
            pdf.savefig(fig)
            if showgraphs:
                plt.show()
            plt.close()

            # This section will graph an FFT of the input and output when the if condition above is met.
            SLICE_START = 10
            SLICE_END = N//10
            fig = plt.figure()
            plt.plot(freq_array[SLICE_START:SLICE_END]/1e9, out_fft_dBm[SLICE_START:SLICE_END], label='Output Signal', color='black')
            plt.plot(freq_array[SLICE_START:SLICE_END]/1e9, in_fft_dBm[SLICE_START:SLICE_END], label='Input Signal', color='red')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('FFT Magnitude (dBm)')
            plt.title(f'Frequency Domain; Pin = {Pin_dBm:.1f} dBm; {model_name} Case')
            plt.legend(loc='lower left', edgecolor='black')
            plt.grid(True)
            pdf.savefig(fig)
            if showgraphs:
                plt.show()
            plt.close()

    # Create a tangent line of the fundamental tone, and then collect several intercept points.
    tanline = create_tanline(INPUT_POWERS, fund_powers)
    iip2, oip2, _ = find_intercept(INPUT_POWERS, tanline, harm2_powers)
    iip3, oip3, _ = find_intercept(INPUT_POWERS, tanline, harm3_powers)
    cp_in, cp_out, comp_idx = find_intercept(INPUT_POWERS, tanline-1, fund_powers)

    if verbose:
        # Print a summary of the intercept points to the terminal.
        print(f'{model_name} Case:')
        print(f'\tGain = {oip3-iip3:.1f} dB')
        print(f'\tCP In = {cp_in:.1f} dBm; CP Out = {cp_out:.1f} dBm')
        print(f'\tIIP3 = {iip3:.1f} dBm; OIP3 = {oip3:.1f} dBm')
        print(f'\tIIP2 = {iip2:.1f} dBm; OIP2 = {oip2:.1f} dBm')

    if makegraphs:
        # Create the graph of input power vs output power. This will include the fundamental tone, its tangent, the 2nd harmonic, and the 3rd harmonic.
        SLICE_END = comp_idx + 6
        fig = plt.figure()
        plt.plot(INPUT_POWERS[:SLICE_END], fund_powers[:SLICE_END], label='Fundamental Tone', color='red')
        plt.plot(INPUT_POWERS, tanline, label='Tangent Line', linestyle='dashed', color='black')
        plt.plot(INPUT_POWERS, harm2_powers, label='Second Harmonic', color='tab:blue')
        plt.plot(INPUT_POWERS, harm3_powers, label='Third Harmonic', color='tab:green')
        plt.xlabel('Input Power (dBm)')
        plt.ylabel('Output Power (dBm)')
        plt.title(f'Input Power Sweep; {model_name} Case')
        plt.ylim([min(fund_powers)-5, oip2+5])
        plt.xlim([min(INPUT_POWERS), iip2+5])
        plt.legend(loc='upper left', edgecolor='black')
        plt.figtext(0.68, 0.15, f'Gain = {oip3-iip3:.2f} dB\nInput CP = {cp_in} dBm\nIIP3 = {iip3} dBm\nIIP2 = {iip2} dBm', bbox=dict(facecolor='white', alpha=0.7))
        plt.grid(True)
        if showgraphs:
            plt.show()
        pdf.savefig(fig)
        plt.close()

    # Output an array containing the gain, IIP3, and IIP2 of this run.
    return np.array([oip3-iip3, iip3, iip2]) 

###############################################
##### Search for the correct alpha values #####
###############################################

def search_input_space(target_vector: np.array, num_outputs=5):
    """The purpose of this function is to search for the alpha values which will produce the requested gain, IIP3, and IIP2.\n
       It does this by creating a nested loop over lots of different alpha values, and recording the gain, IIP3, and IIP2 of each one.\n
       Once it collects all of that information from about 27,000 iterations, it finds the best ones using linear algebra.\n
       All it does is find the distance between every vector and the target vector, then selects the best ones and prints that to a txt file.\n
       This function will not be used in the final iteration of the homework that I turn in.\n
       *****Inputs*****\n
       target_vector: This contains the gain, IIP3, and IIP2 we are searching for. It could be something like 14.5, 14.5, 23.5.\n
       num_outputs: The number of outputs to print to the txt file. That txt file contains the best alpha values."""

    # These are the alpha values that we will sweep. The following block of code will make a nested loop out of this.
    ALPHA1_VALUES = np.linspace(5.5, 6.5, num=30)
    ALPHA2_VALUES = np.linspace(2.0, 2.5, num=30)
    ALPHA3_VALUES = np.linspace(-8, -9, num=30)

    # input_array is now a nested loop of those alpha values above, where every combination is included. 
    # output_array will store the outputs for comparison with the target.
    aa1, aa2, aa3 = np.meshgrid(ALPHA1_VALUES, ALPHA2_VALUES, ALPHA3_VALUES, indexing='ij')
    input_array = np.stack([aa1.ravel(), aa2.ravel(), aa3.ravel()], axis=1)
    output_array = np.zeros([*input_array.shape])
    
    # Loop over every element in the input array, which is each combination of alpha that we requested.
    t0 = time.time()
    for idx, input_comb in enumerate(input_array):

        # The output is a numpy vector containing gain, IIP3, and IIP2 for that iteration. We store it and move on.
        out = sweep_input_power(intermod_model, 'Intermodulation', ALPHA=input_comb, verbose=False, makegraphs=False)
        output_array[idx] = out

        # I am printing the duration for each run to the terminal to ensure things aren't moving too slow.
        t1 = time.time()
        print(f'{t1-t0:.2f} seconds have elapsed after {idx} iterations.')
    
    # This section will find the distance from the output vectors to the target vector, and find the 5 vectors closest to the target.
    distances = np.linalg.norm(output_array-target_vector, axis=1, keepdims=True)
    indices = np.argsort(distances, axis=0)[:num_outputs].flatten()
    top_points = output_array[indices]
    top_distances = distances[indices]

    # This will print the results of our analysis to a txt file. This will contain the alpha values and their outputs.
    output_path = os.path.join(PROJ_FOLDER, 'best_alpha_values.txt')
    with open(output_path, 'w') as f:

        print('Searched the input space for alpha values which yield the following:', file=f)
        print(f'\tGain = {target_vector[0]:.1f} dB\n\tIIP3 = {target_vector[1]:.1f} dBm\n\tIIP2 = {target_vector[2]:.1f} dBm', file=f)

        print('\nBest alpha values:', file=f)
        print(input_array[indices], file=f)

        print('\nTheir distances to the target vector:', file=f)
        print(top_distances, file=f)

        print('\nThe gain, IIP3, and IIP2 of those alpha value combinations:', file=f)
        print(top_points, file=f)

#############################
##### Program Execution #####
#############################

if __name__ == "__main__":

    search_input_space(np.array([14.5, 14.5, 23.5]))

    # # Call the function to sweep the input power twice, once with the harmonic only model and a second time with intermodulation included.
    # sweep_input_power(harmonic_model, 'Harmonic', ALPHA)
    # sweep_input_power(intermod_model, 'Intermodulation', ALPHA)

    # Close the pdf object to free up memory.
    pdf.close()