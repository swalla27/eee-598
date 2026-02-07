# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 5 February 2026

# Homework 2 on QPSK

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.fft import fft
import os, sys

N = 10_000 # The number of bits
q_gain = 0.9 # The gain of the quadrature, where one is ideal
phase_err_deg = -5 # The phase error in degrees
phi = np.radians(phase_err_deg) # Convert the phase error to radians
const_diag_scale = 3.5 # The scale of the constellation diagram

# Place random binary integers into a list of bits
bits = list()
for idx in range(N):
    bits.append(str(randint(0, 1)) + str(randint(0, 1)))

# Convert the bits into I (in phase) and Q (quadrature) arrays
I = np.zeros(N)
Q = np.zeros(N)
for idx in range(N):
    if bits[idx] == '00':
        I[idx] = 1
        Q[idx] = 1
    elif bits[idx] == '01':
        I[idx] = -1
        Q[idx] = 1
    elif bits[idx] == '11':
        I[idx] = -1
        Q[idx] = -1
    else:
        I[idx] = 1
        Q[idx] = -1

# This function will classify a complex number into one of the four groups 
def classify_data_point(complex_number: complex):
    num_angle = np.angle(complex_number)
    if 0 <= num_angle < np.pi/2: 
        return '00' # First Quadrant
    elif np.pi/2 <= num_angle <= np.pi: 
        return '01' # Second Quadrant
    elif -np.pi <= num_angle <= -np.pi/2: 
        return '11' # Third Quadrant
    else: 
        return '10' # Fourth Quadrant
    
# This block of code will find the bit error rate given the noisy signal and the original bit stream
def find_bit_error_rate(S_N: np.array, bits: list):
    bit_errors = 0
    for idx, entry in enumerate(S_N):
        decoded_signal = classify_data_point(entry)
        original_signal = bits[idx]

        if decoded_signal != original_signal:
            bit_errors += 1

    BER = bit_errors/N
    return BER

for SNR_dB in range(6, 24, 3):

    # Convert the SNR to a ratio and define the signal power
    SNR_ratio = 10**(SNR_dB/10)
    Psig = 1

    # Calculate the noise std dev from the SNR and signal power
    noise_std_dev = np.sqrt(Psig/SNR_ratio)
    awgn = np.random.normal(0, noise_std_dev, size=[2, N])

    # Do array math to find the signal complex number without and then with noise
    S = I + 1j*Q
    S_N = S + awgn[0] + 1j*awgn[1] # Noise Only
    S_imp = (S_N.real + 1j*q_gain*S_N.imag)*np.exp(1j*phi) # With both noise and impairment
    
    ##################################
    ##### Constellation diagrams #####
    ##################################

    # Make a constellation diagram for the noise only case
    BER = find_bit_error_rate(S_N, bits)
    plt.close()
    plt.scatter(S_N.real, S_N.imag, color='red', marker='x')
    plt.xlabel('Real Component')
    plt.xlim([-const_diag_scale, const_diag_scale])
    plt.ylabel('Complex Component')
    plt.ylim([-const_diag_scale, const_diag_scale])
    plt.title(f'Constellation Diagram, Noise Only, SNR = {SNR_dB} dB', fontweight='bold')
    plt.figtext(0.15, 0.15, f'BER = {BER:.3f}', fontweight='bold')
    plt.grid(True)
    plt.savefig(f'homework/hw2/const_noise_only_{SNR_dB}dB.png', dpi=300)

    # Make a constellation diagram for the noise and impairment case
    BER = find_bit_error_rate(S_imp, bits)
    plt.close()
    plt.scatter(S_imp.real, S_imp.imag, color='black', marker='x')
    plt.xlabel('Real Component')
    plt.xlim([-const_diag_scale, const_diag_scale])
    plt.ylabel('Complex Component')
    plt.ylim([-const_diag_scale, const_diag_scale])
    plt.title(f'Constellation Diagram, With Impairment, SNR = {SNR_dB} dB', fontweight='bold')
    plt.figtext(0.15, 0.15, f'BER = {BER:.3f}', fontweight='bold', color='red')
    plt.figtext(0.15, 0.18, f'Q Gain = {q_gain:.1f}', fontweight='bold', color='red')
    plt.figtext(0.15, 0.21, f'Phase Error = {phase_err_deg:.0f}', fontweight='bold', color='red')
    plt.grid(True)
    plt.savefig(f'homework/hw2/const_impaired_{SNR_dB}dB.png', dpi=300)

    #############################
    ##### Time domain graph #####
    #############################

    mod_freq = 10e3 # Modulating frequency is 10 kHz
    car_freq = 4e6 # Carrier frequency is 4 MHz
    number_symbols_to_show = 10 # The number of symbols to show in this time domain graph

    M = 500 # Number of sample points per symbol
    carrier_amplitude = 1 # The amplitude of the carrier waveform
    mod_period = 1/mod_freq # The modulating waveform period
    car_period = 1/car_freq # The carrier waveform period
    T = car_period/M # The amount of time between each sample

    time_points_partial = np.linspace(0, stop=(1/mod_freq), num=M) # A time point array used to create the waveform for each symbol
    time_waveform_list = list() # This list will contain every point for all 10,000 symbols

    for entry in S_N: # Loop over every symbol in the S_N array

        cos_term = entry.real * carrier_amplitude * np.cos(2 * np.pi * car_freq * time_points_partial)
        sin_term = -entry.imag * carrier_amplitude * np.sin(2 * np.pi * car_freq * time_points_partial)

        time_waveform_list.extend(cos_term + sin_term) # Extend the time waveform list with a new batch of points from this symbol

    time_points_complete = np.linspace(0, stop=(N/mod_freq), num=M*N) # This variable includes the time points for all symbols, not just one
    time_waveform_array = np.array(time_waveform_list)

    # Graph the time domain signal and save the output in my homework directory
    plt.close()
    plt.plot(time_points_complete, time_waveform_array)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.xlim([0, number_symbols_to_show/mod_freq]) # This determines how many symbols are visible shown in the graph
    plt.ylim([-3, 3])
    plt.title(f'Time Domain, {number_symbols_to_show} Symbols Shown, SNR = {SNR_dB} dB', fontweight = 'bold')
    plt.grid(True)
    plt.savefig(f'homework/hw2/time_domain_{SNR_dB}dB.png', dpi = 300)

    ##################################
    ##### Frequency domain graph #####
    ##################################

    # Find the fft of the above waveform and make an array for its x-axis
    freq_waveform = fft(time_waveform_array)
    freq_points = np.linspace(0.0, 10e6, N*M//2)

    # Convert the frequency points to MHz
    freq_points_MHz = freq_points / 1e6

    # Graph the fft and save the figure in my homework directory
    plt.close()
    plt.plot(freq_points_MHz, 2.0/N * np.abs(freq_waveform[:N*M//2]))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('FFT Magnitude')
    plt.title(f'Frequency Domain, SNR = {SNR_dB} dB', fontweight = 'bold')
    plt.grid(True)
    plt.savefig(f'homework/hw2/freq_domain_{SNR_dB}dB.png', dpi = 300)

##################################
##### Results Interpretation #####
##################################

"""This homework has involved many things, but this python program is the majority of the points. In this program, we have made constallation diagrams, time domain graphs,
and frequency domain graphs. Each of those is repeated for several SNR values, and the constellation graphs include graphs with noise only, but also some with phase shift.
The constellation diagrams clearly show that as the SNR increases, the points become closer together. This makes sense because a higher SNR means the noise is lower in power
with respect to the signal itself. The constellation graphs with phase shift in addition to noise have a slightly oblique angle. This does affect the BER in some simulations,
but not all of them. Of course, each simulation is different because I am running creating white gaussian noise without a consistent seed. Moving on to the time domain now.
In these time domain graphs, you can see 10 symbols shown for each SNR that was tested. I chose to show 10 symbols because 10,000 would probably be way too many. Regardless,
you can see how the amplitude and phase is very volatile and subject to sudden changes in the case of low SNR, and the symbols become more regular when the SNR increases to
21 dB. Finally, the frequency domain graphs show the FFT of the entire waveform, including all 10,000 points. Since I used a carrier frequency of 4 MHz, that is what all 
those graphs show. That is, each one has a peak at 4 MHz. This is to be expected, and I think the noise around that tone is also expected because we have added white gaussian
noise to the signal. The peak becomes slightly more pronounced as the SNR increases, but that difference is hardly noticeable. I would expect the noise floor to decrease in
magnitude as the SNR increases, because that is how I coded this problem. That is, I increased the SNR by decreasing the power of the noise, while leaving the power of the 
signal alone. Since I did not use pulse shaping, """