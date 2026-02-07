# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 5 February 2026

# Homework 2 on QPSK

import numpy as np
import matplotlib.pyplot as plt
from random import randint
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
    
    # Make a graph for the noise only case
    plt.close()
    plt.scatter(S_N.real, S_N.imag, color='red', marker='x')
    plt.xlabel('Real Component')
    plt.xlim([-const_diag_scale, const_diag_scale])
    plt.ylabel('Complex Component')
    plt.ylim([-const_diag_scale, const_diag_scale])
    plt.title(f'Constellation Diagram, Noise Only, SNR = {SNR_dB} dB', fontweight='bold')
    plt.figtext(0.15, 0.15, f'BER = {find_bit_error_rate(S_N, bits):.3f}', fontweight='bold')
    plt.grid(True)
    plt.savefig(f'../eee-598/homework/hw2/const_diag_noise_only_{SNR_dB}dB.png', dpi=300)

    # Make a graph for the noise and impairment case
    plt.close()
    plt.scatter(S_imp.real, S_imp.imag, color='black', marker='x')
    plt.xlabel('Real Component')
    plt.xlim([-const_diag_scale, const_diag_scale])
    plt.ylabel('Complex Component')
    plt.ylim([-const_diag_scale, const_diag_scale])
    plt.title(f'Constellation Diagram, With Impairment, SNR = {SNR_dB} dB', fontweight='bold')
    plt.figtext(0.15, 0.15, f'BER = {find_bit_error_rate(S_imp, bits):.3f}', fontweight='bold', color='red')
    plt.figtext(0.15, 0.18, f'Q Gain = {q_gain:.1f}', fontweight='bold', color='red')
    plt.figtext(0.15, 0.21, f'Phase Error = {phase_err_deg:.0f}', fontweight='bold', color='red')
    plt.grid(True)
    plt.savefig(f'../eee-598/homework/hw2/const_diag_impaired_{SNR_dB}dB.png', dpi=300)