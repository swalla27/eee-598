# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 5 March 2026

# Lab 3 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time
import sys
import os

#################################
##### Functions and Classes #####
#################################

def dB_to_rat(dB: float):
    """A function which converts a dB value to a ratio."""

    return 10**(dB/10)

def rat_to_dB(rat: float):
    """A function which converts a ratio to dB."""

    return 10*np.log10(rat)

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

def display_voltage(voltage: float):
    if voltage < 1e-3:
        return f'{voltage*1e6:.2f} uV'
    elif voltage < 1:
        return f'{voltage*1e3:.2f} mV'
    else:
        return f'{voltage:.2f} V' 

class Component():
    def __init__(self, name: str, gain_dB: float, nfig: float, iip3_dBm: float, cpout: float):
        self.name = name # A string describing what this component is.
        self.gain_dB = gain_dB # Gain in dB.
        self.nfig = nfig # Noise Figure, which has units of dB.
        self.iip3_dBm = iip3_dBm # Component IIP3 in dBm.
        self.cpout = cpout # Output 1dB compression point.

        self.cpin = cpout - gain_dB # Input 1dB compression point.
        self.nfac = dB_to_rat(self.nfig) # Noise Factor, which is unitless.
        self.gain_rat = dB_to_rat(self.gain_dB) # Gain as a ratio.
        self.iip3_W = 1e-3 * dB_to_rat(self.iip3_dBm) # IIP3 in units of W.
        self.dyn_range = self.cpin - PIN # Dynamic range of this component.

#####################
##### Constants #####
#####################
PIN = -70 # Input power in dBm.
POUT_WANTED = -33.8 # Desired output power in dBm.
BANDWIDTH = 1e6 # The bandwidth in Hz.
pnoise_in = -174 + 10*np.log10(BANDWIDTH) # The noise power at the input in dBm.

#############################
##### Define Components #####
#############################

# Define all of the components in the signal chain.
components = [
    Component(name='Switch', gain_dB=-0.8, nfig=0.8, iip3_dBm=30, cpout=20),
    Component(name='Front BPF', gain_dB=-1.0, nfig=1.0, iip3_dBm=30, cpout=20),
    Component(name='LNA', gain_dB=10, nfig=2, iip3_dBm=-5, cpout=5),
    Component(name='Mixer', gain_dB=-6, nfig=6, iip3_dBm=5, cpout=10),
    Component(name='BB Amp', gain_dB=50, nfig=6, iip3_dBm=10, cpout=15),
    Component(name='ADC Filter', gain_dB=-1, nfig=1, iip3_dBm=30, cpout=20)
]

###############################
##### System Noise Factor #####
###############################

# Find the noise factor of the system.
nfac_sys = components[0].nfac
denominator = 1
for idx, component in enumerate(components[1:], start=1):
    numerator = component.nfac - 1
    denominator *= components[idx-1].gain_rat
    nfac_sys += numerator/denominator

nfig_sys = rat_to_dB(nfac_sys) # Noise Figure of the whole system.
print('\nCALCULATED SYSTEM PROPERTIES')
print(f'System Noise Figure: {nfig_sys:.2f} dB')

#############################
##### Total System Gain #####
#############################

# Find the total gain of the system.
gain_dB_sys = 0
for component in components:
    gain_dB_sys += component.gain_dB
print(f'Total System Gain: {gain_dB_sys:.2f} dB')

################################
##### Output SNR and Power #####
################################

# Find the output SNR.
snr_in = PIN - pnoise_in # Input SNR in dB.
snr_out = snr_in - nfig_sys # Output SNR in dB.
print(f'Output SNR: {snr_out:.2f} dB')

# Find the output power and peak to peak voltage at the output.
pout = PIN + gain_dB_sys # The output power in dBm.
Vpp_out = 2*dBm_to_Vpk(pout)
print(f'Vpp at Output: {Vpp_out*1e3:.2f} mV')

########################
##### IM3 Products #####
########################

# This section will find the IIP3 of the entire system.
running_sum = 1 / components[0].iip3_W
gain_term = 1
for idx, component in enumerate(components[1:], start=1):
    gain_term *= components[idx-1].gain_rat
    new_term = component.iip3_W * gain_term
    running_sum += 1/new_term
iip3_sys_W = 1/running_sum
iip3_sys_dBm = rat_to_dB(iip3_sys_W/1e-3) # The system IIP3 in dBm.
print(f'System IIP3: {iip3_sys_dBm:.2f} dBm')

# Find the input power at which IM3 products become problematic (this is step 4 in the example).
# There will be severe IM3 desense when two tones having this power are applied to the input.
p_im3 = (POUT_WANTED - gain_dB_sys + 2*iip3_sys_dBm) / 3
print(f'Severe IM3 Desense: {p_im3:.2f} dBm')

######################
##### ADC Sizing #####
######################

# Calculations regarding how to size the ADC.
noise_adc_in = pnoise_in + nfig_sys + gain_dB_sys
Pfs = Vpk_to_dBm(1 / 2) # The argument to this function is the voltage amplitude at the ADC input.
Pq = noise_adc_in - 10
snr_adc = Pfs - Pq
enob = (snr_adc-1.76) / 6.02 # Estimated number of bits.
print(f'Required ADC bits: {enob:.1f} bits')

##########################
##### BER Estimation #####
##########################

# I already calculated the output SNR, so I do not need to do that again.
ber = 0.5 * np.exp(-snr_out / 2)
print(f'Estimated BER: {ber:.2e} (unitless)\n')

############################################
##### Power and Voltage for Each Block #####
############################################

current_sig_pwr = PIN
current_noise_pwr = pnoise_in
current_Vpk = dBm_to_Vpk(PIN)
print('POWER AND VOLTAGE FOR EACH COMPONENT')

for component in components:

    print(f'Component Name: {component.name}')
    print(f'\tInput Signal Power: {current_sig_pwr:.2f} dBm')
    print(f'\tInput Noise Power: {current_noise_pwr:.2f} dBm')
    print(f'\tInput Signal Vpk: {display_voltage(current_Vpk)}')

    current_sig_pwr += component.gain_dB
    current_noise_pwr += component.gain_dB + component.nfig
    current_Vpk = dBm_to_Vpk(current_sig_pwr)

    print(f'\tOutput Signal Power: {current_sig_pwr:.2f} dBm')
    print(f'\tOutput Noise Power: {current_noise_pwr:.2f} dBm')
    print(f'\tOutput Signal Vpk: {display_voltage(current_Vpk)}')