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

sys.path.append('/home/steven-wallace/Documents/asu/eee-598')
from unit_conv_func import *

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "mathtext.fontset": "cm"
})

#################################
##### Functions and Classes #####
#################################

def display_voltage(voltage: float):
    if voltage < 1e-3:
        return f'{voltage*1e6:.1f} uV'
    elif voltage < 1:
        return f'{voltage*1e3:.1f} mV'
    else:
        return f'{voltage:.1f} V' 

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

TABLE_DIR = '/home/steven-wallace/Documents/asu/eee-598/labs/lab3'

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

#############################
##### Total System Gain #####
#############################

# Find the total gain of the system.
gain_dB_sys = 0
for component in components:
    gain_dB_sys += component.gain_dB

################################
##### Output SNR and Power #####
################################

# Find the output SNR.
snr_in = PIN - pnoise_in # Input SNR in dB.
snr_out = snr_in - nfig_sys # Output SNR in dB.

# Find the output power and peak to peak voltage at the output.
pout = PIN + gain_dB_sys # The output power in dBm.
Vpp_out = 2*dBm_to_Vpk(pout)

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

# Find the input power at which IM3 products become problematic (this is step 4 in the example).
# There will be severe IM3 desense when two tones having this power are applied to the input.
p_im3 = (POUT_WANTED - gain_dB_sys + 2*iip3_sys_dBm) / 3

######################
##### ADC Sizing #####
######################

# Calculations regarding how to size the ADC.
noise_adc_in = pnoise_in + nfig_sys + gain_dB_sys
Pfs = Vpk_to_dBm(1 / 2) # The argument to this function is the voltage amplitude at the ADC input.
Pq = noise_adc_in - 10
snr_adc = Pfs - Pq
enob = (snr_adc-1.76) / 6.02 # Estimated number of bits.

##########################
##### BER Estimation #####
##########################

# I already calculated the output SNR, so I do not need to do that again.
ber = 0.5 * np.exp(-snr_out / 2)

##############################################
##### Calculated System Properties Table #####
##############################################

# Make a table from the calculated system properties.
data_dict = {
    'Metric': [
        'System Noise Figure', 'Total System Gain', 'Output SNR', 
        'Vpp at Output', 'System IIP3', 'Severe IM3 Desense', 
        'Required ADC Bits', 'Estimated BER'
    ],
    'Value': [
        f'{nfig_sys:.2f} dB', f'{gain_dB_sys:.2f} dB', f'{snr_out:.2f} dB',
        display_voltage(Vpp_out), f'{iip3_sys_dBm:.2f} dBm', 
        f'{p_im3:.2f} dBm', f'{enob:.2f} bits', f'{ber:.2e} (unitless)'
    ]
}

df = pd.DataFrame(data=data_dict)
fig, ax = plt.subplots(figsize=(4,2))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
plt.savefig(os.path.join(TABLE_DIR, 'calculated_sys_properties.png'), dpi=300)

###################################
##### Power and Voltage Table #####
###################################

# Initalize the variables used in this table.
points = ['A', 'B', 'C', 'D', 'E', 'F']
signal_powers = list()
noise_powers = list()
signal_amplitudes = list()
snr_values = list()

# Fill in the data for this table.
current_sig_pwr = PIN
current_noise_pwr = pnoise_in
current_Vpk = dBm_to_Vpk(PIN)
for component in components:

    signal_powers.append(f'{current_sig_pwr:.1f}')
    noise_powers.append(f'{current_noise_pwr:.0f}')
    signal_amplitudes.append(display_voltage(current_Vpk))
    snr_values.append(f'{current_sig_pwr - current_noise_pwr:.1f}')

    current_sig_pwr += component.gain_dB
    current_noise_pwr += component.gain_dB + component.nfig
    current_Vpk = dBm_to_Vpk(current_sig_pwr)

# Consolidate the data into a single dictionary.
data_dict = {
    'Point': points,
    'Signal Power (dBm)': signal_powers,
    'Noise Power (dBm)': noise_powers,
    'SNR (dB)': snr_values,
    'Signal Amplitude': signal_amplitudes
}

# Create and save the table.
df = pd.DataFrame(data=data_dict)
fig, ax = plt.subplots(figsize=(8,2))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.savefig(os.path.join(TABLE_DIR, 'sig_pwr_voltages.png'), dpi=300)


##############################
##### System Input Table #####
##############################

# Build the dictionary to fill in this table.
names = list()
gain_vals = list()
nfig_vals = list()
iip3_vals = list()
cpout_vals = list()
for component in components:
    names.append(component.name)
    gain_vals.append(component.gain_dB)
    nfig_vals.append(component.nfig)
    iip3_vals.append(component.iip3_dBm)
    cpout_vals.append(component.cpout)

data_dict = {
    'Block Name': names,
    'Gain (dB)': gain_vals,
    'Noise Figure (dB)': nfig_vals,
    'IIP3 (dBm)': iip3_vals,
    'Output Comp (dBm)': cpout_vals
}

# Create the table from that dictinoary.
df = pd.DataFrame(data=data_dict)
fig, ax = plt.subplots(figsize=(7,2))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.savefig(os.path.join(TABLE_DIR, 'sys_inputs.png'), dpi=300)