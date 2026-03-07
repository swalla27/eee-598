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
from unit_conversion import dB_to_rat, rat_to_dB, Vpk_to_dBm, dBm_to_Vpk

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
    def __init__(self, name: str, gain_dB: float, nfig: float, iip3_dBm: float, iip2_dBm: float, cpout_dBm: float):
        self.name = name # A string describing what this component is.
        self.gain_dB = gain_dB # Gain in dB.
        self.nfig = nfig # Noise Figure, which has units of dB.
        self.iip3_dBm = iip3_dBm # Component IIP3 in dBm.
        self.iip2_dBm = iip2_dBm # Component IIP2 in dBm.
        self.cpout_dBm = cpout_dBm # Output 1dB compression point.

        self.cpin_dBm = (cpout_dBm+1) - gain_dB # Input 1dB compression point.
        self.nfac = dB_to_rat(self.nfig) # Noise Factor, which is unitless.
        self.gain_rat = dB_to_rat(self.gain_dB) # Gain as a ratio.
        self.iip3_W = 1e-3 * dB_to_rat(self.iip3_dBm) # IIP3 in units of W.
        self.iip2_W = 1e-3 * dB_to_rat(self.iip2_dBm) # IIP2 in units of W.
        self.dyn_range = self.cpin_dBm - PIN # Dynamic range of this component.
        self.cpout_W = 1e-3 * dB_to_rat(self.cpout_dBm) # The output compression point in units of W.

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
# My values.
components = [
    Component(name='Switch', gain_dB=-2.0, nfig=0.8, iip3_dBm=30, iip2_dBm=35, cpout_dBm=20),
    Component(name='Front BPF', gain_dB=-1.5, nfig=1.0, iip3_dBm=30, iip2_dBm=35, cpout_dBm=20),
    Component(name='LNA', gain_dB=10, nfig=2, iip3_dBm=-5, iip2_dBm=0, cpout_dBm=5),
    Component(name='Mixer', gain_dB=-6, nfig=6, iip3_dBm=5, iip2_dBm=10, cpout_dBm=10),
    Component(name='BB Amp', gain_dB=60, nfig=6, iip3_dBm=10, iip2_dBm=15, cpout_dBm=15),
    Component(name='ADC Filter', gain_dB=-2.0, nfig=1, iip3_dBm=30, iip2_dBm=35, cpout_dBm=20)
]

# The values from the example.
# components = [
#     Component(name='Switch', gain_dB=-0.8, nfig=0.8, iip3_dBm=30, cpout_dBm=20),
#     Component(name='Front BPF', gain_dB=-1.0, nfig=1.0, iip3_dBm=30, cpout_dBm=20),
#     Component(name='LNA', gain_dB=15, nfig=2, iip3_dBm=-5, cpout_dBm=5),
#     Component(name='Mixer', gain_dB=-6, nfig=6, iip3_dBm=5, cpout_dBm=10),
#     Component(name='BB Amp', gain_dB=30, nfig=6, iip3_dBm=10, cpout_dBm=15),
#     Component(name='ADC Filter', gain_dB=-1, nfig=1, iip3_dBm=30, cpout_dBm=20)
# ]

###############################
##### System Noise Factor #####
###############################

# Find the noise factor of the system.
# This implements the equation found here:
# https://en.wikipedia.org/wiki/Friis_formulas_for_noise
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

#########################
##### Cascaded IIP3 #####
#########################

# This section will find the IIP3 of the entire system.
# It implements the equation found here:
# https://www.rfcafe.com/references/electrical/ip3.htm
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

#########################
##### Cascaded IIP2 #####
#########################

# This section will find the IIP2 of the entire system.
# It implements the equation found here:
# https://www.rfcafe.com/references/electrical/ip2.htm
running_sum = 1 / np.sqrt(components[0].iip2_W)
gain_term = 1
for idx, component in enumerate(components[1:], start=1):
    gain_term *= components[idx-1].gain_rat
    new_term = np.sqrt(component.iip2_W * gain_term)
    running_sum += 1/new_term
iip2_sys_W = (1/running_sum)**2
iip2_sys_dBm = rat_to_dB(iip2_sys_W/1e-3) # The system IIP2 in dBm.

######################################
##### Cascaded Compression Point #####
######################################

# This section will find the compression point of the entire system.
# It implements the equation found here:
# https://www.rfcafe.com/references/electrical/p1db.htm
running_sum = 1 / components[0].cpout_W
gain_term = 1
for idx, component in enumerate(components[1:], start=1):
    gain_term *= components[idx-1].gain_rat
    new_term = component.cpout_W * gain_term
    running_sum += 1/new_term
cpout_sys_W = (1/running_sum)**2
cpout_sys_dBm = rat_to_dB(cpout_sys_W/1e-3) # The system compression point in dBm.

##############################################
##### Calculated System Properties Table #####
##############################################

# Make a table from the calculated system properties.
data_dict = {
    'Index': np.arange(1, 10+1),
    'Metric': [
        'System Noise Figure', 'Total System Gain', 'Output SNR', 
        'Vpp at Output', 'System IIP3', 'Severe IM3 Desense', 
        'Required ADC Bits', 'Estimated BER', 'System IIP2', 'System Out CP'
    ],
    'Value': [
        f'{nfig_sys:.2f} dB', f'{gain_dB_sys:.2f} dB', f'{snr_out:.2f} dB',
        display_voltage(Vpp_out), f'{iip3_sys_dBm:.2f} dBm', 
        f'{p_im3:.2f} dBm', f'{enob:.2f} bits', f'{ber:.2e} (unitless)',
        f'{iip2_sys_dBm:.2f} dBm', f'{cpout_sys_dBm:.2f} dBm'
    ]
}

df = pd.DataFrame(data=data_dict)
fig, ax = plt.subplots(figsize=(4,2))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.savefig(os.path.join(TABLE_DIR, 'cascaded_properties.png'), dpi=300, bbox_inches='tight')

##############################
##### Signal Power Table #####
##############################

# Initalize the variables used in this table.
points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
signal_powers = list()
noise_powers = list()
signal_amplitudes = list()
snr_values = list()

# Fill in the data for this table.
current_sig_pwr = PIN
current_noise_pwr = pnoise_in
current_Vpk = dBm_to_Vpk(PIN)

signal_powers.append(f'{current_sig_pwr:.1f}')
noise_powers.append(f'{current_noise_pwr:.0f}')
signal_amplitudes.append(display_voltage(current_Vpk))
snr_values.append(f'{current_sig_pwr - current_noise_pwr:.1f}')

for component in components:

    current_sig_pwr += component.gain_dB
    current_noise_pwr += component.gain_dB + component.nfig
    current_Vpk = dBm_to_Vpk(current_sig_pwr)

    signal_powers.append(f'{current_sig_pwr:.1f}')
    noise_powers.append(f'{current_noise_pwr:.0f}')
    signal_amplitudes.append(display_voltage(current_Vpk))
    snr_values.append(f'{current_sig_pwr - current_noise_pwr:.1f}')

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
fig, ax = plt.subplots(figsize=(8,1.5))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.savefig(os.path.join(TABLE_DIR, 'signal_powers.png'), dpi=300, bbox_inches='tight')

##########################################
##### Component Specifications Table #####
##########################################

# Build the dictionary to fill in this table.
names = list()
gain_vals = list()
nfig_vals = list()
iip3_vals = list()
iip2_vals = list()
cpout_dBm_vals = list()
dyn_ranges = list()
for component in components:
    names.append(component.name)
    gain_vals.append(component.gain_dB)
    nfig_vals.append(component.nfig)
    iip3_vals.append(component.iip3_dBm)
    iip2_vals.append(component.iip2_dBm)
    cpout_dBm_vals.append(component.cpout_dBm)
    dyn_ranges.append(component.dyn_range)

data_dict = {
    'Block Name': names,
    'Gain (dB)': gain_vals,
    'Noise Figure (dB)': nfig_vals,
    'IIP3 (dBm)': iip3_vals,
    'IIP2 (dBm)': iip2_vals,
    'Output Comp (dBm)': cpout_dBm_vals,
    'Dynamic Range (dB)': dyn_ranges
}

# Create the table from that dictinoary.
df = pd.DataFrame(data=data_dict)
fig, ax = plt.subplots(figsize=(9,1.5))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.savefig(os.path.join(TABLE_DIR, 'component_specs.png'), dpi=300, bbox_inches='tight')