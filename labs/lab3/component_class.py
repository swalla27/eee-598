# Steven Wallace
# Professor Sayfe Kiaei
# EEE 598
# 3 March 2026

# Lab 3 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time
import sys
import os

print(os.getcwd())
print(os.listdir())


from unit_conv_func import *


class Component():
    def __init__(self, name: str, gdb: float, nfig: float, iip3: float, cpout: float):
        self.name = name
        self.gdb = gdb # Gain in dB.
        self.nfig = nfig # Noise Figure, which has units of dB.
        self.iip3 = iip3
        self.cpout = cpout # Output 1dB compression point.

        self.nfac = dB_to_rat(self.nfig) # Noise Factor, which is unitless.
        self.grat = dB_to_rat(self.gdb) # Gain as a ratio.

components = [
    Component(name='Switch', gdb=-0.8, nfig=0.8, iip3=30, cpout=20),
    Component(name='Front BPF', gdb=-1.0, nfig=1.0, iip3=30, cpout=20),
    Component(name='LNA', gdb=15, nfig=2.0, iip3=-5, cpout=5),
    Component(name='Mixer', gdb=-6, nfig=6, iip3=5, cpout=10),
    Component(name='BB Amp', gdb=30, nfig=6, iip3=10, cpout=15),
    Component(name='ADC Filter', gdb=-1, nfig=1, iip3=30, cpout=20)
]

# Find the noise factor of the system.
nfac_sys = components[0].nfac
denominator = 1
for idx, component in enumerate(components[1:], start=1):
    numerator = component.nfac - 1
    denominator *= components[idx-1].grat
    nfac_sys += numerator/denominator

nfig_sys = rat_to_dB(nfac_sys) # Noise Figure of the whole system.

# Find the total gain of the system.
gdb_sys = 0
for component in components:
    gdb_sys += component.gdb

NOISE_PWR = -114 # Noise power in dBm.
PIN = -70 # Input power in dBm.
snr_in = PIN - NOISE_PWR # Input SNR in dB.
snr_out = snr_in - nfig_sys # Output SNR in dB.

pout = PIN + gdb_sys # The output power in dBm.