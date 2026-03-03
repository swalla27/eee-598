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

# Frequencies have units of Hz and powers have units of dBm.
class Tone():
    def __init__(self, freq: float, pwr: float, snr: float):
        self.init_freq = freq
        self.init_pwr = pwr
        self.init_snr = snr
        self.freq = freq
        self.pwr = pwr
        self.snr = snr

desired = Tone(2402e6, -30, 0)
blocker = Tone(2403e6, -20, 0)
signals = [desired, blocker]

def bandpass(sig, f_center=2402e6, bandwidth=40e6, atten_pass=0, atten_stop=35, nf=2):
    sig.snr -= nf
    if (f_center - bandwidth/2) < sig.freq < (f_center + bandwidth/2):
        sig.pwr -= atten_pass
        return
    else:
        sig.pwr -= atten_stop
        return
    
def lna(sig, gain=15, nf=2):
    sig.pwr += gain
    sig.snr -= nf
    return

def mixer(sig, f_osc=2412e6, gain=0, nf=2):
    sig.freq = abs(sig.freq - f_osc)
    sig.pwr += gain
    sig.snr -= nf
    return

def lowpass(sig, f_corner=15e6, atten_pass=0, atten_stop=50, nf=2):
    sig.snr -= nf
    if sig.freq <= f_corner:
        sig.pwr -= atten_pass
        return
    else:
        sig.pwr -= atten_stop
        return

def if_amp(sig, gain=60, nf=2):
    sig.pwr += gain
    sig.snr -= nf
    return

for signal in signals:
    bandpass(signal)
    lna(signal)
    mixer(signal)
    lowpass(signal)
    if_amp(signal)
    bandpass(signal, f_center=10e6, bandwidth=1e6)

print(f'Desired Signal:')
print(f'\t{desired.init_freq/1e6:.0f} MHz -> {desired.freq/1e6:.0f} MHz')
print(f'\t{desired.init_pwr} dBm -> {desired.pwr} dBm')
# print(f'\t{desired.init_snr} dB -> {desired.snr} dB')


print(f'Blocker Signal:')
print(f'\t{blocker.init_freq/1e6:.0f} MHz -> {blocker.freq/1e6:.0f} MHz')
print(f'\t{blocker.init_pwr} dBm -> {blocker.pwr} dBm')
# print(f'\t{blocker.init_snr} dB -> {blocker.snr} dB')
