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
        self.freq = freq
        self.pwr = pwr
        self.noise_pwr = pwr - snr

desired = Tone(2402e6, -30, 0)
blocker = Tone(2403e6, -20, 0)
signals = [desired, blocker]

def bandpass(signals, f_center: float, bandwidth: float, atten_pass: float, atten_stop: float, nf: float):
    for sig in signals:
        sig.snr -= nf
        if (f_center - bandwidth/2) < sig.freq < (f_center + bandwidth/2):
            sig.pwr -= atten_pass
        else:
            sig.pwr -= atten_stop
    return
    
def amplifier(signals, gain: float, nf: float):
    for sig in signals:
        sig.pwr += gain
        sig.snr -= nf
    return

def mixer(signals, f_osc: float, gain: float, nf: float):
    x = list()
    for sig in signals:
        sig.freq = abs(sig.freq - f_osc)
        sig.pwr += gain
        sig.snr -= nf
        x.append(sig)
        x.append(Tone(freq=sig.freq+f_osc, pwr=sig.pwr+gain, snr=sig.snr-nf))
    return x

def lowpass(signals, f_corner: float, atten_pass: float, atten_stop: float, nf: float):
    for sig in signals:
        sig.snr -= nf
        if sig.freq <= f_corner:
            sig.pwr -= atten_pass
        else:
            sig.pwr -= atten_stop
    return

def create_graph(signals: list, name: str):
    for sig in signals:
        plt.scatter(sig.freq/1e6, sig.pwr)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dBm)')
    plt.title(f'Frequency Spectrum ({name})')
    plt.grid(True)
    # plt.show()

bandpass(signals, f_center=2402e6, bandwidth=40e6, atten_pass=0, atten_stop=35, nf=2)
# create_graph(signals, name='A')

amplifier(signals, gain=15, nf=2)
# create_graph(signals, name='B')

signals = mixer(signals, f_osc=2412e6, gain=0, nf=2)
# create_graph(signals, name='C')

lowpass(signals, f_corner=15e6, atten_pass=0, atten_stop=50, nf=2)
# create_graph(signals, name='D')

amplifier(signals, gain=60, nf=10)
# create_graph(signals, name='E')

bandpass(signals, f_center=10e6, bandwidth=1e6, atten_pass=0, atten_stop=35, nf=2)
# create_graph(signals, name='F')


print(f'Desired Signal:')
print(f'\t{desired.init_freq/1e6:.0f} MHz -> {desired.freq/1e6:.0f} MHz')
print(f'\t{desired.init_pwr} dBm -> {desired.pwr} dBm')
# print(f'\t{desired.init_snr} dB -> {desired.snr} dB')

print(f'Blocker Signal:')
print(f'\t{blocker.init_freq/1e6:.0f} MHz -> {blocker.freq/1e6:.0f} MHz')
print(f'\t{blocker.init_pwr} dBm -> {blocker.pwr} dBm')
# print(f'\t{blocker.init_snr} dB -> {blocker.snr} dB')
