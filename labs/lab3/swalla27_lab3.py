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
        self.freq = freq
        self.pwr = pwr
        self.noise_pwr = pwr - snr

    def get_snr(self):
        return self.pwr - self.noise_pwr

def bandpass(signals, f_center: float, bandwidth: float, atten_pass: float, atten_stop: float, nf: float):
    for sig in signals:
        sig.noise_pwr += nf
        if (f_center - bandwidth/2) < sig.freq < (f_center + bandwidth/2):
            sig.pwr -= atten_pass
        else:
            sig.pwr -= atten_stop
    return signals
    
def amplifier(signals, gain: float, nf: float, iip3: float):

    tmp = list()

    # Handle the fundamental tone first.
    signals[0].pwr += gain
    signals[0].noise_pwr += nf
    tmp.append(signals[0])

    for sig in signals[1:]:
        # Handle the blocker itself, it gets amplified and its noise power changes.
        sig.pwr += gain
        sig.noise_pwr += nf
        tmp.append(sig)

        # Now, create the third order intermodulation products.
        im_product_1 = Tone(2*signals[0].freq - sig.freq, 3*signals[0].pwr - 2*iip3 + gain, 0)
        im_product_2 = Tone(2*sig.freq - signals[0].freq, 3*signals[0].pwr - 2*iip3 + gain, 0)
        tmp.append(im_product_1)
        tmp.append(im_product_2)

    return tmp

def mixer(signals, f_osc: float, gain: float, nf: float):
    for sig in signals:
        sig.freq = abs(sig.freq - f_osc)
        sig.pwr += gain
        sig.noise_pwr += nf
    return signals

def lowpass(signals, f_corner: float, atten_pass: float, atten_stop: float, nf: float):
    for sig in signals:
        sig.noise_pwr += nf
        if sig.freq <= f_corner:
            sig.pwr -= atten_pass
        else:
            sig.pwr -= atten_stop
    return signals


if __name__ == "__main__":

    # All frequencies have units of MHz and all powers have units of dBm.

    block_freq = 2403
    block_pwr = -60

    desired = Tone(freq=2402, pwr=-60, snr=18)
    blocker = Tone(freq=block_freq, pwr=block_pwr, snr=18)
    signals = [desired, blocker]

    signals = bandpass(signals, f_center=2402, bandwidth=40, atten_pass=0, atten_stop=35, nf=2)

    signals = amplifier(signals, gain=15, nf=2, iip3=14.5)

    signals = mixer(signals, f_osc=2412, gain=0, nf=2)

    signals = lowpass(signals, f_corner=15, atten_pass=0, atten_stop=50, nf=2)

    signals = amplifier(signals, gain=60, nf=10, iip3=14.5)

    signals = bandpass(signals, f_center=10, bandwidth=1, atten_pass=0, atten_stop=35, nf=2)

    for idx, sig in enumerate(signals):
        print(f'Signal {idx}:')
        print(f'\tFreq: {sig.freq} MHz')
        print(f'\tPower: {sig.pwr} dBm')
        print(f'\tSNR: {sig.get_snr()} dB')


