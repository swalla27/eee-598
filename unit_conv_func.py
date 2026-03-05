# Functions used for unit conversion throughout this class.

import numpy as np

def dB_to_rat(dB: float):
    return 10**(dB/10)

def rat_to_dB(rat: float):
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