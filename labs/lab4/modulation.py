import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq, fft, ifft
import scipy.signal as signal
import sys

CAR_FREQ = 2400e6
CAR_PER = 1 / CAR_FREQ
BIT_DUR = 100e-6
BIT_RATE = 1 / BIT_DUR
NUM_BITS = 10
SIM_DUR = NUM_BITS * BIT_DUR
TSTEP = 0.1 / CAR_FREQ
FREQWIN_WIDTH = 20e3

rng = np.random.default_rng()
bits = rng.integers(0, 2, NUM_BITS)
bits = np.where(bits == 0, 0.5, 2)

#######################
##### Time Domain #####
#######################

# TSTEP = 0.1/BIT_RATE

cutoff_factor = 1.0
cutoff_hz = cutoff_factor * BIT_RATE
fs = 1.0 / TSTEP
Wn = cutoff_hz / (fs/2.0)

t = np.arange(0, SIM_DUR, TSTEP)
modsig_t = np.repeat(bits, int(BIT_DUR/TSTEP))
lpf = signal.butter(N=3, Wn=Wn, btype='low', output='sos')
filtsig_t = signal.sosfilt(lpf, modsig_t)

# plt.plot(t, modsig_t, label='Unfiltered')
# plt.plot(t, filtsig_t, label='Filtered')
# plt.legend()
# plt.show()
# sys.exit()

carsig_t = np.sin(2*np.pi*CAR_FREQ*t)
rfsig_t = filtsig_t * carsig_t

plt.subplot(3, 1, 1)
plt.plot(t, filtsig_t)
plt.ylabel('Mod Signal')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, carsig_t)
plt.ylabel('Carrier Signal')
plt.xlim([0, 5*CAR_PER])
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, rfsig_t)
plt.xlabel('Time (s)')
plt.ylabel('RF Signal')
plt.grid(True)

plt.suptitle('Time Domain Signals')
plt.show()

############################
##### Frequency Domain #####
############################

f_raw = fftfreq(rfsig_t.size, d=TSTEP)
pidxs = np.nonzero(f_raw > 0)
f_pos = f_raw[pidxs]

filtsig_f = fft(filtsig_t)
carsig_f = fft(carsig_t)
rfsig_f = fft(rfsig_t)

filtsig_fpow = np.abs(filtsig_f)[pidxs]
carsig_fpow = np.abs(carsig_f)[pidxs]
rfsig_fpow = np.abs(rfsig_f)[pidxs]

plt.subplot(3, 1, 1)
plt.plot(f_pos, filtsig_fpow)
plt.ylabel('Mod Signal')
plt.xlim([0, 5*BIT_RATE])
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(f_pos, carsig_fpow)
plt.ylabel('Carrier Signal')
plt.xlim([CAR_FREQ-FREQWIN_WIDTH, CAR_FREQ+FREQWIN_WIDTH])
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(f_pos, rfsig_fpow)
plt.xlabel('Frequency (Hz)')
plt.ylabel('RF Signal')
plt.xlim([CAR_FREQ-FREQWIN_WIDTH, CAR_FREQ+FREQWIN_WIDTH])
plt.grid(True)

plt.suptitle('Frequency Domain Signals')
plt.show()