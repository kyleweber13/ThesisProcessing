import Filtering
import numpy as np
import ImportEDF
import matplotlib.pyplot as plt
import scipy.signal
import math

# RAW IMPORT =========================================================================================================
# Loads in raw data. Creates 3-column array

x = ImportEDF.GENEActiv(filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3036_01_GA_RWrist_Accelerometer.EDF",
                        load_raw=True, start_offset=0, end_offset=0)

raw_accel = np.asarray([x.x[0:75*86400], x.y[0:75*86400], x.z[0:75*86400]])
raw_mag = [abs(math.sqrt(math.pow(x.x[i], 2) + math.pow(x.y[i], 2) + math.pow(x.z[i], 2)) - 1) for i in range(75*86400)]

# STEP 0: DOWNSAMPLES TO 30Hz ========================================================================================
# Downsamples raw data to 30Hz
print("\n" + "Resampling data to 30Hz...")
accel30 = scipy.signal.resample(x=raw_accel, num=int(len(raw_accel[0])*30/75), axis=1)
mag30 = scipy.signal.resample(x=raw_mag, num=int(len(raw_accel[0])*30/75))

# STEP 1: 0.01-7Hz BP FILTERING ======================================================================================
# Applies 0.01-7Hz bandpass filter to 30Hz data. I chose order 3.

print("\n" + "Applying 0.01-7Hz bandpass filter...")
step1_filter = Filtering.filter_signal(data=accel30, type="bandpass",
                                       low_f=0.01, high_f=7, filter_order=3, sample_f=75)

mag_step1_filter = Filtering.filter_signal(data=mag30, type="bandpass",
                                           low_f=0.01, high_f=7, filter_order=3, sample_f=75)

del accel30, mag30

# STEP 2: 0.29-1.63Hz BANDSTOP FILTERING ==============================================================================
# Unclear on algorithm. May be bandpass or bandstop filter in range 0.29-1.63 Hz. Applies it to 30Hz data

print("\n" + "Applying mystical Step #2 filter...")

"""step2_filter = Filtering.filter_signal(data=step1_filter, type="lowpass",
                                       low_f=0.29, high_f=None, filter_order=1, sample_f=30)

step2_filter = Filtering.filter_signal(data=step2_filter, type="highpass",
                                       low_f=None, high_f=1.63, filter_order=1, sample_f=30)"""

step2_filter = Filtering.filter_signal(data=step1_filter, type="bandpass",
                                       low_f=0.29, high_f=1.63, filter_order=1, sample_f=30)

mag_step2_filter = Filtering.filter_signal(data=mag_step1_filter, type="bandpass",
                                           low_f=0.29, high_f=1.63, filter_order=1, sample_f=30)

del step1_filter, mag_step1_filter

# STEP 3: DOWNSAMPLE TO 10HZ =========================================================================================
# Downsamples ActiGraph filtered data to 10Hz

print("\n" + "Resampling down to 10Hz...")

accel10 = scipy.signal.resample(x=step2_filter, num=int(len(step2_filter[0])*10/30), axis=1)
mag10 = scipy.signal.resample(x=mag_step2_filter, num=int(len(mag_step2_filter)*10/30))

del step2_filter, mag_step2_filter

# STEP 4: TRUNCATE TO 2.13G's ========================================================================================
# Clips 10Hz data at ± 2.13 G's

print("\n" + "Truncating data to ± 2.13 G's...")

truncated = np.copy(accel10)
truncated[truncated >= 2.13] = 2.13
truncated[truncated <= -2.13] = -2.13

mag_truncated = np.copy(mag10)
mag_truncated[mag_truncated >= 2.13] = 2.13
mag_truncated[mag_truncated <= -2.13] = -2.13

del accel10, mag10

# STEP 5: RECTIFICATION ==============================================================================================
# Takes absolute value of truncated data

print("\n" + "Rectifying data...")

rectified = np.absolute(truncated)
mag_rectified = np.absolute(mag_truncated)

del truncated

# STEP 6: DEADBAND BELOW 0.068G's ====================================================================================
# Changes values below 0.068G's to 0

print("\n" + "Applying deadband (< 0.068 G's) filter...")

deadband = np.copy(rectified)
deadband[deadband <= 0.068] = 0

mag_deadband = np.copy(mag_rectified)
mag_deadband[mag_deadband <= 0.068] = 0

# STEP 7: 8-BIT CONVERSION ===========================================================================================
# Converts 10Hz deadband data to the equivalent 8-bit resolution

print("\n" + "Converting data to 8-bit resolution...")

# Bins representing value ranges covered by what would be 8-bit resolution: range = 0 to 2.13 G's
bins = np.linspace(start=0, stop=2.13, num=128)

# Arrays of what bin each value falls into
digititzed_x = np.digitize(x=deadband[0], bins=bins)
digititzed_y = np.digitize(x=deadband[1], bins=bins)
digititzed_z = np.digitize(x=deadband[2], bins=bins)

digititzed_mag = np.digitize(x=mag_deadband, bins=bins)

# Array of actual G values that correspond to each bin
bit8 = np.array([[bins[1]*i for i in digititzed_x],
                [bins[1]*i for i in digititzed_y],
                [bins[1]*i for i in digititzed_z]])

mag_bit8 = np.array([bins[1]*i for i in digititzed_mag])

del deadband, digititzed_x, digititzed_y, digititzed_z, digititzed_mag

# STEP 8: EPOCHING ===================================================================================================
# Epochs into 1-second and 60-second epochs

print("\n" + "Epoching the data...")

epoched1s_y = [sum(bit8[1, i:i+10]) for i in np.arange(1, len(bit8[1]), 10)]
epoched60s_y = [sum(epoched1s_y[i:i+60]) for i in np.arange(0, len(epoched1s_y), 60)]

epoched1s_mag = [sum(mag_bit8[i:i+10]) for i in np.arange(1, len(mag_bit8), 10)]
epoched60s_mag = [sum(mag_bit8[i:i+60]) for i in np.arange(1, len(epoched1s_mag), 60)]
