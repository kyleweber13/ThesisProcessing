import Filtering
import numpy as np
import ImportEDF
import matplotlib.pyplot as plt
import scipy.signal
import math

# RAW IMPORT =========================================================================================================


def import_edf(filepath):
    """Loads in raw data using ImportEDF script. Returns GENEActiv class object.

    :argument
    -filepath: pathway to EDF file
    """

    data = ImportEDF.GENEActiv(filepath=filepath, load_raw=True, start_offset=0, end_offset=0)

    return data


class ActigraphConversion:

    def __init__(self, raw_data=None, epoch_len=15, start_day=0, end_day=1):
        """Class that follows the processing steps proposed by Brond, Andersen, and Arvidsson (2017) to convert
           raw accelerometer data to ActiGraph counts.

        :argument
        -raw_data: object from import_edf function
        -epoch_len: epoch lenght, seconds
        -start_day, end_day: used for cropping data by day
        """

        self.raw_data = raw_data
        self.sample_rate = self.raw_data.sample_rate

        self.epoch_len = epoch_len

        self.start_day = start_day
        self.end_day = end_day

        self.raw_accel = None
        self.raw_mag = None

        # Step 1 data
        self.accel30hz = None
        self.mag_start_30hz = None

        # Step 2 data
        self.step1_filter = None
        self.mag_start_step1_filter = None

        # Step 3 data
        self.step2_filter = None
        self.mag_start_step2_filter = None

        # Step 4 data
        self.accel10hz = None
        self.mag_start_10hz = None

        # Step 5 data
        self.truncated = None
        self.mag_start_truncated = None

        # Step 6 data
        self.rectified = None
        self.mag_start_rectified = None

        # Step 7 data
        self.deadband = None
        self.mag_start_deadband = None

        # Step 8 data
        self.bit8 = None
        self.mag_start_bit8 = None
        self.mag_end_8bit = None

        self.epoch_y = None
        self.epoch_mag_start = None
        self.epoch_mag_end = None

        """RUNS METHODS"""
        self.crop_data()
        self.downsample_30hz()  # Step 0
        self.antialias_filter()  # Step 1
        self.actigraph_filter()  # Step 2
        self.downsample_10hz()  # Step 3
        self.truncate_data()  # Step 4
        self.rectify_data()  # Step 5
        self.deadband_filter()  # Step 6
        self.convert_to_8bit()  # Step 7
        self.epoch_data()  # Step 8

    def crop_data(self):
        """-Crops data according to start_day and end_day arguments.
           -Calculates vector magnitude for selected section.
        """

        self.raw_accel = np.asarray([self.raw_data.x[self.sample_rate * 86400 * self.start_day:
                                                     self.sample_rate * 86400 * self.end_day],
                                     self.raw_data.y[self.sample_rate * 86400 * self.start_day:
                                                     self.sample_rate * 86400 * self.end_day],
                                     self.raw_data.x[self.sample_rate * 86400 * self.start_day:
                                                     self.sample_rate * 86400 * self.end_day]])

        # Subtracts gravity
        self.raw_mag = [abs(math.sqrt((math.pow(self.raw_data.x[i], 2) +
                                       math.pow(self.raw_data.y[i], 2) +
                                       math.pow(self.raw_data.z[i], 2))) - 1)
                        for i in range(int((self.end_day - self.start_day) * self.sample_rate * 86400))]

        # Does not subtract gravity
        """self.raw_mag = [abs(math.sqrt((math.pow(self.raw_data.x[i], 2) +
                                       math.pow(self.raw_data.y[i], 2) +
                                       math.pow(self.raw_data.z[i], 2))))
                        for i in range(int((self.end_day - self.start_day) * self.sample_rate * 86400))]"""

    def downsample_30hz(self):
        """Downsamples data to 30Hz to match typical ActiGraph sampling rate."""

        # STEP 0: DOWNSAMPLES TO 30Hz =================================================================================
        print("\n" + "Resampling data to 30Hz...")

        self.accel30hz = scipy.signal.resample(x=self.raw_accel,
                                               num=int(len(self.raw_accel[0]) * 30 / self.sample_rate), axis=1)

        self.mag_start_30hz = scipy.signal.resample(x=self.raw_mag, num=int(len(self.raw_mag) * 30 / self.sample_rate))

        print("Complete.")

    def antialias_filter(self):
        """Applies 0.01 - 7.0 Hz bandpass filter to prevent aliasing."""

        # STEP 1: 0.01-7Hz BP FILTERING ==============================================================================

        print("\n" + "Applying 0.01-7Hz bandpass filter...")

        self.step1_filter = Filtering.filter_signal(data=self.accel30hz, type="bandpass",
                                                    low_f=0.01, high_f=7, filter_order=1, sample_f=self.sample_rate)

        self.mag_start_step1_filter = Filtering.filter_signal(data=self.mag_start_30hz, type="bandpass",
                                                              low_f=0.01, high_f=7,
                                                              filter_order=1, sample_f=self.sample_rate)

        print("Complete.")

    def actigraph_filter(self):
        """Applies 0.29 - 1.63 Hz bandpass filter to match on-board ActiGraph processing."""

        # STEP 2: 0.29-1.63Hz BANDPASS FILTERING =====================================================================

        print("\n" + "Applying mystical Step #2 filter...")

        self.step2_filter = Filtering.filter_signal(data=self.step1_filter, type="bandpass",
                                                    low_f=0.29, high_f=1.63, filter_order=1, sample_f=30)

        self.mag_start_step2_filter = Filtering.filter_signal(data=self.mag_start_step1_filter, type="bandpass",
                                                              low_f=0.29, high_f=1.63, filter_order=1, sample_f=30)

        print("Complete.")

    def downsample_10hz(self):
        """Further downsamples data to 10Hz."""

        # STEP 3: DOWNSAMPLE TO 10HZ =================================================================================

        print("\n" + "Resampling down to 10Hz...")

        self.accel10hz = scipy.signal.resample(x=self.step2_filter,
                                               num=int(len(self.step2_filter[0]) * 10 / 30), axis=1)

        self.mag_start_10hz = scipy.signal.resample(x=self.mag_start_step2_filter,
                                                    num=int(len(self.mag_start_step2_filter) * 10 / 30))

        print("Complete.")

    def truncate_data(self):
        """Truncates data to the ± 2.13 G range to match ActiGraph response range."""

        # STEP 4: TRUNCATE TO 2.13G's ================================================================================

        print("\n" + "Truncating data to ± 2.13 G's...")

        self.truncated = np.copy(self.accel10hz)
        self.truncated[self.truncated >= 2.13] = 2.13
        self.truncated[self.truncated <= -2.13] = -2.13

        self.mag_start_truncated = np.copy(self.mag_start_10hz)
        self.mag_start_truncated[self.mag_start_truncated >= 2.13] = 2.13

        print("Complete.")

    def rectify_data(self):
        """Rectifies the data."""

        # STEP 5: RECTIFICATION ======================================================================================

        print("\n" + "Rectifying data...")

        self.rectified = np.absolute(self.truncated)
        self.mag_start_rectified = np.absolute(self.mag_start_truncated)

        print("Complete.")

    def deadband_filter(self):
        """Applies deadband filter corresponding to < 0.068 G."""

        # STEP 6: DEADBAND BELOW 0.068G's ============================================================================

        print("\n" + "Applying deadband (< 0.068 G's) filter...")

        self.deadband = np.copy(self.rectified)
        self.deadband[self.deadband <= 0.068] = 0

        self.mag_start_deadband = np.copy(self.mag_start_rectified)
        self.mag_start_deadband[self.mag_start_deadband <= 0.068] = 0

        print("Complete.")

    def convert_to_8bit(self):
        """-Converts data to the equivalent if it had been collected with 8-bit resolution.
           -Calculates vector magnitude in a second method.
        """

        # STEP 7: 8-BIT CONVERSION ===================================================================================

        print("\n" + "Converting data to 8-bit resolution...")

        # Bins representing value ranges covered by what would be 8-bit resolution: range = 0 to 2.13 G's
        bins = np.linspace(start=0, stop=2.13, num=128)

        # Arrays of what bin each value falls into
        digititzed_x = np.digitize(x=self.deadband[0], bins=bins)
        digititzed_y = np.digitize(x=self.deadband[1], bins=bins)
        digititzed_z = np.digitize(x=self.deadband[2], bins=bins)

        digititzed_mag = np.digitize(x=self.mag_start_deadband, bins=bins)

        # Array of actual G values that correspond to each bin
        self.bit8 = np.array([[bins[1]*i for i in digititzed_x],
                              [bins[1]*i for i in digititzed_y],
                              [bins[1]*i for i in digititzed_z]])

        self.mag_start_bit8 = np.array([bins[1]*i for i in digititzed_mag])

        # Subtracts gravity
        """self.mag_end_8bit = [abs(math.sqrt(math.pow(self.bit8[0, i], 2) +
                                           math.pow(self.bit8[1, i], 2) +
                                           math.pow(self.bit8[2, i], 2)) - 1)
                             for i in range(len(self.bit8[0]))]"""

        # Does not subtract gravity
        self.mag_end_8bit = [abs(math.sqrt(math.pow(self.bit8[0, i], 2) +
                                           math.pow(self.bit8[1, i], 2) +
                                           math.pow(self.bit8[2, i], 2)))
                             for i in range(len(self.bit8[0]))]

        print("Complete.")

    def epoch_data(self):
        """Epochs data by taking the sum of a jumping window."""

        # STEP 8: EPOCHING ===========================================================================================

        print("\n" + "Epoching the data...")

        self.epoch_y = [sum(self.bit8[1, i:i + self.epoch_len * 10])
                        for i in np.arange(0, len(self.bit8[1]), self.epoch_len * 10)]

        self.epoch_mag_start = [sum(self.mag_start_bit8[i:i + self.epoch_len * 10])
                                for i in np.arange(1, len(self.mag_start_bit8), self.epoch_len * 10)]

        self.epoch_mag_end = [sum(self.mag_end_8bit[i: i + self.epoch_len * 10])
                              for i in np.arange(1, len(self.mag_end_8bit), self.epoch_len * 10)]

        print("Complete.")

    def plot_epoched(self):
        """Plots epoched data (single-axis and vector magnitude)."""

        fig, (ax1) = plt.subplots(1, figsize=(10, 7))

        ax1.plot(np.arange(0, len(self.epoch_y))/(self.epoch_len/60), self.epoch_y, color='red', label="Y-axis")

        ax1.plot(np.arange(0, len(self.epoch_mag_start))/(self.epoch_len/60), self.epoch_mag_start,
                 color='black', label="Magnitude Start")

        ax1.plot(np.arange(0, len(self.epoch_mag_end))/(self.epoch_len/60), self.epoch_mag_end,
                 color='blue', label="Magnitude End")

        ax1.legend(loc='upper left')
        ax1.set_xlabel("Minutes")
        ax1.set_title("{}-second epoched data".format(self.epoch_len))

    def plot_steps(self):
        """Plots data from most of the processing steps."""

        fig, axs = plt.subplots(nrows=4, ncols=1, sharex='col', figsize=(10, 7))
        axs[0].set_title("Most Processing Steps")

        axs[0].plot(np.arange(0, len(self.raw_accel[1])) / (self.sample_rate * 60), self.raw_accel[1],
                    label="Raw_y ({}Hz)".format(self.sample_rate), color='red')
        axs[0].set_ylabel("G")
        axs[0].legend(loc='upper left')

        axs[1].plot(np.arange(0, len(self.step2_filter[1])) / (30 * 60), self.step2_filter[1],
                    label="0.29-1.63Hz BP (30Hz)", color='black')
        axs[1].plot(np.arange(0, len(self.truncated[1])) / (10 * 60), self.truncated[1],
                    label="Truncated (10Hz)", color='red')
        axs[1].set_ylabel("G")
        axs[1].legend(loc='upper left')

        axs[2].plot(np.arange(0, len(self.rectified[1])) / (10 * 60), self.rectified[1],
                    label="Rectified (10Hz)", color='black')
        axs[2].plot(np.arange(0, len(self.bit8[1])) / (10 * 60), self.bit8[1],
                    label="Deadband + 8-bit (10Hz)", color='red')
        axs[2].legend(loc='upper left')
        axs[2].set_ylabel("G")

        axs[3].bar(x=np.arange(0, len(self.epoch_y)) / (60 / self.epoch_len), height=self.epoch_y, align='edge',
                   color="red", edgecolor='black', width=60 / self.epoch_len,
                   label="{}-sec epoch (vertical)".format(self.epoch_len))
        axs[3].legend(loc='upper left')
        axs[3].set_xlabel("Minutes")
        axs[3].set_ylabel("Counts")

        plt.show()
