import ImportEDF

from ecgdetectors import Detectors
# https://github.com/luishowell/ecg-detectors

from matplotlib import pyplot as plt
import numpy as np
import statistics
import scipy.stats as stats
from datetime import datetime
import csv
import progressbar
from matplotlib.ticker import PercentFormatter
from random import randint


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- ECG CLASS OBJECT ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class ECG:

    def __init__(self, filepath=None, output_dir=None, age=0, start_offset=0, end_offset=0,
                 rest_hr_window=60, n_epochs_rest=10,
                 epoch_len=15,
                 filter=False, low_f=1, high_f=30, f_type="bandpass",
                 load_raw=False, from_processed=True, write_results=True):
        """Class that contains raw and processed ECG data.

        :argument
        DATA IMPORT
        -filepath: full pathway to EDF file
        -load_raw: boolean of whether to load raw ECG data. Can be used in addition to from_processed = True
        -from_processed: boolean of whether to read in already processed data
                         (epoch timestamps, epoch HR, quality control check)
        -output_dir: where files are written to OR where processed data files are read in from
        -start_offset, end_offset: indexes used to crop data to match other devices

        DATA EPOCHING
        -rest_hr_window: number of seconds over which HR is averaged when calculating resting HR
            -Creates a rolling average of HR over this many seconds (rounded to match epoch length)
        -n_epochs_rest: number of epochs used in the resting HR calculation
                        (averages HR over the n_epochs_rest lower HRs)
        -epoch_len: time period over which data is processed, seconds

        FILTERING
        -filter: whether or not to filter the data
        -low_f, high_1: cut-off frequencies for the filter. Set to None if irrelevant. In Hz.
        -f_type: type of filter; "lowpass", "highpass", "bandpass"

        OTHER
        -write_results: boolean of whether to write output file
        -age: participant age in years. Needed for HRmax calculation.
        """

        print()
        print("============================================= ECG DATA ==============================================")

        self.filepath = filepath
        self.filename = self.filepath.split("/")[-1].split(".")[0]
        self.subjectID = self.filename.split("_")[2]
        self.output_dir = output_dir
        self.age = age
        self.epoch_len = epoch_len
        self.rest_hr_window = rest_hr_window
        self.n_epochs_rest = n_epochs_rest
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.filter = filter
        self.low_f = low_f
        self.high_f = high_f
        self.f_type = f_type

        self.load_raw = load_raw
        self.from_processed = from_processed
        self.write_results = write_results

        # Raw data
        if self.load_raw:
            self.ecg = ImportEDF.Bittium(filepath=self.filepath,
                                         start_offset=self.start_offset, end_offset=self.end_offset,
                                         filter=self.filter, low_f=self.low_f, high_f=self.high_f, f_type=self.f_type)

            self.sample_rate = self.ecg.sample_rate
            self.raw = self.ecg.raw
            self.filtered = self.ecg.filtered
            self.timestamps = self.ecg.timestamps
            self.epoch_timestamps = self.ecg.epoch_timestamps

            del self.ecg

        # Performs quality control check on raw data and epochs data
        if not self.from_processed:
            self.epoch_validity, self.epoch_hr = self.check_quality()

        # Loads epoched data from existing file
        if self.from_processed:
            self.epoch_timestamps, self.epoch_validity, self.epoch_hr = self.load_processed()

        # List of epoched heart rates but any invalid epoch is marked as None instead of 0 (as is self.epoch_hr)
        self.valid_hr = [self.epoch_hr[i] if self.epoch_validity[i] == 0 else None for i in range(len(self.epoch_hr))]

        self.quality_report = self.generate_quality_report()

        self.rolling_avg_hr = None
        self.rest_hr = None
        self.perc_hrr = None
        self.epoch_intensity = None
        self.epoch_intensity_totals = None

        # This block is called later from the Subject class after sleep data is processed
        # self.rolling_avg_hr, self.rest_hr, self.awake_hr = self.find_resting_hr(window_size=self.rest_hr_window)
        # self.perc_hrr = self.calculate_percent_hrr()
        # self.epoch_intensity, self.intensity_totals = self.calculate_intensity()

        if self.write_results:
            self.write_output()

    def check_quality(self):
        """Performs quality check using Orphanidou et al. (2015) algorithm that has been tweaked to factor in voltage
           range as well.

           This function runs a loop that creates object from the class CheckQuality for each epoch in the raw data.
        """

        print("\n" + "Running quality check with Orphanidou et al. (2015) algorithm...")

        t0 = datetime.now()

        validity_list = []
        epoch_hr = []

        bar = progressbar.ProgressBar(maxval=len(self.raw),
                                      widgets=[progressbar.Bar('>', '', '|'), ' ',
                                               progressbar.Percentage()])
        bar.start()

        for start_index in range(0, int(len(self.raw)), self.epoch_len*self.sample_rate):
            bar.update(start_index + 1)

            qc = CheckQuality(ecg_object=self, start_index=start_index, epoch_len=self.epoch_len)

            if qc.valid_period:
                validity_list.append(0)
                epoch_hr.append(round(qc.hr, 2))
            if not qc.valid_period:
                validity_list.append(1)
                epoch_hr.append(0)

        bar.finish()

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("\n" + "Quality check complete ({} seconds).".format(round(proc_time, 2)))

        return validity_list, epoch_hr

    def generate_quality_report(self):
        """Calculates how much of the data was usable. Returns values in dictionary."""

        invalid_epochs = self.epoch_validity.count(1)  # number of invalid epochs
        hours_lost = round(invalid_epochs / (60 / self.epoch_len) / 60, 2)  # hours of invalid data
        perc_invalid = round(invalid_epochs / len(self.epoch_validity) * 100, 1)  # percent of invalid data

        # Longest valid period
        longest_valid = count = 0
        current = ''
        for epoch in self.epoch_validity:
            if epoch == current and epoch == 0:
                count += 1
            else:
                count = 1
                current = epoch
            longest_valid = max(count, longest_valid)

        # Longest invalid
        longest_invalid = count = 0
        current = ''
        for epoch in self.epoch_validity:
            if epoch == current and epoch == 1:
                count += 1
            else:
                count = 1
                current = epoch
            longest_invalid = max(count, longest_invalid)

        quality_report = {"Invalid epochs": invalid_epochs, "Hours lost": hours_lost,
                          "Percent invalid": perc_invalid,
                          "Longest valid period": longest_valid, "Longest invalid period": longest_invalid,
                          "Average valid duration (minutes)": None}

        print("{}% of the data is valid.".format(round(100-perc_invalid), 2))

        return quality_report

    def load_processed(self):
        """Method to load previously-processed epoched timestamp, HR and validity data.

        :returns
        -epoch_timestamps: timestamp for start of each epoch
        -epoch_validity: binary list (0=valid; 1=invalid) of whether data is usable
        -epoch_hr: average HR in epoch
        """

        print("\n" + "Loading existing data for {}...".format(self.filepath))

        epoch_timestamps, epoch_validity, epoch_hr = np.loadtxt(fname=self.output_dir + "Model Output/" +
                                                                      self.filename + "_IntensityData.csv",
                                                                delimiter=",", skiprows=1, usecols=(0, 1, 2),
                                                                unpack=True, dtype="str")

        # epoch_timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in epoch_timestamps]
        # epoch_timestamps = [datetime.strptime(i[:-3], "%Y-%m-%dT%H:%M:%S.%f") for i in epoch_timestamps]

        epoch_timestamps_formatted = []
        for epoch in epoch_timestamps:
            try:
                epoch_timestamps_formatted.append(datetime.strptime(epoch[:-3], "%Y-%m-%dT%H:%M:%S.%f"))
            except ValueError:
                epoch_timestamps_formatted.append(datetime.strptime(epoch.split(".")[0], "%Y-%m-%d %H:%M:%S"))

        epoch_validity = [int(i) for i in epoch_validity]
        epoch_hr = [round(float(i), 2) for i in epoch_hr]

        print("Complete.")

        return epoch_timestamps_formatted, epoch_validity, epoch_hr

    def find_resting_hr(self, window_size=60, n_windows=10, sleep_status=None):
        """Function that calculates resting HR based on inputs.

        :argument
        -window_size: size of window over which rolling average is calculated, seconds
        -n_windows: number of epochs over which resting HR is averaged (lowest n_windows number of epochs)
        -sleep_status: data from class Sleep that corresponds to asleep/awake epochs
        """

        # Sets integer for window length based on window_size and epoch_len
        window_len = int(window_size / self.epoch_len)

        rolling_avg = [statistics.mean(self.epoch_hr[i:i + window_len]) if 0 not in self.epoch_hr[i:i + window_len]
                       else None for i in range(len(self.epoch_hr))]

        # Calculates resting HR during waking hours if sleep_log available --------------------------------------------
        if sleep_status is not None:
            print("\n" + "Calculating resting HR from periods of wakefulness...")

            awake_hr = [rolling_avg[i] for i in range(0, min(len(sleep_status), len(rolling_avg)))
                        if sleep_status[i] == 0 and rolling_avg[i] is not None]

            sorted_hr = sorted(awake_hr)

            resting_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

            print("Resting HR (average of {} lowest {}-second periods while awake) is {} bpm.".format(n_windows,
                                                                                                      window_size,
                                                                                                      resting_hr))

        # Calculates resting HR during all hours if sleep_log not available -------------------------------------------
        if sleep_status is None:
            print("\n" + "Calculating resting HR from periods of all data (sleep data not available)...")

            awake_hr = None

            sorted_hr = sorted(rolling_avg)

            resting_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

            print("No sleep data found so resting HR cannot be calculated. "
                  "But you probably knew that since you likely got an error...")
            print("If you did want an estimate of resting HR including sleep, 'resting heart rate' (average of {} "
                  "lowest {}-second periods) is {} bpm.".format(n_windows, window_size, resting_hr))

        return rolling_avg, resting_hr, awake_hr

    def calculate_percent_hrr(self):
        """Calculates HR as percent of heart rate reserve using resting heart rate and predicted HR max using the
           equation from Tanaka et al. (2001).
           Removes negative %HRR values which are possible due to how resting HR is defined.
        """

        hr_max = 208 - 0.7 * self.age

        perc_hrr = [round(100 * (hr - self.rest_hr) / (hr_max - self.rest_hr), 2) if hr
                    is not None else None for hr in self.valid_hr]

        # A single epoch's HR can be below resting HR based on how it's defined
        # Changes any negative values to 0, maintains Nones and positive values
        # Can't figure out how to do this as a list comprehension - don't judge
        perc_hrr_final = []

        for i in perc_hrr:
            if i is not None:
                if i >= 0:
                    perc_hrr_final.append(i)
                if i < 0:
                    perc_hrr_final.append(0)
            if i is None:
                perc_hrr_final.append(None)

        return perc_hrr_final

    def calculate_intensity(self):
        """Calculates intensity category based on %HRR ranges.
           Sums values to determine total time spent in each category.

        :returns
        -intensity: epoch-by-epoch categorization by intensity. 0=sedentary, 1=light, 2=moderate, 3=vigorous
        -intensity_minutes: total minutes spent at each intensity, dictionary
        """

        # INTENSITIY DEFINITIONS
        # Sedentary = %HRR < 30, light = 30 < %HRR <= 40, moderate = 40 < %HRR <= 60, vigorous = %HRR >= 60

        intensity = []

        for hrr in self.perc_hrr:
            if hrr is None:
                intensity.append(None)

            if hrr is not None:
                if hrr < 30:
                    intensity.append(0)
                if 30 <= hrr < 40:
                    intensity.append(1)
                if 40 <= hrr < 60:
                    intensity.append(2)
                if hrr >= 60:
                    intensity.append(3)

        n_valid_epochs = len(self.valid_hr) - self.quality_report["Invalid epochs"]

        if n_valid_epochs == 0:
            n_valid_epochs = len(self.valid_hr)

        # Calculates time spent in each intensity category
        intensity_totals = {"Sedentary": intensity.count(0) / (60 / self.epoch_len),
                            "Sedentary%": round(intensity.count(0) / n_valid_epochs, 3),
                            "Light": intensity.count(1) / (60 / self.epoch_len),
                            "Light%": round(intensity.count(1) / n_valid_epochs, 3),
                            "Moderate": intensity.count(2) / (60 / self.epoch_len),
                            "Moderate%": round(intensity.count(2) / n_valid_epochs, 3),
                            "Vigorous": intensity.count(3) / (60 / self.epoch_len),
                            "Vigorous%": round(intensity.count(3) / n_valid_epochs, 3)
                            }

        print("\n" + "HEART RATE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        return intensity, intensity_totals

    def plot_histogram(self):
        """Generates a histogram of heart rates over the course of the collection with a bin width of 5 bpm.
           Marks calculated average and resting HR."""

        # Data subset: only valid HRs
        valid_heartrates = [i for i in self.valid_hr if i is not None]
        avg_hr = sum(valid_heartrates) / len(valid_heartrates)

        # Bins of width 5bpm between 40 and 180 bpm
        n_bins = np.arange(40, 180, 5)

        plt.figure(figsize=(10, 7))
        plt.hist(valid_heartrates, weights=np.ones(len(valid_heartrates)) / len(valid_heartrates), bins=n_bins,
                 edgecolor='black', color='grey')
        plt.axvline(x=avg_hr, color='red', linestyle='dashed', label="Average HR ({} bpm)".format(round(avg_hr, 1)))
        plt.axvline(x=self.rest_hr, color='green', linestyle='dashed',
                    label='Calculated resting HR ({} bpm)'.format(round(self.rest_hr, 1)))

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.ylabel("% of Epochs")
        plt.xlabel("HR (bpm)")
        plt.title("Heart Rate Histogram")
        plt.legend(loc='upper left')
        plt.show()

    def plot_random_qc(self, input_index=None):
        """Method that generates a random 10-minute sample of data. Overlays filtered data with quality check output.

        :argument
        -start_index: able to input desired start index. If None, randomly generated
        """

        # Generates random start index
        if input_index is not None:
            start_index = input_index
        if input_index is None:
            start_index = randint(0, len(self.filtered) - self.epoch_len * self.sample_rate)

        # Rounds random start to an index that corresponds to start of an epoch
        start_index -= start_index % (self.epoch_len * self.sample_rate)

        print("\n" + "Index {}.".format(start_index))

        # End index: one epoch
        end_index = start_index + self.epoch_len * self.sample_rate

        # Data point index converted to seconds
        seconds_seq_raw = np.arange(0, self.epoch_len * self.sample_rate) / self.sample_rate

        # Epoch's quality check
        validity_data = CheckQuality(ecg_object=self, start_index=start_index, epoch_len=self.epoch_len)

        print()
        print("Valid HR: {} (passed {}/5 conditions)".format(validity_data.rule_check_dict["Valid Period"],
                                                             validity_data.rule_check_dict["HR Valid"] +
                                                             validity_data.rule_check_dict["Max RR Interval Valid"] +
                                                             validity_data.rule_check_dict["RR Ratio Valid"] +
                                                             validity_data.rule_check_dict["Voltage Range Valid"] +
                                                             validity_data.rule_check_dict["Correlation Valid"]))

        print("-HR range ({} bpm): {}".format(validity_data.rule_check_dict["HR"],
                                              validity_data.rule_check_dict["HR Valid"]))
        print("-Max RR interval ({} sec): {}".format(validity_data.rule_check_dict["Max RR Interval"],
                                                     validity_data.rule_check_dict["Max RR Interval Valid"]))
        print("-RR ratio ({}): {}".format(validity_data.rule_check_dict["RR Ratio"],
                                          validity_data.rule_check_dict["RR Ratio Valid"]))
        print("-Voltage range ({} uV): {}".format(validity_data.rule_check_dict["Voltage Range"],
                                                  validity_data.rule_check_dict["Voltage Range Valid"]))
        print("-Correlation (r={}): {}".format(validity_data.rule_check_dict["Correlation"],
                                               validity_data.rule_check_dict["Correlation Valid"]))

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))

        valid_period = "Valid" if validity_data.rule_check_dict["Valid Period"] else "Invalid"

        ax1.set_title("Participant {}: {} (index = {})".format(self.subjectID, valid_period, start_index))

        # Filtered ECG data
        ax1.plot(seconds_seq_raw, self.filtered[start_index:end_index], color='black', label="Filtered ECG")
        ax1.plot(validity_data.r_peaks/self.sample_rate,
                 [self.filtered[start_index+peak] for peak in validity_data.r_peaks],
                 linestyle="", marker="x", color='green')
        ax1.set_ylabel("Voltage")
        ax1.legend(loc='upper left')

        # Filtered ECG data
        ax2.plot(seconds_seq_raw, self.raw[start_index:end_index], color='red', label="Raw ECG")
        ax2.plot(validity_data.r_peaks/self.sample_rate,
                 [self.raw[start_index+peak] for peak in validity_data.r_peaks],
                 linestyle="", marker="x", color='black')
        ax2.set_ylabel("Voltage")
        ax2.legend(loc='upper left')

        ax2.set_xlabel("Seconds")

        # Turns background green if valid
        if valid_period == "Valid":
            ax1.fill_between(x=seconds_seq_raw,
                             y1=min(self.filtered[start_index:end_index]),
                             y2=max(self.filtered[start_index:end_index]), color='green', alpha=0.1)

    def write_output(self):
        """Writes csv of epoched timestamps, validity category."""

        with open(self.output_dir + "Model Output/" + self.filename + "_IntensityData.csv", "w") as outfile:
            writer = csv.writer(outfile, delimiter=',', lineterminator="\n")

            writer.writerow(["Timestamp", "Validity(1=invalid)", "AverageHR", "%HRR", "IntensityCategory"])
            writer.writerows(zip(self.epoch_timestamps, self.epoch_validity, self.epoch_hr,
                                 self.perc_hrr, self.epoch_intensity))

        print("\n" + "Complete. File {} saved.".format(self.output_dir + "Model Output/" +
                                                       self.filename + "_IntensityData.csv"))


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- Quality Check ----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class CheckQuality:
    """Class method that implements the Orphanidou ECG signal quality assessment algorithm on raw ECG data.

       Orphanidou, C. et al. (2015). Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
       Derivation and Applications to Wireless Monitoring. IEEE Journal of Biomedical and Health Informatics.
       19(3). 832-838.
    """

    def __init__(self, ecg_object, start_index, voltage_thresh=250, epoch_len=15):
        """Initialization method.

        :param
        -ecg_object: EcgData class instance created by ImportEDF script
        -random_data: runs algorithm on randomly-generated section of data; False by default.
                      Takes priority over start_index.
        -start_index: index for windowing data; 0 by default
        -epoch_len: window length in seconds over which algorithm is run; 15 seconds by default
        """

        self.voltage_thresh = voltage_thresh
        self.epoch_len = epoch_len
        self.fs = ecg_object.sample_rate
        self.start_index = start_index

        self.raw_data = ecg_object.raw[self.start_index:self.start_index+self.epoch_len*self.fs]
        self.filt_data = ecg_object.filtered[self.start_index:self.start_index+self.epoch_len*self.fs]
        self.index_list = np.arange(0, len(self.raw_data), self.epoch_len*self.fs)

        self.rule_check_dict = {"Valid Period": None,
                                "HR Valid": None, "HR": None,
                                "Max RR Interval Valid": None, "Max RR Interval": None,
                                "RR Ratio Valid": None, "RR Ratio": None,
                                "Voltage Range Valid": None, "Voltage Range": None,
                                "Correlation Valid": None, "Correlation": None}

        # prep_data parameters
        self.r_peaks = None
        self.removed_peak = []
        self.enough_beats = True
        self.hr = 0
        self.delta_rr = []
        self.removal_indexes = []
        self.rr_ratio = None
        self.volt_range = 0

        # apply_rules parameters
        self.valid_hr = None
        self.valid_rr = None
        self.valid_ratio = None
        self.valid_range = None
        self.valid_corr = None
        self.rules_passed = None

        # adaptive_filter parameters
        self.median_rr = None
        self.ecg_windowed = []
        self.average_qrs = None
        self.average_r = 0

        # calculate_correlation parameters
        self.beat_ppmc = []
        self.valid_period = None

        """RUNS METHODS"""
        # Peak detection and basic outcome measures
        self.prep_data()

        # Runs rules check if enough peaks found
        if self.enough_beats:
            self.adaptive_filter()
            self.calculate_correlation()
            self.apply_rules()

    def prep_data(self):
        """Function that:
        -Initializes ecgdetector class instance
        -Runs stationary wavelet transform peak detection
            -Implements 0.1-10Hz bandpass filter
            -DB3 wavelet transformation
            -Pan-Tompkins peak detection thresholding
        -Calculates RR intervals
        -Removes first peak if it is within median RR interval / 2 from start of window
        -Calculates average HR in the window
        -Determines if there are enough beats in the window to indicate a possible valid period
        """

        # Initializes Detectors class instance with sample rate
        detectors = Detectors(self.fs)

        # Runs peak detection on raw data ----------------------------------------------------------------------------
        # Uses ecgdetectors package -> stationary wavelet transformation + Pan-Tompkins peak detection algorithm
        self.r_peaks = detectors.swt_detector(unfiltered_ecg=self.filt_data)

        # Checks to see if there are enough potential peaks to correspond to correct HR range ------------------------
        # Requires number of beats in window that corresponds to ~40 bpm to continue
        # Prevents the math in the self.hr calculation from returning "valid" numbers with too few beats
        # i.e. 3 beats in 3 seconds (HR = 60bpm) but nothing detected for rest of epoch
        if len(self.r_peaks) >= np.floor(40/60*self.epoch_len):
            self.enough_beats = True

            n_beats = len(self.r_peaks)  # number of beats in window
            delta_t = (self.r_peaks[-1] - self.r_peaks[0]) / self.fs  # time between first and last beat, seconds
            self.hr = 60 * (n_beats-1) / delta_t  # average HR, bpm

        # Stops function if not enough peaks found to be a potential valid period
        # Threshold corresponds to number of beats in the window for a HR of 40 bpm
        if len(self.r_peaks) < np.floor(40/60*self.epoch_len):
            self.enough_beats = False
            self.valid_period = False
            return

        # Calculates RR intervals in seconds -------------------------------------------------------------------------
        for peak1, peak2 in zip(self.r_peaks[:], self.r_peaks[1:]):
            rr_interval = (peak2 - peak1) / self.fs
            self.delta_rr.append(rr_interval)

        # Approach 1: median RR characteristics ----------------------------------------------------------------------
        # Calculates median RR-interval in seconds
        median_rr = np.median(self.delta_rr)

        # Converts median_rr to samples
        self.median_rr = int(median_rr * self.fs)

        # Removes any peak too close to start/end of data section: affects windowing later on ------------------------
        # Peak removed if within median_rr/2 samples of start of window
        # Peak removed if within median_rr/2 samples of end of window
        for i, peak in enumerate(self.r_peaks):
            if peak < (self.median_rr/2 + 1) or (self.epoch_len*self.fs - peak) < (self.median_rr/2 + 1):
                self.removed_peak.append(self.r_peaks.pop(i))
                self.removal_indexes.append(i)

        # Removes RR intervals corresponding to
        if len(self.removal_indexes) != 0:
            self.delta_rr = [self.delta_rr[i] for i in range(len(self.r_peaks)) if i not in self.removal_indexes]

        # Calculates range of ECG voltage ----------------------------------------------------------------------------
        self.volt_range = max(self.raw_data) - min(self.raw_data)

    def adaptive_filter(self):
        """Method that runs an adaptive filter that generates the "average" QRS template for the window of data.

        - Calculates the median RR interval
        - Generates a sub-window around each peak, +/- RR interval/2 in width
        - Deletes the final beat sub-window if it is too close to end of data window
        - Calculates the "average" QRS template for the window
        """

        # Approach 1: calculates median RR-interval in seconds  -------------------------------------------------------
        # See previous method

        # Approach 2: takes a window around each detected R-peak of width peak +/- median_rr/2 ------------------------
        for peak in self.r_peaks:
            window = self.filt_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            self.ecg_windowed.append(window)  # Adds window to list of windows

        # Approach 3: determine average QRS template ------------------------------------------------------------------
        self.ecg_windowed = np.asarray(self.ecg_windowed)[1:]  # Converts list to np.array; omits first empty array

        # Calculates "correct" length (samples) for each window (median_rr number of datapoints)
        correct_window_len = 2*int(self.median_rr/2)

        # Removes final beat's window if its peak is less than median_rr/2 samples from end of window
        # Fixes issues when calculating average_qrs waveform
        if len(self.ecg_windowed[-1]) != correct_window_len:
            self.removed_peak.append(self.r_peaks.pop(-1))
            self.ecg_windowed = self.ecg_windowed[:-2]

        # Calculates "average" heartbeat using windows around each peak
        try:
            self.average_qrs = np.mean(self.ecg_windowed, axis=0)
        except ValueError:
            print("Failed to calculate mean QRS template.")

    def calculate_correlation(self):
        """Method that runs a correlation analysis for each beat and the average QRS template.

        - Runs a Pearson correlation between each beat and the QRS template
        - Calculates the average individual beat Pearson correlation value
        - The period is deemed valid if the average correlation is >= 0.66, invalid is < 0.66
        """

        # Calculates correlation between each beat window and the average beat window --------------------------------
        for beat in self.ecg_windowed:
            r = stats.pearsonr(x=beat, y=self.average_qrs)
            self.beat_ppmc.append(abs(r[0]))

        self.average_r = float(np.mean(self.beat_ppmc))
        self.average_r = round(self.average_r, 3)

    def apply_rules(self):
        """First stage of algorithm. Checks data against three rules to determine if the window is potentially valid.
        -Rule 1: HR needs to be between 40 and 180bpm
        -Rule 2: no RR interval can be more than 3 seconds
        -Rule 3: the ratio of the longest to shortest RR interval is less than 2.2
        -Rule 4: the amplitude range of the raw ECG voltage must exceed n microV (approximate range for non-wear)
        -Rule 5: the average correlation coefficient between each beat and the "average" beat must exceed 0.66
        -Verdict: all rules need to be passed
        """

        # Rule 1: "The HR extrapolated from the sample must be between 40 and 180 bpm" -------------------------------
        if 40 <= self.hr <= 180:
            self.valid_hr = True
        else:
            self.valid_hr = False

        # Rule 2: "the maximum acceptable gap between successive R-peaks is 3s ---------------------------------------
        for rr_interval in self.delta_rr:
            if rr_interval < 3:
                self.valid_rr = True

            if rr_interval >= 3:
                self.valid_rr = False
                break

        # Rule 3: "the ratio of the maximum beat-to-beat interval to the minimum beat-to-beat interval... ------------
        # should be less than 2.5"
        self.rr_ratio = max(self.delta_rr) / min(self.delta_rr)

        if self.rr_ratio >= 3:
            self.valid_ratio = False

        if self.rr_ratio < 3:
            self.valid_ratio = True

        # Rule 4: the range of the raw ECG signal needs to be >= 250 microV ------------------------------------------
        if self.volt_range <= self.voltage_thresh:
            self.valid_range = False

        if self.volt_range > self.voltage_thresh:
            self.valid_range = True

        # Rule 5: Determines if average R value is above threshold of 0.66 -------------------------------------------
        if self.average_r >= 0.66:
            self.valid_corr = True

        if self.average_r < 0.66:
            self.valid_corr = False

        # FINAL VERDICT: valid period if all rules are passed --------------------------------------------------------
        if self.valid_hr and self.valid_rr and self.valid_ratio and self.valid_range and self.valid_corr:
            self.valid_period = True
        else:
            self.valid_period = False

        self.rule_check_dict = {"Valid Period": self.valid_period,
                                "HR Valid": self.valid_hr, "HR": round(self.hr, 1),
                                "Max RR Interval Valid": self.valid_rr, "Max RR Interval": round(max(self.delta_rr), 1),
                                "RR Ratio Valid": self.valid_ratio, "RR Ratio": round(self.rr_ratio, 1),
                                "Voltage Range Valid": self.valid_range, "Voltage Range": round(self.volt_range, 1),
                                "Correlation Valid": self.valid_corr, "Correlation": self.average_r}
