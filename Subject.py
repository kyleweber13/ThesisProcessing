import ImportDemographics
import Accelerometer
import ECG
import DeviceSync
import SleepData
import ModelStats
import ValidData
import ImportCropIndexes
import ImportEDF
import HRAcc
import DailyReports
import Nonwear

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
warnings.filterwarnings("ignore")


# ====================================================================================================================
# ================================================== SUBJECT CLASS ===================================================
# ====================================================================================================================

# Gets Desktop pathway; used as default write directory
try:
    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop/')
except KeyError:
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/')


class Subject:

    def __init__(self, from_processed, raw_edf_folder=None, subjectID=None,
                 load_wrist=False, load_ankle=False, load_ecg=False,
                 load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                 epoch_len=15, remove_epoch_baseline=False,
                 rest_hr_window=60, n_epochs_rest_hr=10, hracc_threshold=30,
                 crop_index_file=None, filter_ecg=False,
                 output_dir=desktop_path, processed_folder=None,
                 write_results=False, treadmill_log_file=None, nonwear_log_file=None, run_zhou=False,
                 demographics_file=None, sleeplog_file=None):

        print()
        print("========================================= SUBJECT #{} "
              "=============================================".format(subjectID))
        print()

        processing_start = datetime.now()

        # Model objects
        self.wrist = None
        self.ankle = None
        self.ecg = None
        self.hr_acc = None

        self.subjectID = subjectID
        self.raw_edf_folder = raw_edf_folder  # Where raw EDF files are stored

        # Processing variables
        self.epoch_len = epoch_len  # Epoch length; seconds
        self.remove_epoch_baseline = remove_epoch_baseline  # Removes lowest value from epoched data to remove bias
        self.rest_hr_window = rest_hr_window  # Number of epochs over which rolling average HR is calculated
        self.n_epochs_rest_hr = n_epochs_rest_hr  # Number of epochs over which resting HR is calculate
        self.hracc_threshold = hracc_threshold  # Threshold for HR-Acc model

        self.from_processed = from_processed  # Whether to import already-processed data
        self.write_results = write_results  # Whether to write results to CSV

        if self.from_processed:
            self.write_results = False

        self.output_dir = output_dir  # Where files are written
        self.processed_folder = processed_folder  # Folder that contains already-processed data files

        self.demographics_file = demographics_file
        self.treadmill_log_file = treadmill_log_file
        self.sleeplog_file = sleeplog_file
        self.nonwear_file = nonwear_log_file
        self.run_zhou = run_zhou

        # Data cropping
        self.starttime_dict = {"Ankle": None, "Wrist": None, "ECG": None}
        self.start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
        self.end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
        self.crop_index_file = crop_index_file  # CSV file of crop indexes
        self.crop_indexes_found = False

        # Which data to load
        self.load_wrist = load_wrist
        self.load_raw_wrist = load_raw_wrist
        self.wrist_filepath = None

        self.load_ankle = load_ankle
        self.load_raw_ankle = load_raw_ankle
        self.ankle_filepath = None

        self.load_ecg = load_ecg
        self.ecg_filepath=None
        self.load_raw_ecg = load_raw_ecg
        self.filter_ecg = filter_ecg

        if not self.load_ecg and (self.load_ankle or self.load_wrist):
            self.accel_only = True
        if self.load_ecg:
            self.accel_only = False

        self.valid_all = None
        self.valid_accelonly = None

        self.demographics = {"Age": 18, "Sex": None, "Weight": 1, "Height": 1, "Hand": "Right", "RestVO2": 3.5}

        self.stats = None
        self.daily_summary = None

        # =============================================== RUNS METHODS ================================================
        self.wrist_filepath, self.wrist_temp_filepath, self.ankle_filepath, self.ecg_filepath = self.get_raw_filepaths()

        ImportDemographics.import_demographics(subject_object=self)

        self.get_raw_filepaths()
        self.crop_files()
        self.create_objects()

        self.sleep = SleepData.Sleep(subject_object=self)

        self.nonwear = Nonwear.NonwearLog(subject_object=self)  # Manually-logged

        if self.run_zhou and self.wrist_temp_filepath is not None and self.wrist_filepath is not None:
            self.nonwear_zhou = Nonwear.ZhouNonwear(subject_object=self)
        if not self.run_zhou or self.wrist_filepath is None or self.wrist_temp_filepath is None:
            print("\nZhou non-wear algorithm is not being run.")
            self.nonwear_zhou = None

        self.update_ecg_with_sleep_data()

        self.create_hracc()

        self.check_validity()

        # Runs stats if multiple devices available
        if self.load_wrist + self.load_ecg + self.load_ankle > 1:
            # Runs statistical analysis
            self.stats = ModelStats.Stats(subject_object=self)

        # Runs daily summary measures
        if self.load_ecg and self.load_wrist:
            self.daily_summary = DailyReports.DailyReport(subject_object=self)

        processing_end = datetime.now()

        print()
        print("======================================================================================================")
        print("TOTAL PROCESSING TIME = {} SECONDS.".format(round((processing_end-processing_start).seconds, 1)))
        print("======================================================================================================")

    def get_raw_filepaths(self):
        """Retrieves filenames associated with current subject."""

        subject_file_list = [i for i in os.listdir(self.raw_edf_folder) if ".EDF" in i and str(self.subjectID) in i]
        dom_hand = self.demographics["Hand"][0]

        wrist_filename = None
        wrist_temperature_filename = None
        ankle_filename = None
        ecg_filename = None

        if self.load_wrist:
            wrist_filenames = [self.raw_edf_folder + i for i in subject_file_list
                               if "Wrist" in i and "Accelerometer" in i]
            wrist_temperature_filenames = [self.raw_edf_folder + i for i in subject_file_list
                                           if "Wrist" in i and "Temperature" in i]

            if len(wrist_filenames) == 2:
                wrist_filename = [i for i in wrist_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_filenames) == 1:
                wrist_filename = wrist_filenames[0]
            if len(wrist_filenames) == 0:
                print("Could not find the correct wrist accelerometer file.")
                wrist_filename = None

            if len(wrist_temperature_filenames) == 2:
                wrist_temperature_filename = [i for i in wrist_temperature_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_temperature_filenames) == 1:
                wrist_temperature_filename = wrist_temperature_filenames[0]
            if len(wrist_temperature_filenames) == 0:
                print("Could not find the correct wrist temperature file.")
                wrist_temperature_filename = None

        if self.load_ankle:
            ankle_filenames = [self.raw_edf_folder + i for i in subject_file_list if "Ankle" in i]

            if len(ankle_filenames) == 2:
                ankle_filename = [i for i in ankle_filenames if dom_hand + "Ankle" not in i][0]
            if len(ankle_filenames) == 1:
                ankle_filename = ankle_filenames[0]
            if len(ankle_filenames) == 0:
                print("Could not find the correct ankle accelerometer file.")
                ankle_filename = None

        if self.load_ecg:
            ecg_filename = [self.raw_edf_folder + i for i in subject_file_list if "BF" in i][0]
            if len([self.raw_edf_folder + i for i in subject_file_list if "BF" in i]) == 0:
                print("Could not find the correct ECG file.")
                ecg_filename = None

        return wrist_filename, wrist_temperature_filename, ankle_filename, ecg_filename

    def crop_files(self):

        if self.from_processed:
            print("Data is being imported from processed. Skipping file crop.")
            return None

        if not self.from_processed:
            # File summaries
            print("Raw EDF file summaries...")
            ImportEDF.check_file(self.ankle_filepath)
            ImportEDF.check_file(self.wrist_filepath)
            ImportEDF.check_file(self.ecg_filepath)

            # If only one device available
            if self.load_ecg + self.load_wrist + self.load_ankle == 1:
                self.start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
                self.end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

            # If ECG and at least one accelerometer are available
            if self.ecg_filepath is not None and (self.wrist_filepath is not None or self.ankle_filepath is not None):

                # Reads in data from file if available
                if self.crop_index_file is not None:
                    self.start_offset_dict, self.end_offset_dict, self.crop_indexes_found = \
                        ImportCropIndexes.import_crop_indexes(subject=self.subjectID, crop_file=self.crop_index_file)

                # Reads data from raw if crop file not available or no data found for participant
                if self.crop_index_file is None or not self.crop_indexes_found:
                    self.start_offset_dict = DeviceSync.crop_start(subject_object=self)
                    self.end_offset_dict = DeviceSync.crop_end(subject_object=self)

            # If ECG not available but wrist and ankle accelerometers are
            if self.ecg_filepath is None and self.wrist_filepath is not None and self.ankle_filepath is not None:

                # Reads from csv if available
                if self.crop_index_file is not None:
                    self.start_offset_dict, self.end_offset_dict, self.crop_indexes_found = \
                        ImportCropIndexes.import_crop_indexes(subject=self.subjectID, crop_file=self.crop_index_file)

                # Reads from raw if participant not found in csv or csv does not exist
                if not self.crop_indexes_found:
                    print("Crop file not entered/found. Ankle treadmill protocol indexes may be incorrect.")
                    self.start_offset_dict = DeviceSync.crop_start(subject_object=self)

                # Overwrites end indexes with values from raw accel files (excludes ECG)
                self.end_offset_dict = DeviceSync.crop_end(subject_object=self)

            # Sets to default values if reading from processed (values not used) if not raw data is read in
            if self.from_processed and not self.load_raw_ecg and not self.load_raw_ankle and not self.load_raw_wrist:
                self.start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
                self.end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

            print("Start indexes: ankle = {}, wrist = {}, ECG = {}".format(self.start_offset_dict["Ankle"],
                                                                           self.start_offset_dict["Wrist"],
                                                                           self.start_offset_dict["ECG"]))
            print("Data points to be read: ankle = {}, wrist = {}, ECG = {}".format(self.end_offset_dict["Ankle"],
                                                                                    self.end_offset_dict["Wrist"],
                                                                                    self.end_offset_dict["ECG"]))

    def create_objects(self):

        if self.ecg_filepath is not None:
            self.ecg = ECG.ECG(filepath=self.ecg_filepath,
                               from_processed=self.from_processed, load_raw=self.load_raw_ecg,
                               filter=self.filter_ecg,
                               output_dir=self.output_dir, write_results=self.write_results, epoch_len=self.epoch_len,
                               start_offset=self.start_offset_dict["ECG"], end_offset=self.end_offset_dict["ECG"],
                               age=self.demographics["Age"],
                               rest_hr_window=self.rest_hr_window, n_epochs_rest=self.n_epochs_rest_hr)

        # Objects from Accelerometer script
        if self.wrist_filepath is not None:
            self.wrist = Accelerometer.Wrist(subjectID=self.subjectID,
                                             filepath=self.wrist_filepath,
                                             temperature_filepath=self.wrist_temp_filepath,
                                             load_raw=self.load_raw_wrist,
                                             output_dir=self.output_dir, accel_only=self.accel_only,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             write_results=self.write_results,
                                             start_offset=self.start_offset_dict["Wrist"],
                                             end_offset=self.end_offset_dict["Wrist"],
                                             ecg_object=self.ecg)

        if self.ankle_filepath is not None:
            self.ankle = Accelerometer.Ankle(subjectID=self.subjectID,
                                             filepath=self.ankle_filepath, load_raw=self.load_raw_ankle,
                                             output_dir=self.output_dir, accel_only=self.accel_only,
                                             remove_baseline=self.remove_epoch_baseline,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             treadmill_log_file=self.treadmill_log_file,
                                             write_results=self.write_results,
                                             start_offset=self.start_offset_dict["Ankle"],
                                             end_offset=self.end_offset_dict["Ankle"],
                                             age=self.demographics["Age"], rvo2=self.demographics["RestVO2"],
                                             bmi=self.demographics["BMI"], ecg_object=self.ecg)

        if self.ankle_filepath is None and self.wrist_filepath is None and self.ecg_filepath is None:
            print("No files were imported.")
            return None

    def update_ecg_with_sleep_data(self):

        # Adds data to self.ecg since it relies on self.sleep to be complete
        if self.load_ecg:
            print()
            print("========================================= MORE ECG DATA "
                  "=============================================")

            self.ecg.rolling_avg_hr, self.ecg.rest_hr, self.ecg.awake_hr \
                = self.ecg.find_resting_hr(window_size=self.ecg.rest_hr_window, n_windows=self.ecg.n_epochs_rest,
                                           sleep_status=self.sleep.status)

            self.ecg.perc_hrr = self.ecg.calculate_percent_hrr()
            self.ecg.epoch_intensity, self.ecg.intensity_totals = self.ecg.calculate_intensity()

    def create_hracc(self):

        if self.ankle_filepath is not None and self.ecg_filepath is not None:
            self.hr_acc = HRAcc.HRAcc(subject_object=self)

    def check_validity(self):

        # Validity check if ECG + at least one accelerometer is available
        if self.wrist_filepath is not None and self.ankle_filepath is not None and self.ecg_filepath is None:
            self.valid_all = None
            self.valid_accelonly = ValidData.AccelOnly(subject_object=self, write_results=self.write_results)

        # Validity check if multiple accelerometers and no ECG is available
        if self.load_wrist + self.load_ecg + self.load_ankle > 1 and self.ecg_filepath is not None:
            # Creates subsets of data where only epochs where all data was valid are included
            self.valid_all = ValidData.AllDevices(subject_object=self, write_results=self.write_results)
            self.valid_accelonly = None

    def plot_sleeplog(self):
        """Plots epoched accelerometer data with shaded regions marking sleep log data.
           Plots ankle/wrist/HR data if available"""

        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(12, 7))

        if self.sleeplog_file is not None:
            ax1.set_title("Subject {}: Sleep Log Data (green = sleep; red = nap)".format(self.subjectID))

        if self.sleeplog_file is None or not os.path.exists(self.sleeplog_file):
            ax1.set_title("Subject {}: No Sleep Log Data".format(self.subjectID))

        # WRIST ACCELEROMETER ----------------------------------------------------------------------------------------
        try:
            ax1.plot(self.wrist.epoch.timestamps[:len(self.wrist.epoch.svm)],
                     self.wrist.epoch.svm[:len(self.wrist.epoch.timestamps)],
                     label='Wrist', color='black')
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")

            for day in self.sleep.data:
                for index, value in enumerate(day):
                    if value != "N/A":
                        if index == 0:
                            ax1.axvline(x=value, color="green", label="Woke up")

                        if index == 3:
                            ax1.axvline(x=value, color="green", label="To bed")

                        if index == 1:
                            ax1.axvline(x=value, color="red", label="Nap")

                        if index == 2:
                            ax1.axvline(x=value, color="red", label="Wake up from nap")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep.data[:], self.sleep.data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, max(self.wrist.epoch.svm)),
                                      color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":
                    # Naps --> red
                    ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, max(self.wrist.epoch.svm)),
                                      color='red', alpha=0.35)

        except (AttributeError, TypeError):
            pass

        # ANKLE ACCELEROMETER ----------------------------------------------------------------------------------------
        try:
            ax2.plot(self.ankle.epoch.timestamps[:len(self.ankle.epoch.svm)],
                     self.ankle.epoch.svm[:len(self.ankle.epoch.timestamps)],
                     label='Ankle', color='black')
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Counts")

            for day in self.sleep.data:
                for index, value in enumerate(day):
                    if value != "N/A":
                        if index == 0:
                            ax2.axvline(x=value, color="green", label="Woke up")

                        if index == 3:
                            ax2.axvline(x=value, color="green", label="To bed")

                        if index == 1:
                            ax2.axvline(x=value, color="red", label="Nap")

                        if index == 2:
                            ax2.axvline(x=value, color="red", label="Wake up from nap")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep.data[:], self.sleep.data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, max(self.ankle.epoch.svm)),
                                      color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":
                    # Naps --> red
                    ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, max(self.ankle.epoch.svm)),
                                      color='red', alpha=0.35)

        except (AttributeError, TypeError):
            pass

        # HEART RATE -------------------------------------------------------------------------------------------------

        try:
            ax3.plot(self.ecg.epoch_timestamps[:len(self.ecg.epoch_hr)],
                     self.ecg.epoch_hr[:len(self.ecg.epoch_timestamps)],
                     label='HR', color='black')
            ax3.legend(loc='upper left')
            ax3.set_ylabel("HR (bpm)")

            for day in self.sleep.data:
                for index, value in enumerate(day):
                    if value != "N/A":
                        if index == 0:
                            ax3.axvline(x=value, color="green", label="Woke up")

                        if index == 3:
                            ax3.axvline(x=value, color="green", label="To bed")

                        if index == 1:
                            ax3.axvline(x=value, color="red", label="Nap")

                        if index == 2:
                            ax3.axvline(x=value, color="red", label="Wake up from nap")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep.data[:], self.sleep.data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax3.fill_betweenx(x1=day1[3], x2=day2[0],
                                      y=np.arange(min([i for i in self.ecg.epoch_hr if i is not None]),
                                                  max([i for i in self.ecg.epoch_hr if i is not None])),
                                      color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":

                    # Naps --> red
                    ax3.fill_betweenx(x1=day1[2], x2=day1[1],
                                      y=np.arange(min([i for i in self.ecg.epoch_hr if i is not None]),
                                                  max([i for i in self.ecg.epoch_hr if i is not None])),
                                      color='red', alpha=0.35)

        except (AttributeError, TypeError, ValueError):
            pass

        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

    def plot_epoched(self):
        """Plots epoched wrist, ankle, and HR data on 3 subplots. Data is not removed for invalid periods; all
           available data is plotted."""

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(10, 7))
        ax1.set_title("Multi-Device Data: {}".format(self.subjectID))

        # Timestamp x-axis formatting
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 8, 16], interval=1)

        # WRIST
        try:
            ax1.plot(self.wrist.epoch.timestamps[0:len(self.wrist.epoch.svm)],
                     self.wrist.epoch.svm[0:len(self.wrist.epoch.timestamps)], color='#606060', label="Wrist Acc.")
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")
        except AttributeError:
            pass

        # ANKLE
        try:
            ax2.plot(self.ankle.epoch.timestamps[0:len(self.ankle.epoch.svm)],
                     self.ankle.epoch.svm[0:len(self.ankle.epoch.svm)], color='#606060', label="Ankle Acc.")
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Counts")
        except AttributeError:
            pass

        # HEART RATE
        try:
            ax3.plot(self.ecg.epoch_timestamps[0:len(self.ecg.valid_hr)],
                     self.ecg.valid_hr[0:len(self.ecg.epoch_timestamps)], color='red',
                     label="HR ({} sec)".format(self.epoch_len))
            ax3.plot(self.ecg.epoch_timestamps[0:len(self.ecg.rolling_avg_hr)],
                     self.ecg.rolling_avg_hr[0:len(self.ecg.epoch_timestamps)], color='black',
                     label="Rolling Average ({} sec)".format(self.rest_hr_window))
            ax3.axhline(y=self.ecg.rest_hr,
                        linestyle='dashed', color='green', label="Resting HR ({} bpm)".format(self.ecg.rest_hr))

            ax3.legend(loc='upper left')
            ax3.set_ylabel("HR (bpm)")
        except AttributeError:
            pass

        # Timestamp x-axis formatting
        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

    def plot_wrist_sleep_nonwear(self):

        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 7))

        plt.suptitle("Subject {}: Wrist - Sleep and Nonwear Data".format(self.subjectID))

        # WRIST ACCELEROMETER ----------------------------------------------------------------------------------------
        try:
            ax1.plot(self.wrist.epoch.timestamps[:len(self.wrist.epoch.svm)],
                     self.wrist.epoch.svm[:len(self.wrist.epoch.timestamps)],
                     label='Wrist', color='black')
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")

            ax2.plot(self.sleep.epoch_timestamps,
                     ["Awake" if i == 0 else "Asleep" for i in self.sleep.status[:len(self.sleep.epoch_timestamps)]],
                     color='green', label="Sleep")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep.data[:], self.sleep.data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax2.fill_between(x=[day1[3], day2[0]], y1="Awake", y2="Asleep", color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":
                    # Naps --> red
                    ax2.fill_between(x=[day1[2], day1[1]], y1="Awake", y2="Asleep", color='red', alpha=0.35)

        except (AttributeError, TypeError):
            pass

        if self.run_zhou and self.wrist_filepath is not None and self.wrist_temp_filepath is not None:

            ax2.plot(self.nonwear_zhou.epoch_timestamps[:len(self.nonwear_zhou.status)],
                     ["Wear " if i == 0 else "Non-Wear " for i in self.nonwear_zhou.status],
                     color='turquoise', label="Zhou Algorithm")
            ax2.fill_between(x=self.nonwear_zhou.epoch_timestamps[:len(self.nonwear_zhou.status)],
                             y1="Wear ", y2=["Wear " if i == 0 else "Non-Wear " for i in self.nonwear_zhou.status],
                             color='turquoise', alpha=0.25)

        if self.nonwear_file is not None:
            ax2.plot(self.nonwear.epoch_timestamps, ["Wear" if i == 0 else "Non-Wear" for i in self.nonwear.status],
                     color='gray', label="Nonwear Log")
            ax2.fill_between(x=self.nonwear.epoch_timestamps, y1="Wear",
                             y2=["Wear" if i == 0 else "Non-Wear" for i in self.nonwear.status],
                             color='gray', alpha=0.5)

            ax2.legend()

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

    def plot_total_activity(self, validity_object):
        """Generates barplots of total activity minutes for each model.
        """

        def autolabel(rects, valid_time):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                plt.annotate('{} mins'.format(round(height / 100 * 60 * valid_time, 1)),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

        # 0s are placeholders for Hr-Acc model
        sedentary_minutes = [validity_object.wrist_totals["Sedentary"],
                             validity_object.ankle_totals["Sedentary"],
                             validity_object.hr_totals["Sedentary"],
                             validity_object.hracc_totals["Sedentary"]]

        light_minutes = [validity_object.wrist_totals["Light"],
                         validity_object.ankle_totals["Light"],
                         validity_object.hr_totals["Light"],
                         validity_object.hracc_totals["Light"]]

        moderate_minutes = [validity_object.wrist_totals["Moderate"],
                            validity_object.ankle_totals["Moderate"],
                            validity_object.hr_totals["Moderate"],
                            validity_object.hracc_totals["Moderate"]]

        vigorous_minutes = [validity_object.wrist_totals["Vigorous"],
                            validity_object.ankle_totals["Vigorous"],
                            validity_object.hr_totals["Vigorous"],
                            validity_object.hracc_totals["Vigorous"]]

        plt.subplots(2, 2, figsize=(10, 7))
        plt.suptitle("Participant {}: Valid Only Data".format(self.subjectID))

        # Sedentary activity
        plt.subplot(2, 2, 1)
        plt.title("Sedentary")
        sed_plot = plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"],
                           [i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                            100 for i in sedentary_minutes],
                           color='grey', edgecolor='black')
        autolabel(sed_plot, validity_object.validity_dict["Total Hours Valid"])
        plt.ylim(0, max([i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                         100 for i in sedentary_minutes]) * 1.2)
        plt.ylabel("% of valid time")

        # Light activity
        plt.subplot(2, 2, 2)
        plt.title("Light Activity")
        light_plot = plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"],
                             [i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                              100 for i in light_minutes],
                             color='green', edgecolor='black')
        plt.ylim(0, max([i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                         100 for i in light_minutes]) * 1.2)
        autolabel(light_plot, validity_object.validity_dict["Total Hours Valid"])

        # Moderate activity
        plt.subplot(2, 2, 3)
        plt.title("Moderate Activity")
        mod_plot = plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"],
                           [i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                            100 for i in moderate_minutes],
                           color='#EA5B19', edgecolor='black')
        plt.ylim(0, max([i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                         100 for i in moderate_minutes]) * 1.2)
        autolabel(mod_plot, validity_object.validity_dict["Total Hours Valid"])
        plt.ylabel("% of valid time")

        # Vigorous activity
        plt.subplot(2, 2, 4)
        plt.title("Vigorous Activity")
        vig_plot = plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"],
                           [i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                            100 for i in vigorous_minutes],
                           color='red', edgecolor='black')
        plt.ylim(0, max([i / 60 / validity_object.validity_dict["Total Hours Valid"] *
                         100 for i in vigorous_minutes]) * 1.2)
        autolabel(vig_plot, validity_object.validity_dict["Total Hours Valid"])

        print()
        print("========================================= TOTAL ACTIVITY SUMMARY =====================================")
        print("Sedentary: wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(sedentary_minutes[0], sedentary_minutes[1],
                                                             sedentary_minutes[2], sedentary_minutes[3]))
        print("Light:     wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(light_minutes[0], light_minutes[1],
                                                             light_minutes[2], light_minutes[3]))
        print("Moderate:  wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(moderate_minutes[0], moderate_minutes[1],
                                                             moderate_minutes[2], moderate_minutes[3]))
        print("Vigorous:  wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(vigorous_minutes[0], vigorous_minutes[1],
                                                             vigorous_minutes[2], vigorous_minutes[3]))

    def plot_accel_ecg_quality(self):
        """Plots raw ankle and wrist accelerometer data, raw ECG data, and ECG validity status."""

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 7))

        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 8, 16], interval=1)

        ax1.set_title("Participant {}: Movement's effect on ECG validity".format(self.subjectID))

        if self.wrist_filepath is not None and self.load_raw_wrist:
            ax1.plot(self.wrist.raw.timestamps[::3], self.wrist.raw.x[::3], color='black',
                     label='Wrist ({}Hz)'.format(int(self.wrist.raw.sample_rate) / 3))
            ax1.set_ylabel("G's")
            ax1.legend(loc='upper left')
            ax1.set_ylim(-8, 8)

        if self.ankle_filepath is not None and self.load_raw_ankle:
            ax2.plot(self.ankle.raw.timestamps[::3], self.ankle.raw.x[::3], color='black',
                     label='Ankle ({}Hz'.format(int(self.ankle.raw.sample_rate) / 3))
            ax2.set_ylabel("G's")
            ax2.legend(loc='upper left')
            ax2.set_ylim(-8, 8)

        if self.ecg_filepath is not None and self.load_raw_ecg:
            ax3.plot(self.ecg.timestamps[::5], self.ecg.filtered[::5], color='red',
                     label='ECG ({}Hz, filtered)'.format(int(self.ecg.sample_rate) / 5))
            ax3.set_ylabel("Voltage")
            ax3.legend(loc='upper left')

        if self.ecg_filepath is not None and self.load_raw_wrist and self.ecg.epoch_validity is not None:
            ax4.plot(self.ecg.epoch_timestamps, self.ecg.epoch_validity, color='black', label="ECG Validity")
            ax4.fill_between(x=self.ecg.epoch_timestamps, y1=0, y2=self.ecg.epoch_validity, color='grey')
            ax4.set_ylabel("1 = invalid")
            ax4.legend(loc='upper left')

        ax4.xaxis.set_major_formatter(xfmt)
        ax4.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

    def plot_treadmill_protocol(self):

        if not self.load_ankle:
            print("No ankle data available.")

        if self.load_ankle:

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col')

            start_index = int(self.ankle.treadmill.walk_indexes[0] - 5 * (60 / self.epoch_len))
            if start_index < 0:
                start_index = 0

            end_index = int(self.ankle.treadmill.walk_indexes[-1] + 5 * (60 / self.epoch_len))

            plt.suptitle("Participant {}: HR and Accel Data during Treadmill Protocol".format(self.subjectID))

            if self.wrist_filepath is not None:
                ax1.plot(self.wrist.epoch.timestamps[start_index:end_index],
                         self.wrist.epoch.svm[start_index:end_index], label="Wrist", color='black')
                ax1.set_ylabel("Counts")
                ax1.legend(loc='upper left')

            ax2.plot(self.ankle.epoch.timestamps[start_index:end_index],
                     self.ankle.epoch.svm[start_index:end_index], label="Ankle", color='black')
            ax2.set_ylabel("Counts")
            ax2.legend(loc='upper left')

            if self.load_ecg:
                ax3.plot(self.ecg.epoch_timestamps[start_index:end_index],
                         self.ecg.epoch_hr[start_index:end_index], label="HR", color='red')
                ax3.set_ylabel("HR (bpm)")

            plt.show()

    def plot_nonwear(self):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
        plt.suptitle("Subject {}: Wrist Non-Wear Data".format(self.subjectID))

        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 8, 16], interval=1)

        if self.load_wrist:
            ax1.plot(self.wrist.epoch.timestamps, self.wrist.epoch.svm, color='black', label='Wrist')
            ax1.set_ylabel("SVM")
            ax1.legend()

        if self.run_zhou and self.wrist_filepath is not None and self.wrist_temp_filepath is not None:
            ax2.plot(self.nonwear_zhou.epoch_timestamps, self.nonwear_zhou.temperature_values,
                     color='red', label='Temperature')
            ax2.set_ylabel("ºC")
            ax2.legend()

            ax3.plot(self.nonwear_zhou.epoch_timestamps[:len(self.nonwear_zhou.status)],
                     ["Wear " if i == 0 else "Non-Wear " for i in self.nonwear_zhou.status],
                     color='turquoise', label="Zhou Algorithm")
            ax3.fill_between(x=self.nonwear_zhou.epoch_timestamps[:len(self.nonwear_zhou.status)],
                             y1="Wear ", y2=["Wear " if i == 0 else "Non-Wear " for i in self.nonwear_zhou.status],
                             color='turquoise', alpha=0.25)

        if self.nonwear_file is not None:
            ax3.plot(self.nonwear.epoch_timestamps, ["Wear" if i == 0 else "Non-Wear" for i in self.nonwear.status],
                     color='gray', label="Nonwear Log")
            ax3.fill_between(x=self.nonwear.epoch_timestamps, y1="Wear",
                             y2=["Wear" if i == 0 else "Non-Wear" for i in self.nonwear.status],
                             color='gray', alpha=0.5)

            ax3.legend()

        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)
