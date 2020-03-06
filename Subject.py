import ImportDemographics
import Accelerometer
import ECG
import DeviceSync
import SleepLog
import NonWearLog
import ModelStats
import FindValidEpochs
import ImportCropIndexes
import ImportEDF

import os
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


class Subject:

    def __init__(self, from_processed, raw_edf_folder=None, subjectID=None,
                 load_wrist=False, load_ankle=False, load_ecg=False,
                 load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                 epoch_len=15, remove_epoch_baseline=False,
                 rest_hr_window=60, n_epochs_rest_hr=10,
                 crop_index_file=None, filter_ecg=False, plot_data=False,
                 output_dir=None,
                 write_results=False, treadmill_log_file=None,
                 demographics_file=None, sleeplog_folder=None):

        print()
        print("========================================= SUBJECT #{} "
              "=============================================".format(subjectID))
        print()

        processing_start = datetime.now()

        self.wrist = None
        self.ankle = None
        self.ecg = None
        self.hracc = None  # Model not yet made

        self.subjectID = subjectID
        self.raw_edf_folder = raw_edf_folder
        self.crop_index_file = crop_index_file
        self.load_wrist = load_wrist
        self.load_ankle = load_ankle
        self.load_ecg = load_ecg
        self.load_raw_ecg = load_raw_ecg
        self.load_raw_ankle = load_raw_ankle
        self.load_raw_wrist = load_raw_wrist

        self.demographics_file = demographics_file
        self.demographics = ImportDemographics.import_demographics(demographics_file=self.demographics_file,
                                                                   subjectID=self.subjectID)

        self.wrist_filepath, self.ankle_filepath, self.ecg_filepath = self.get_raw_filenames()

        self.filter_ecg = filter_ecg

        self.starttime_dict = {"Ankle": None, "Wrist": None, "ECG": None}

        # Sets subject ID
        if self.wrist_filepath is not None:
            self.subjectID = self.wrist_filepath.split("/")[-1].split(".")[0].split("_")[2]
        if self.wrist_filepath is None and self.ankle_filepath is not None:
            self.subjectID = self.ankle_filepath.split("/")[-1].split(".")[0].split("_")[2]
        if self.wrist_filepath is None and self.ankle_filepath is None and self.ecg_filepath is not None:
            self.subjectID = self.ecg_filepath.split(".")[0].split("_")[2]

        self.epoch_len = epoch_len
        self.remove_epoch_baseline = remove_epoch_baseline
        self.rest_hr_window = rest_hr_window
        self.n_epochs_rest_hr = n_epochs_rest_hr

        self.from_processed = from_processed

        self.write_results = write_results

        # Creates timestamp formatting error if read from processed and then write new file. Fix later.
        if self.from_processed:
            self.write_results = False

        self.output_dir = output_dir
        self.processed_folder = self.output_dir + "Model Output/"

        self.treadmill_log_file = treadmill_log_file
        self.sleeplog_folder = sleeplog_folder

        # FILE CROPPING ----------------------------------------------------------------------------------------------
        # Looks for previously-determined crop indexes from .csv
        if self.crop_index_file is not None:
            self.start_offset_dict, self.end_offset_dict, self.crop_indexes_found = \
                ImportCropIndexes.import_crop_indexes(subject=self.subjectID, crop_file=self.crop_index_file)
        if self.crop_index_file is None:
            self.crop_indexes_found = False

        # If no ECG file is loaded, the whole wrist/ankle files are loaded
        if self.ecg_filepath is None and self.ankle_filepath is not None and self.wrist_filepath is not None:
            self.crop_indexes_found = False

        # If no existing crop indexes exist...
        if not self.crop_indexes_found:

            # Only bothers to run this if reading raw data
            if not self.from_processed:
                self.start_offset_dict = DeviceSync.crop_start(subject_object=self)
                self.end_offset_dict = DeviceSync.crop_end(subject_object=self)

            if self.from_processed:
                self.start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
                self.end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

        # DATA IMPORT ------------------------------------------------------------------------------------------------
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
            self.wrist = Accelerometer.Wrist(filepath=self.wrist_filepath, load_raw=self.load_raw_wrist,
                                             output_dir=self.output_dir,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             write_results=self.write_results,
                                             start_offset=self.start_offset_dict["Wrist"],
                                             end_offset=self.end_offset_dict["Wrist"],
                                             ecg_object=self.ecg)

        if self.ankle_filepath is not None:
            self.ankle = Accelerometer.Ankle(filepath=self.ankle_filepath, load_raw=self.load_raw_ankle,
                                             output_dir=self.output_dir,
                                             remove_baseline=self.remove_epoch_baseline,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             treadmill_log_file=self.treadmill_log_file,
                                             write_results=self.write_results,
                                             start_offset=self.start_offset_dict["Ankle"],
                                             end_offset=self.end_offset_dict["Ankle"],
                                             age=self.demographics["Age"], rvo2=self.demographics["RestVO2"],
                                             ecg_object=self.ecg)

        if self.ankle_filepath is None and self.wrist_filepath is None and self.ecg_filepath is None:
            print("No files were imported.")
            quit()

        # Sleep data -------------------------------------------------------------------------------------------------
        self.sleep = SleepLog.SleepLog(subject_object=self, sleeplog_file=self.sleeplog_folder)

        # UPDATING ECG DATA ------------------------------------------------------------------------------------------
        # Adds data to self.ecg since it relies on self.sleep to be complete
        if self.load_ecg:
            print()
            print("========================================= MORE ECG DATA "
                  "=============================================")

            self.ecg.rolling_avg_hr, self.ecg.rest_hr, self.ecg.awake_hr \
                = self.ecg.find_resting_hr(window_size=self.ecg.rest_hr_window, n_windows=self.ecg.n_epochs_rest,
                                           sleep_status=self.sleep.sleep_status)

            self.ecg.perc_hrr = self.ecg.calculate_percent_hrr()
            self.ecg.epoch_intensity, self.ecg.intensity_totals = self.ecg.calculate_intensity()

        # Processing that is only run if more than one device is loaded ----------------------------------------------
        # if self.load_wrist + self.load_ecg + self.load_ankle > 1:
        if self.wrist_filepath is not None and self.ankle_filepath is not None:
            self.valid_all = None
            self.valid_accelonly = FindValidEpochs.AccelOnly(subject_object=self, write_results=self.write_results)

        if self.load_wrist + self.load_ecg + self.load_ankle > 1:
            # Creates subsets of data where only epochs where all data was valid are included
            self.valid_all = FindValidEpochs.AllDevices(subject_object=self, write_results=self.write_results)

            # Runs statistical analysis
            self.stats = ModelStats.Stats(subject_object=self)

        processing_end = datetime.now()

        print()
        print("======================================================================================================")
        print("TOTAL PROCESSING TIME = {} SECONDS.".format(round((processing_end-processing_start).seconds, 1)))
        print("======================================================================================================")

    def get_raw_filenames(self):

        subject_file_list = [i for i in os.listdir(self.raw_edf_folder) if ".EDF" in i and str(self.subjectID) in i]
        dom_hand = self.demographics["Hand"][0]

        wrist_filename = None
        ankle_filename = None
        ecg_filename = None

        if self.load_wrist:
            wrist_filenames = [self.raw_edf_folder + i for i in subject_file_list if "Wrist" in i]

            if len(wrist_filenames) == 2:
                wrist_filename = [i for i in wrist_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_filenames) == 1:
                wrist_filename = wrist_filenames[0]
            if len(wrist_filenames) == 0:
                print("Could not find the correct wrist accelerometer file.")
                wrist_filename = None

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

        return wrist_filename, ankle_filename, ecg_filename

    def plot_epoched(self):
        """Plots epoched wrist, ankle, and HR data on 3 subplots. Data is not removed for invalid periods; all
           available data is plotted."""

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(10, 7))
        ax1.set_title("Multi-Device Data: {}".format(self.subjectID))

        # Timestamp x-axis formatting
        # xfmt = mdates.DateFormatter("%a, %I:%M %p")
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M")
        locator = mdates.HourLocator(byhour=[0, 8, 16], interval=1)

        # WRIST
        try:
            ax1.plot(self.wrist.epoch.timestamps[0:len(self.wrist.epoch.svm)],
                     self.wrist.epoch.svm[0:len(self.wrist.epoch.timestamps)], color='#606060', label="Wrist Acc.")
            ax1.axhline(y=0, color='red', linewidth=1, linestyle='dashed')
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")
        except AttributeError:
            pass

        # ANKLE
        try:
            ax2.plot(self.ankle.epoch.timestamps[0:len(self.ankle.epoch.svm)],
                     self.ankle.epoch.svm[0:len(self.ankle.epoch.svm)], color='#606060', label="Ankle Acc.")
            ax2.axhline(y=0, color='red', linewidth=1, linestyle='dashed')
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

    def plot_total_activity(self, validity_object):
        """Generates barplots of total activity minutes for each model.
        """

        # 0s are placeholders for Hr-Acc model
        sedentary_minutes = [validity_object.wrist_totals["Sedentary"],
                             validity_object.ankle_totals["Sedentary"],
                             validity_object.hr_totals["Sedentary"],
                             0]

        light_minutes = [validity_object.wrist_totals["Light"],
                         validity_object.ankle_totals["Light"],
                         validity_object.hr_totals["Light"],
                         0]

        moderate_minutes = [validity_object.wrist_totals["Moderate"],
                            validity_object.ankle_totals["Moderate"],
                            validity_object.hr_totals["Moderate"],
                            0]

        vigorous_minutes = [validity_object.wrist_totals["Vigorous"],
                            validity_object.ankle_totals["Vigorous"],
                            validity_object.hr_totals["Vigorous"],
                            0]

        plt.subplots(2, 2, figsize=(10, 7))

        # Sedentary activity
        plt.subplot(2, 2, 1)
        plt.title("Sedentary")
        plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"], sedentary_minutes, color='grey', edgecolor='black')
        plt.ylabel("Minutes")

        # Light activity
        plt.subplot(2, 2, 2)
        plt.title("Light Activity")
        plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"], light_minutes, color='green', edgecolor='black')

        # Moderate activity
        plt.subplot(2, 2, 3)
        plt.title("Moderate Activity")
        plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"], moderate_minutes, color='#EA5B19', edgecolor='black')
        plt.ylabel("Minutes")

        # Vigorous activity
        plt.subplot(2, 2, 4)
        plt.title("Vigorous Activity")
        plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"], vigorous_minutes, color='red', edgecolor='black')

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
        """Plots raw ankle and wrist accelerometer data, raw ECG data, and ECG validity data."""

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
