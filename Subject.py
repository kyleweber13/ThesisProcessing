import ImportDemographics
import Accelerometer
import ECG
import DeviceSync
import SleepLog
import NonWearLog
import ModelStats
import FindValidEpochs

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
warnings.filterwarnings("ignore")

# ====================================================================================================================
# ================================================== SUBJECT CLASS ===================================================
# ====================================================================================================================


class Subject:

    def __init__(self, raw_edf_folder=None, subjectID=None,
                 load_wrist=False, load_ankle=False, load_ecg=False,
                 load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                 epoch_len=15, remove_epoch_baseline=False,
                 rest_hr_window=60, n_epochs_rest_hr=10,
                 filter_ecg=False, plot_data=False,
                 from_processed=True, output_dir=None,
                 write_results=False, treadmill_processed=False, treadmill_log_file=None,
                 demographics_file=None, sleeplog_folder=None):

        processing_start = datetime.now()

        self.wrist = None
        self.ankle = None
        self.ecg = None
        self.hracc = None  # Model not yet made

        self.subjectID = subjectID
        self.raw_edf_folder = raw_edf_folder
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
        self.plot_data = plot_data

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
        self.output_dir = output_dir
        self.processed_folder = self.output_dir + "Model Output/"

        self.write_results = write_results
        self.treadmill_processed = treadmill_processed

        self.treadmill_log_file = treadmill_log_file
        self.sleeplog_folder = sleeplog_folder

        if not self.from_processed:
            self.start_offset_dict = DeviceSync.crop_start(subject_object=self)
            self.end_offset_dict = DeviceSync.crop_end(subject_object=self)
        if self.from_processed:
            self.start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
            self.end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

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
                                             treadmill_processed=True, treadmill_log_file=self.treadmill_log_file,
                                             write_results=self.write_results,
                                             start_offset=self.start_offset_dict["Ankle"],
                                             end_offset=self.end_offset_dict["Ankle"],
                                             age=self.demographics["Age"], rvo2=self.demographics["RestVO2"],
                                             ecg_object=self.ecg)

        # Sleep data
        self.sleep = SleepLog.SleepLog(subject_object=self,
                                       sleeplog_file=self.sleeplog_folder)

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

        # Processing that is only run if more than one device is loaded
        if self.load_wrist + self.load_ecg + self.load_ankle > 1:

            # Creates subsets of data where only epochs where all data was valid are included
            self.valid = FindValidEpochs.ValidData(subject_object=self)

            # Runs statistical analysis
            self.stats = ModelStats.Stats(subject_object=self)

        processing_end = datetime.now()

        print()
        print("======================================================================================================")
        print("TOTAL PROCESSING TIME = {} SECONDS.".format(round((processing_end-processing_start).seconds, 1)))
        print("======================================================================================================")

        if self.plot_data:
            self.plot_epoched()
            self.valid.plot_validity_data()

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

        if self.load_ankle:
            ankle_filenames = [self.raw_edf_folder + i for i in subject_file_list if "Ankle" in i]

            if len(ankle_filenames) == 2:
                ankle_filename = [i for i in ankle_filenames if dom_hand + "Ankle" not in i][0]
            if len(ankle_filenames) == 1:
                ankle_filename = ankle_filenames[0]

        if self.load_ecg:
            ecg_filename = [self.raw_edf_folder + i for i in subject_file_list if "BF" in i][0]

        return wrist_filename, ankle_filename, ecg_filename

    def plot_epoched(self):
        """Plots epoched wrist, ankle, and HR data on 3 subplots."""

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
            ax1.axhline(y=0, color='black', linewidth=1)
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")
        except AttributeError:
            pass

        # ANKLE
        try:
            ax2.plot(self.ankle.epoch.timestamps[0:len(self.ankle.epoch.svm)],
                     self.ankle.epoch.svm[0:len(self.ankle.epoch.svm)], color='#606060', label="Ankle Acc.")
            ax2.axhline(y=0, color='black', linewidth=1)
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

    def plot_total_activity(self):
        """Generates barplots of total activity minutes for each model.
        """

        sedentary_minutes = [self.valid.wrist_totals["Sedentary"],
                             self.valid.ankle_totals["Sedentary"],
                             self.valid.hr_totals["Sedentary"],
                             0]

        light_minutes = [self.valid.wrist_totals["Light"],
                         self.valid.ankle_totals["Light"],
                         self.valid.hr_totals["Light"],
                         0]

        moderate_minutes = [self.valid.wrist_totals["Moderate"],
                            self.valid.ankle_totals["Moderate"],
                            self.valid.hr_totals["Moderate"],
                            0]

        vigorous_minutes = [self.valid.wrist_totals["Vigorous"],
                            self.valid.ankle_totals["Vigorous"],
                            self.valid.hr_totals["Vigorous"],
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


x = Subject(raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
            subjectID=3036,
            load_ecg=False, load_ankle=False, load_wrist=True,
            load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=True,
            from_processed=False,

            treadmill_processed=True,

            rest_hr_window=30,
            n_epochs_rest_hr=30,
            filter_ecg=True,
            epoch_len=5,

            treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Treadmill_Log.csv",
            demographics_file="/Users/kyleweber/Desktop/Data/OND07/Participant Information/Demographics_Data.csv",
            # sleeplog_folder="/Users/kyleweber/Desktop/Data/OND07/Sleep Logs/",
            output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",

            write_results=False,
            plot_data=False)


"""output_dict = {"Valid ECG %": 100 - x.ecg.quality_report["Percent invalid"],
               "ECG Hours Lost": x.ecg.quality_report["Hours lost"],
               "Sleep %": x.sleep.sleep_report["Sleep%"],
               "Sleep Hours Lost": x.sleep.sleep_report["SleepDuration"]/60,
               "Total Valid %": x.valid.percent_valid}

with open("/Users/kyleweber/Desktop/QC Data/" + x.subjectID + "_ValidityData.csv", "w") as outfile:

    fieldnames = ['Valid ECG %', 'ECG Hours Lost', 'Sleep %', 'Sleep Hours Lost', 'Total Valid %']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow(output_dict)"""


"""print("PERCENT VALID: ", 100 - x.ecg.quality_report["Percent invalid"])
x.ecg.plot_random_qc()"""
