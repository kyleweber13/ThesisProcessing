import ImportDemographics
import Accelerometer
import ECG
import DeviceSync
import SleepLog
import NonWearLog
import ModelStats
import FindValidEpochs

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# ====================================================================================================================
# ================================================== SUBJECT CLASS ===================================================
# ====================================================================================================================


class Subject:

    def __init__(self, raw_edf_folder=None, subjectID=None,
                 load_wrist=False, load_ankle=False, load_ecg=False,
                 wrist_filepath=None, ankle_filepath=None, ecg_filepath=None,
                 epoch_len=15, remove_epoch_baseline=False, rest_hr_window=60,
                 filter_ecg=False, plot_data=False,
                 from_processed=True, load_raw_ecg=False, output_dir=None,
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
        self.wrist_filepath, self.ankle_filepath, self.ecg_filepath = self.get_raw_filenames()
        print(self.wrist_filepath)
        print(self.ankle_filepath)
        print(self.ecg_filepath)

        # self.wrist_filepath = wrist_filepath
        # self.ankle_filepath = ankle_filepath
        # self.ecg_filepath = ecg_filepath
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

        self.from_processed = from_processed
        self.load_raw_ecg = load_raw_ecg
        self.output_dir = output_dir
        self.processed_folder = self.output_dir + "Model Output/"

        self.write_results = write_results
        self.treadmill_processed = treadmill_processed

        self.treadmill_log_file = treadmill_log_file
        self.demographics_file = demographics_file
        self.sleeplog_folder = sleeplog_folder

        self.demographics = ImportDemographics.import_demographics(demographics_file=self.demographics_file,
                                                                   subjectID=self.subjectID)

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
                               rest_hr_window=self.rest_hr_window)

        # Objects from Accelerometer script
        if self.wrist_filepath is not None:
            self.wrist = Accelerometer.Wrist(filepath=self.wrist_filepath,output_dir=self.output_dir,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             write_results=self.write_results,
                                             start_offset=self.start_offset_dict["Wrist"],
                                             end_offset=self.end_offset_dict["Wrist"],
                                             ecg_object=self.ecg)

        if self.ankle_filepath is not None:
            self.ankle = Accelerometer.Ankle(filepath=self.ankle_filepath, output_dir=self.output_dir,
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

        # Creates subsets of data where only epochs where all data was valid are included
        if self.load_wrist + self.load_ecg + self.load_ankle > 1:
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

        file_list = [i for i in os.listdir(self.raw_edf_folder) if ".EDF" in i and str(self.subjectID) in i]

        wrist_filename = None
        ankle_filename = None
        ecg_filename = None

        if self.load_wrist:
            wrist_filename = [self.raw_edf_folder + i for i in file_list if "Wrist" in i][0]

        if self.load_ankle:
            ankle_filename = [self.raw_edf_folder + i for i in file_list if "Ankle" in i][0]

        if self.load_ecg:
            ecg_filename = [self.raw_edf_folder + i for i in file_list if "BF" in i][0]

        return wrist_filename, ankle_filename, ecg_filename

    def plot_epoched(self):
        """Plots epoched wrist, ankle, and HR data on 3 subplots."""

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(10, 7))
        ax1.set_title("Multi-Device Data: {}".format(self.subjectID))

        # Timestamp x-axis formatting
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

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

            ax3.axvline(x=self.ecg.epoch_timestamps[np.argwhere(np.asarray(self.ecg.rolling_avg_hr)
                                                                == self.ecg.rest_hr)[0][0]],
                        color='#289CE1', linestyle='dashed',
                        label="Rest HR ({} bpm)".format(round(self.ecg.rest_hr, 1)))

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
        print("Light:             wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(light_minutes[0], light_minutes[1],
                                                             light_minutes[2], light_minutes[3]))
        print("Moderate: wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(moderate_minutes[0], moderate_minutes[1],
                                                             moderate_minutes[2], moderate_minutes[3]))
        print("Vigorous: wrist = {} minutes, ankle = {} minutes, "
              "HR = {} minutes, HR-Acc = {} minutes.".format(vigorous_minutes[0], vigorous_minutes[1],
                                                             vigorous_minutes[2], vigorous_minutes[3]))


x = Subject(raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
            subjectID=3036,
            load_ecg=False, load_ankle=True, load_wrist=False,

            treadmill_processed=True,
            treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Treadmill_Log.csv",

            remove_epoch_baseline=True,

            rest_hr_window=60,
            load_raw_ecg=False,

            epoch_len=15,
            demographics_file="/Users/kyleweber/Desktop/Data/OND07/Participant Information/Demographics_Data.csv",
            sleeplog_folder="/Users/kyleweber/Desktop/Data/OND07/Sleep Logs/",

            output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
            from_processed=False,

            write_results=False,
            plot_data=False)

# More dict writers for results
# Set threshold based on walking bout duration
