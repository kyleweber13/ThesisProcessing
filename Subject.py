import ImportDemographics
import Accelerometer
import ECG
import DeviceSync
import SleepLog
import NonWearLog
import ModelStats

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# ====================================================================================================================
# ================================================== SUBJECT CLASS ===================================================
# ====================================================================================================================


class Subject:

    def __init__(self, wrist_filepath=None, ankle_filepath=None, ecg_filepath=None,
                 epoch_len=15, remove_epoch_baseline=False, rest_hr_window=60,
                 filter_ecg=False, plot_data=False,
                 from_processed=True, load_raw_ecg=False, output_dir=None,
                 write_results=False, treadmill_processed=False, treadmill_log_file=None, demographics_file=None):

        processing_start = datetime.now()

        self.wrist = None
        self.ankle = None
        self.ecg = None

        self.wrist_filepath = wrist_filepath
        self.ankle_filepath = ankle_filepath
        self.ecg_filepath = ecg_filepath
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

        # self.valid_ankle, self.valid_wrist, self.valid_ecg = self.locate_all_valid_data()
        # self.stats = ModelStats.Stats(subject_object=self)

        processing_end = datetime.now()
        print("======================================================================================================")
        print("TOTAL PROCESSING TIME = {} SECONDS.".format(round((processing_end-processing_start).seconds, 1)))
        print("======================================================================================================")

        if self.plot_data:
            self.plot_epoched()

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

        sedentary_minutes = [self.wrist.model.intensity_totals["Sedentary"],
                             self.ankle.model.intensity_totals["Sedentary"],
                             self.ecg.intensity_totals["Sedentary"],
                             0]

        light_minutes = [self.wrist.model.intensity_totals["Light"],
                         self.ankle.model.intensity_totals["Light"],
                         self.ecg.intensity_totals["Light"],
                         0]

        moderate_minutes = [self.wrist.model.intensity_totals["Moderate"],
                            self.ankle.model.intensity_totals["Moderate"],
                            self.ecg.intensity_totals["Moderate"],
                            0]

        vigorous_minutes = [self.wrist.model.intensity_totals["Vigorous"],
                            self.ankle.model.intensity_totals["Vigorous"],
                            self.ecg.intensity_totals["Vigorous"],
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
        plt.ylabel("Minutes")

        # Moderate activity
        plt.subplot(2, 2, 3)
        plt.title("Moderate Activity")
        plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"], moderate_minutes, color='#EA5B19', edgecolor='black')
        plt.ylabel("Minutes")

        # Vigorous activity
        plt.subplot(2, 2, 4)
        plt.title("Vigorous Activity")
        plt.bar(["Wrist", "Ankle", "HR", "HR-Acc"], vigorous_minutes, color='red', edgecolor='black')
        plt.ylabel("Minutes")


x = Subject(ankle_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3037_01_GA_LAnkle_Accelerometer.EDF",
            treadmill_processed=True,
            treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Treadmill_Log.csv",

            # wrist_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3037_01_GA_LWrist_Accelerometer.EDF",

            remove_epoch_baseline=True,

            ecg_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3037_01_BF.EDF",
            rest_hr_window=60,
            load_raw_ecg=True,

            epoch_len=15,
            demographics_file="/Users/kyleweber/Desktop/Data/OND07/Participant Information/Demographics_Data.csv",

            output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
            from_processed=False,

            write_results=False,
            plot_data=False)

# TO DO
