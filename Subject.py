import ImportDemographics
import Accelerometer
import ECG
import SyncStarts
import SleepLog
import NonWearLog

import matplotlib.pyplot as plt


# ====================================================================================================================
# ================================================== SUBJECT CLASS ===================================================
# ====================================================================================================================


class Subject:

    def __init__(self, wrist_filepath=None, ankle_filepath=None, ecg_filepath=None,
                 epoch_len=15, remove_epoch_baseline=False,
                 filter_ecg=False,
                 from_processed=True, output_dir=None,
                 write_results=False, treadmill_processed=False, treadmill_log_file=None, demographics_file=None):

        self.wrist_filepath = wrist_filepath
        self.ankle_filepath = ankle_filepath
        self.ecg_filepath = ecg_filepath
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

        self.from_processed = from_processed
        self.output_dir = output_dir
        self.processed_folder = self.output_dir + "Model Output/"

        self.write_results = write_results
        self.treadmill_processed = treadmill_processed

        self.treadmill_log_file = treadmill_log_file
        self.demographics_file = demographics_file

        self.demographics = ImportDemographics.import_demographics(demographics_file=self.demographics_file,
                                                                   subjectID=self.subjectID)

        self.offset_dict = SyncStarts.check_starttimes(subject_object=self)

        # Objects from Accelerometer script
        if self.wrist_filepath is not None:
            self.wrist = Accelerometer.Wrist(filepath=self.wrist_filepath,output_dir=self.output_dir,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             write_results=self.write_results, offset=self.offset_dict["Wrist"])

        if self.ankle_filepath is not None:
            self.ankle = Accelerometer.Ankle(filepath=self.ankle_filepath, output_dir=self.output_dir,
                                             remove_baseline=self.remove_epoch_baseline,
                                             processed_folder=self.processed_folder, from_processed=self.from_processed,
                                             treadmill_processed=True, treadmill_log_file=self.treadmill_log_file,
                                             write_results=self.write_results, offset=self.offset_dict["Ankle"],
                                             age=self.demographics["Age"], rvo2=self.demographics["RestVO2"])

        if self.ecg_filepath is not None:
            self.ecg = ECG.ECG(filepath=self.ecg_filepath, from_processed=self.from_processed, filter=self.filter_ecg,
                               output_dir=self.output_dir, write_results=self.write_results, epoch_len=self.epoch_len,
                               offset=self.offset_dict["ECG"], age=self.demographics["Age"])

    def locate_all_valid_data(self):
        """Method that creates subsets of data that represent only epochs where all devices provided valid data."""

        pass

    def plot_epoched(self):
        """Plots epoched wrist, ankle, and HR data on 3 subplots."""

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(10, 7))
        ax1.set_title("Integrated data: {}".format(self.subjectID))

        # WRIST
        try:
            ax1.plot(self.wrist.epoch.timestamps, self.wrist.epoch.svm, color='black', label="Wrist")
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")
        except AttributeError:
            pass

        # ANKLE
        try:
            ax2.plot(self.ankle.epoch.timestamps, self.ankle.epoch.svm, color='black', label="Ankle")
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Counts")
        except AttributeError:
            pass

        # HEART RATE
        try:
            ax3.plot(self.ecg.epoch_timestamps, self.ecg.valid_hr, color='red', label="HR (valid only)")
            ax3.legend(loc='upper left')
            ax3.set_ylabel("HR (bpm)")
        except AttributeError:
            pass


x = Subject(ankle_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3037_01_GA_LAnkle_Accelerometer.EDF",
            wrist_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3037_01_GA_LWrist_Accelerometer.EDF",
            #ankle_filepath=None,
            #wrist_filepath=None,
            ecg_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3037_01_BF.EDF",
            # ecg_filepath=None,
            output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
            remove_epoch_baseline=True,
            from_processed=True,
            treadmill_processed=True,
            treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Treadmill_Log.csv",
            demographics_file="/Users/kyleweber/Desktop/Data/OND07/Participant Information/Demographics_Data.csv",
            write_results=False)

# TO DO
