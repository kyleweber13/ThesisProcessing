from datetime import datetime
import numpy as np


class EpochAccel:

    def __init__(self, raw_data=None, remove_baseline=False, from_processed=True, processed_folder=None, epoch_len=15):

        self.epoch_len = epoch_len
        self.remove_baseline = remove_baseline
        self.from_processed = from_processed
        self.processed_folder = processed_folder
        self.raw_filename = raw_data.filepath.split("/")[-1].split(".")[0]
        self.processed_file = self.processed_folder + self.raw_filename + "_IntensityData.csv"

        self.svm = []
        self.timestamps = None

        # GENEActiv: ankle only
        self.pred_speed = None
        self.pred_mets = None
        self.intensity_cat = None

        # Epoching from raw data
        if not self.from_processed and raw_data is not None:
            self.epoch_from_raw(raw_data=raw_data)

        # Loads epoched data from existing file
        if self.from_processed:
            self.epoch_from_processed()

        # Removes bias from SVM by subtracting minimum value
        if self.remove_baseline and min(self.svm) != 0.0:
            print("\n" + "Removing bias from SVM calculations...")
            self.svm = [i - min(self.svm) for i in self.svm]
            print("Complete. Bias removed.")

    def epoch_from_raw(self, raw_data):

        # Calculates epochs if from_processed is False
        print("\n" + "Epoching using raw data...")

        self.timestamps = raw_data.timestamps[::self.epoch_len * raw_data.sample_rate]

        # Calculates activity counts
        for i in range(0, len(raw_data.vm), int(raw_data.sample_rate * self.epoch_len)):

            if i + self.epoch_len * raw_data.sample_rate > len(raw_data.vm):
                break

            vm_sum = sum(raw_data.vm[i:i + self.epoch_len * raw_data.sample_rate])
            self.svm.append(round(vm_sum, 5))

        print("Epoching complete.")

    def epoch_from_processed(self):

        print("\n" + "Importing data processed from {}.".format(self.processed_folder))

        # Data import from .csv
        if "Wrist" in self.processed_file:
            epoch_timestamps, svm = np.loadtxt(fname=self.processed_file, delimiter=",", skiprows=1,
                                               usecols=(0, 1), unpack=True, dtype="str")

            self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in epoch_timestamps]
            self.epoch_len = (self.timestamps[1] - self.timestamps[0]).seconds
            self.svm = [float(i) for i in svm]

        if "Ankle" in self.processed_file:
            epoch_timestamps, svm, pred_speed, \
            pred_mets, epoch_intensity = np.loadtxt(fname=self.processed_file, delimiter=",", skiprows=1,
                                                    usecols=(0, 1, 2, 3, 4), unpack=True, dtype="str")

            self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in epoch_timestamps]
            self.epoch_len = (self.timestamps[1] - self.timestamps[0]).seconds
            self.svm = [float(i) for i in svm]

            self.pred_mets = [float(i) for i in pred_mets]
            self.pred_speed = [float(i) for i in pred_speed]
            self.intensity_cat = [int(i) for i in epoch_intensity]

        print("Complete.")
