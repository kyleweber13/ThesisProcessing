import ImportEDF
import EpochData

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
from datetime import datetime
import statistics as stats
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import math


# ====================================================================================================================
# ================================================ WRIST ACCELEROMETER ===============================================
# ====================================================================================================================


class Wrist:

    def __init__(self, subjectID=None, filepath=None, output_dir=None, load_raw=False, accel_only=False,
                 epoch_len=15, start_offset=0, end_offset=0, ecg_object=None,
                 from_processed=True, processed_folder=None, write_results=False):

        print()
        print("======================================== WRIST ACCELEROMETER ========================================")

        self.subjectID = subjectID
        self.filepath = filepath
        self.filename = self.filepath.split("/")[-1].split(".")[0]
        self.output_dir = output_dir

        self.load_raw = load_raw
        self.accel_only = accel_only

        self.epoch_len = epoch_len
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.ecg_obejct = ecg_object

        self.from_processed = from_processed
        self.processed_folder = processed_folder
        self.write_results = write_results

        # Loads raw accelerometer data and generates timestamps
        self.raw = ImportEDF.GENEActiv(filepath=self.filepath,
                                       start_offset=self.start_offset, end_offset=self.end_offset,
                                       load_raw=self.load_raw)

        self.epoch = EpochData.EpochAccel(raw_data=self.raw, accel_only=self.accel_only,
                                          from_processed=self.from_processed, processed_folder=processed_folder)

        # Model
        self.model = WristModel(accel_object=self, ecg_object=self.ecg_obejct)

        # Write results
        if self.write_results:
            self.write_model()

    def write_model(self):
        """Writes csv of epoched timestamps, counts, and intensity categorization to working directory."""

        if not self.accel_only:
            out_filename = self.output_dir + self.filename + "_IntensityData.csv"
        if self.accel_only:
            out_filename = self.output_dir + self.filename + "_IntensityData_AccelOnly.csv"

        with open(out_filename, "w") as outfile:
            writer = csv.writer(outfile, delimiter=',', lineterminator="\n")

            writer.writerow(["Timestamp", "ActivityCount", "IntensityCategory"])
            writer.writerows(zip(self.epoch.timestamps, self.epoch.svm, self.model.epoch_intensity))

        print("\n" + "Complete. File {} saved.".format(out_filename))


class WristModel:

    def __init__(self, accel_object, ecg_object=None):

        self.accel_object = accel_object

        if ecg_object is not None:
            self.valid_ecg = ecg_object.epoch_validity
        if ecg_object is None:
            self.valid_ecg = None

        self.epoch_intensity = []
        self.epoch_intensity_valid = None
        self.intensity_totals = None
        self.intensity_totals_valid = None

        # Calculates intensity based on Powell et al. (2016) cut-points
        self.powell_cutpoints()

    def powell_cutpoints(self):
        """Function that applies Powell et al. (2016) cut-points to epoched accelerometer data. Also calculates
           total minutes spent at each of the 4 intensities in minutes and as a percent of collection.

           :param
           -accel_object: Data class object that contains accelerometer data (epoch)
           """

        print("\n" + "Applying Powell et al. (2016) cut-points to the data...")

        # Conversion factor: epochs to minutes
        epoch_to_minutes = 60 / self.accel_object.epoch_len

        # Sample rate
        sample_rate = self.accel_object.raw.sample_rate

        # Epoch-by-epoch intensity
        for epoch in self.accel_object.epoch.svm:
            if epoch < 47 * sample_rate / 30:
                self.epoch_intensity.append(0)
            if 47 * sample_rate / 30 <= epoch < 64 * sample_rate / 30:
                self.epoch_intensity.append(1)
            if 64 * sample_rate / 30 <= epoch < 157 * sample_rate / 30:
                self.epoch_intensity.append(2)
            if epoch >= 157 * sample_rate / 30:
                self.epoch_intensity.append(3)

        if self.valid_ecg is not None:
            index_list = min([len(self.epoch_intensity), len(self.valid_ecg)])

            self.epoch_intensity_valid = [self.epoch_intensity[i] if self.valid_ecg[i] == 0
                                          else None for i in range(0, index_list)]

        # MODEL TOTALS IF NOT CORRECTED USING VALID ECG EPOCHS -------------------------------------------------------
        # Intensity data: totals
        # In minutes and %
        self.intensity_totals = {"Sedentary": self.epoch_intensity.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.epoch_intensity.count(0) /
                                                     len(self.accel_object.epoch.svm), 3),
                                 "Light": (self.epoch_intensity.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.epoch_intensity.count(1) /
                                                 len(self.accel_object.epoch.svm), 3),
                                 "Moderate": self.epoch_intensity.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.epoch_intensity.count(2) /
                                                    len(self.accel_object.epoch.svm), 3),
                                 "Vigorous": self.epoch_intensity.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.epoch_intensity.count(3) /
                                                    len(self.accel_object.epoch.svm), 3)}

        # MODEL TOTALS IF CORRECTED USING VALID ECG EPOCHS -----------------------------------------------------------
        # Intensity data: totals
        # In minutes and %
        if self.valid_ecg is not None:
            n_valid_epochs = len(self.epoch_intensity_valid) - self.epoch_intensity_valid.count(None)

            if n_valid_epochs == 0:
                n_valid_epochs = len(self.epoch_intensity_valid)

            self.intensity_totals_valid = {"Sedentary": self.epoch_intensity_valid.count(0) / epoch_to_minutes,
                                           "Sedentary%": round(self.epoch_intensity_valid.count(0) / n_valid_epochs, 3),
                                           "Light": (self.epoch_intensity.count(1)) / epoch_to_minutes,
                                           "Light%": round(self.epoch_intensity_valid.count(1) / n_valid_epochs, 3),
                                           "Moderate": self.epoch_intensity.count(2) / epoch_to_minutes,
                                           "Moderate%": round(self.epoch_intensity_valid.count(2) / n_valid_epochs, 3),
                                           "Vigorous": self.epoch_intensity.count(3) / epoch_to_minutes,
                                           "Vigorous%": round(self.epoch_intensity_valid.count(3) / n_valid_epochs, 3)}

        print("Complete.")

        print("\n" + "WRIST MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(self.intensity_totals["Sedentary"],
                                                   round(self.intensity_totals["Sedentary%"]*100, 3)))

        print("Light: {} minutes ({}%)".format(self.intensity_totals["Light"],
                                               round(self.intensity_totals["Light%"]*100, 3)))

        print("Moderate: {} minutes ({}%)".format(self.intensity_totals["Moderate"],
                                                  round(self.intensity_totals["Moderate%"]*100, 3)))

        print("Vigorous: {} minutes ({}%)".format(self.intensity_totals["Vigorous"],
                                                  round(self.intensity_totals["Vigorous%"]*100, 3)))

# ====================================================================================================================
# ================================================ ANKLE ACCELEROMETER ===============================================
# ====================================================================================================================


class Ankle:

    def __init__(self, subjectID=None, filepath=None, load_raw=False, accel_only=False,
                 output_dir=None, rvo2=None, age=None, epoch_len=15,
                 start_offset=0, end_offset=0,
                 remove_baseline=False, ecg_object=None,
                 from_processed=True, treadmill_log_file=None,
                 processed_folder=None, write_results=False):

        print()
        print("======================================== ANKLE ACCELEROMETER ========================================")

        self.subjectID = subjectID
        self.filepath = filepath
        self.filename = self.filepath.split("/")[-1].split(".")[0]
        self.load_raw = load_raw
        self.accel_only = accel_only
        self.output_dir = output_dir

        self.rvo2 = rvo2
        self.age = age

        self.epoch_len = epoch_len
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.remove_baseline = remove_baseline

        self.ecg_object = ecg_object

        self.from_processed = from_processed
        self.processed_folder = processed_folder
        self.treadmill_log_file = treadmill_log_file
        self.treadmill_complete = True
        self.write_results = write_results

        # Loads raw accelerometer data and generates timestamps
        self.raw = ImportEDF.GENEActiv(filepath=self.filepath,
                                       start_offset=self.start_offset, end_offset=self.end_offset,
                                       load_raw=self.load_raw)

        self.epoch = EpochData.EpochAccel(raw_data=self.raw, epoch_len=self.epoch_len,
                                          remove_baseline=self.remove_baseline, accel_only=self.accel_only,
                                          from_processed=self.from_processed, processed_folder=processed_folder)

        if self.treadmill_log_file is None:
            print("\n" + "Need treadmill protocol data to continue. Try again.")

            plt.title("Set treadmill protocol walk indexes on datasheet")
            plt.plot(np.arange(0, len(self.epoch.svm)), self.epoch.svm, color='black')
            plt.ylabel("Counts")
            plt.xlabel("Epoch Index")

        # Create Treadmill object
        self.treadmill = Treadmill(ankle_object=self)

        # Create AnkleModel object
        self.model = AnkleModel(ankle_object=self, write_results=self.write_results,
                                ecg_object=self.ecg_object)

        if self.write_results:
            self.write_model()

    def plot_raw_over_epoch(self, day, downsample=3):
        """Creates a plot of two subplots with vector magnitude (raw) and epoched data."""

        fig, (ax1, ax2) = plt.subplots(2, sharex="col", figsize=(10, 7))

        try:
            indexes = np.arange(self.raw.sample_rate * (day - 1) * 86400, self.raw.sample_rate * day * 86400)

            ax1.plot(indexes[::downsample] / self.raw.sample_rate,
                     self.raw.x[self.raw.sample_rate * (day - 1) * 86400:
                                self.raw.sample_rate * day * 86400:
                                downsample],
                     color='black', label="X-axis ({}Hz)".format(round(self.raw.sample_rate / downsample), 1))

            ax1.legend(loc='upper left')

        except TypeError:
            pass

        ax1.set_ylabel("Vector Magnitude (G)")

        try:

            epoch_start = int((day - 1) * 86400 / self.epoch_len)
            epoch_end = epoch_start + int(86400 / self.epoch_len)

            indexes = np.arange(epoch_start, epoch_end)
            if indexes[-1] > len(self.epoch.svm):
                indexes = np.arange(epoch_start, len(self.epoch.svm))

            ax2.bar(indexes * self.epoch_len, self.epoch.svm[epoch_start:epoch_end],
                    width=15, edgecolor='black', color='grey', alpha=0.75, align="edge")
            ax2.set_ylabel("Counts per {}s".format(self.epoch_len))

            ax2.axhline(y=self.model.linear_dict["Meaningful threshold"], label="33%PrefSpeed", linestyle='dashed',
                        color='red')
            ax2.axhline(y=self.treadmill.avg_walk_counts[2], label="PrefSpeed", linestyle='dashed',
                        color='green')

            ax2.legend(loc='upper left')

        except (TypeError, AttributeError):
            pass

        ax2.set_xlabel("Seconds")
        plt.title("Vector Magnitude and Epoched Data - Day {}".format(day))

    def plot_epoch_hist(self, bin_size=25):
        """Plots histogram of epoched activity counts with adjustable bin size."""

        bin_list = np.arange(0, max(self.epoch.svm), bin_size)

        plt.hist(x=self.epoch.svm, bins=bin_list, weights=np.ones(len(self.epoch.svm)) / len(self.epoch.svm),
                 edgecolor='black', color='grey', label="Histogram (bin={})".format(bin_size))
        plt.xlabel("Counts")
        plt.ylabel("% Total Epochs")
        plt.title("Participant {}: Ankle Activity Count Histogram".format(self.subjectID))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        try:
            plt.axvline(x=self.treadmill.avg_walk_counts[2], linestyle='dashed',
                        color='black', label="33% preferred speed")

            plt.fill_betweenx(x1=0, x2=self.model.linear_dict["Light counts"], y=[0, 1],
                              color='grey', alpha=0.35, label="Sedentary")

            plt.fill_betweenx(x1=self.model.linear_dict["Light counts"], x2=self.model.linear_dict["Moderate counts"],
                              y=[0, 1], color='green', alpha=0.35, label="Light")

            plt.fill_betweenx(x1=self.model.linear_dict["Moderate counts"],
                              x2=self.model.linear_dict["Vigorous counts"],
                              y=[0, 1], color='orange', alpha=0.35, label="Moderate")

            plt.fill_betweenx(x1=self.model.linear_dict["Vigorous counts"], x2=max(self.epoch.svm),
                              y=[0, 1], color='red', alpha=0.35, label="Vigorous")
        except AttributeError:
            pass

        plt.legend(loc='upper right')

    def write_model(self):

        if not self.accel_only:
            out_filename = self.model.anklemodel_outfile
        if self.accel_only:
            out_filename = self.output_dir + self.filename.split(".")[0].split("/")[-1] + \
                           "_IntensityData_AccelOnly.csv"

        # Writes epoch-by-epoch data to .csv
        with open(out_filename, "w") as output:
            writer = csv.writer(output, delimiter=",", lineterminator="\n")

            writer.writerow(["Timestamp", "ActivityCount", "PredictedSpeed", "PredictedMETs", "IntensityCategory"])
            writer.writerows(zip(self.model.epoch_timestamps, self.model.epoch_data,
                                 self.model.linear_speed, self.model.predicted_mets, self.model.epoch_intensity))

        print("\n" + "Complete. File {}".format(out_filename))


class Treadmill:

    def __init__(self, ankle_object):
        """Class that stores treadmill protocol information from tracking spreadsheet.
           Imports protocol start time and each walks' speed. Stores this information in a dictionary.
           Calculates the index from the raw data which corresponds to the protocol start time.

        :returns
        -treadmill_dict: information imported from treadmill protocol spreadsheet
        -walk_speeds: list of walk speeds (easier to use than values in treadmill_dict)
        """

        self.subjectID = ankle_object.subjectID
        self.log_file = ankle_object.treadmill_log_file
        self.epoch_data = ankle_object.epoch.svm
        self.epoch_timestamps = ankle_object.epoch.timestamps
        self.walk_indexes = []

        # Creates treadmill dictionary and walk speed data from spreadsheet data
        self.treadmill_dict, self.walk_speeds, self.walk_indexes = self.import_log()

        self.avg_walk_counts = self.calculate_average_counts()

    def import_log(self):
        """Retrieves treadmill protocol information from spreadsheet for correct subject:
           -Protocol start time, walking speeds in m/s, data index that corresponds to start of protocol"""

        # Reads in relevant treadmill protocol details
        log = np.loadtxt(fname=self.log_file, delimiter=",", dtype="str",
                         usecols=(0, 3, 6, 9, 11, 13, 15, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31), skiprows=1)

        valid_data = False

        for row in log:
            # Only retrieves information for correct subject since all participants in one spreadsheet
            if str(self.subjectID) in row[0]:
                valid_data = True  # Data was found
                date = row[1][0:4] + "/" + str(row[1][4:7]).title() + "/" + row[1][7:] + " " + row[2]
                date_formatted = (datetime.strptime(date, "%Y/%b/%d %H:%M"))

                for i, stamp in enumerate(self.epoch_timestamps):
                    if stamp < date_formatted:
                        epoch_start_index = i

                # Stores data and treadmill speeds (m/s) as dictionary
                treadmill_dict = {"File": row[0], "ProtocolTime": date_formatted,
                                  "StartIndex": epoch_start_index,
                                  "60%": float(row[3]), "80%": float(row[4]),
                                  "100%": float(row[5]), "120%": float(row[6]),
                                  "140%": float(row[7])}

                # Same information as above; easier to access
                walk_speeds = [treadmill_dict["60%"], treadmill_dict["80%"],
                               treadmill_dict["100%"], treadmill_dict["120%"], treadmill_dict["140%"]]

                try:
                    walk_indexes = [int(row[i]) for i in range(8, len(row))]
                    print("\n" + "Previous processed treadmill data found. Skipping processing.")
                except ValueError:
                    walk_indexes = []
                    print("\n" + "No previous treadmill processing found. ")
                    pass

        # Sets treadmill_dict, walk_indexes and walk_speeds to empty objects if no treadmill data found in log
        if not valid_data:
            treadmill_dict = {"File": "N/A", "ProtocolTime": "N/A",
                              "StartIndex": "N/A",
                              "60%": "N/A", "80%": "N/A",
                              "100%": "N/A", "120%": "N/A",
                              "140%": "N/A"}
            walk_indexes = []
            walk_speeds = []

            print("\n" + "No processed treadmill data found. Please try again.")

        return treadmill_dict, walk_speeds, walk_indexes

    def plot_treadmill_protocol(self, ankle_object):
        """Plots raw and epoched data during treadmill protocol on subplots or
           just epoched data if raw not available."""

        # If raw data available ---------------------------------------------------------------------------------------
        if ankle_object.raw.timestamps is not None:

            print("\n" + "Plotting raw and epoched treadmill protocol data.")

            raw_start = ankle_object.treadmill.treadmill_dict["StartIndex"] * \
                        ankle_object.raw.sample_rate * ankle_object.epoch_len

            if type(raw_start) != int:
                # Sets raw start index to 10 minutes prior to start of protocol
                raw_start = (ankle_object.treadmill.walk_indexes[0] - 5 * ankle_object.epoch_len) * \
                            ankle_object.raw.sample_rate * ankle_object.epoch_len
                if raw_start < 0:
                    raw_start = 0

            # If StartIndex is N/A...
            try:
                epoch_start = int(raw_start / (ankle_object.raw.sample_rate * ankle_object.epoch_len))
            except TypeError:
                epoch_start = ankle_object.raw.sample_rate * ankle_object.epoch_len

            # X-axis coordinates that correspond to epoch number
            # One hour = 3600 seconds
            index_list = np.arange(raw_start, raw_start + 4800 * ankle_object.raw.sample_rate) / \
                         ankle_object.raw.sample_rate / ankle_object.epoch_len

            fig, (ax1, ax2) = plt.subplots(2, sharex="col", figsize=(10, 7))

            ax1.set_title("{}: Treadmill Protocol".format(ankle_object.filename))

            # Plots one hour of raw data (3600 seconds)
            ax1.plot(index_list, ankle_object.raw.x[raw_start:raw_start + ankle_object.raw.sample_rate * 4800],
                     color="black")
            ax1.set_ylabel("G's")

            # Epoched data
            ax2.bar(index_list[::ankle_object.epoch_len * ankle_object.raw.sample_rate],
                    ankle_object.epoch.svm[epoch_start:epoch_start + int(4800 / ankle_object.epoch_len)],
                    width=1.0, edgecolor='black', color='grey', alpha=0.75, align="edge")
            ax2.set_ylabel("Counts")

            # Highlights treadmill walks
            for start, stop in zip(self.walk_indexes[::2], self.walk_indexes[1::2]):
                ax2.fill_betweenx(y=(0, max(ankle_object.epoch.svm[self.walk_indexes[0]-10:
                                                                   self.walk_indexes[0] +
                                                                   int(4800 / ankle_object.epoch_len)])),
                                  x1=start, x2=stop, color='green', alpha=0.35)

            plt.show()

        # If raw data not available -----------------------------------------------------------------------------------
        if ankle_object.raw.timestamps is None:

            print("\n" + "Plotting epoched treadmill protocol data. Raw data not available.")

            fig, ax1 = plt.subplots(1, figsize=(10, 7))

            ax1.bar(np.arange(self.walk_indexes[0]-10, self.walk_indexes[0] + int(4800 / ankle_object.epoch_len), 1),
                    ankle_object.epoch.svm[self.walk_indexes[0]-10:
                                           self.walk_indexes[0] + int(4800 / ankle_object.epoch_len)],
                    width=1.0, edgecolor='black', color='grey', alpha=0.75, align="edge")
            ax1.set_ylabel("Counts")
            ax1.set_xlabel("Epoch Number")
            ax1.set_title("Participant {}: Treadmill Protocol - Epoched Data".format(ankle_object.subjectID))

            # Highlights treadmill walks
            for start, stop in zip(self.walk_indexes[::2], self.walk_indexes[1::2]):
                ax1.fill_betweenx(y=(0, max(ankle_object.epoch.svm[self.walk_indexes[0]-10:
                                                                   self.walk_indexes[0] +
                                                                   int(4800 / ankle_object.epoch_len)])),
                                  x1=start, x2=stop, color='green', alpha=0.35)

            plt.show()

    def calculate_average_counts(self):
        """Calculates average counts per epoch from the ankle accelerometer.

        :returns
        -avg_walk_count: I bet you can figure this one out on your own
        """

        try:
            avg_walk_count = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:self.walk_indexes[index+1]]), 2)
                              for index in np.arange(0, len(self.walk_indexes), 2)]

        except IndexError:
            avg_walk_count = [0, 0, 0, 0, 0]

        return avg_walk_count


class AnkleModel:

    def __init__(self, ankle_object, write_results=False, ecg_object=None):
        """Class that stores ankle model data. Performs regression analysis on activity counts vs. gait speed.
        Predicts gait speed and METs from activity counts using ACSM equation that predicts VO2 from gait speed.

        :arguments
        -ankle_object: AnkleAccel class instance
        -treadmill_object: Treadmill class instance
        -output_dir: pathway to folder where data is to be saved
        """

        self.epoch_data = ankle_object.epoch.svm
        self.epoch_len = ankle_object.epoch_len
        self.accel_only = ankle_object.accel_only
        self.epoch_scale = 1
        self.epoch_timestamps = ankle_object.epoch.timestamps
        self.subjectID = ankle_object.subjectID
        self.filepath = ankle_object.filepath
        self.filename = ankle_object.filepath.split("/")[-1].split(".")[0]
        self.rvo2 = ankle_object.rvo2
        self.tm_object = ankle_object.treadmill
        self.walk_indexes = None
        self.write_results = write_results

        if ecg_object is not None:
            self.valid_ecg = ecg_object.epoch_validity
        if ecg_object is None:
            self.valid_ecg = None

        self.output_dir = ankle_object.output_dir

        try:
            # Index multiplier for different epoch lengths since treadmill data processed with 15-second epochs
            self.walk_indexes = self.scale_epoch_indexes()

            # Adds average count data to self.tm_object since it has to be run in a weird order
            self.calculate_average_counts()

            # Values from regression equation
            self.r2 = None

            self.linear_dict, self.linear_speed = self.calculate_linear_regression()
            self.quad_dict, self.quad_speed = self.calculate_quad_regression()
            self.log_dict, self.log_speed = None, None

            self.predicted_mets, self.epoch_intensity, \
                self.intensity_totals = self.calculate_intensity(self.linear_speed)

            # Predicted outcome measures from linear regression
            self.epoch_intensity_valid = None
            self.intensity_totals_valid = None

        except IndexError:
            pass

    def scale_epoch_indexes(self):
        """Scales treadmill walk indexes if epoch length is not 15 seconds. Returns new list."""

        if self.epoch_len != 15:
            self.epoch_scale = int(np.floor(15 / self.epoch_len))

            walk_indexes = [i * self.epoch_scale for i in self.tm_object.walk_indexes]

        if self.epoch_len == 15:
            walk_indexes = self.tm_object.walk_indexes

        return walk_indexes

    def calculate_average_counts(self):
        """Calculates average activity count total for each treadmill walk."""

        self.tm_object.avg_walk_counts = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:
                                                                           self.walk_indexes[index + 1]]), 2)
                                          for index in np.arange(0, 10, 2)]

    def calculate_linear_regression(self):
        """Performs linear regression to predict gait speed from activity counts.
           Calculates predicted speed that would attain 1.5 METs (sedentary -> light).

        :returns
        -y_intercept: y-intercept from gait speed vs. counts regression
        -coefficient: slope from gait speed vs. counts regression
        -threshold_dict: dictionary for predicted speeds and counts for different intensity levels
        """

        # Reshapes data to work with
        counts = np.array(self.tm_object.avg_walk_counts).reshape(-1, 1)
        speed = np.array(self.tm_object.walk_speeds).reshape(-1, 1)  # m/s

        # Linear regression using sklearn
        lm = linear_model.LinearRegression()
        model = lm.fit(counts, speed)
        y_intercept = lm.intercept_[0]
        coefficient = lm.coef_[0][0]
        self.r2 = round(lm.score(counts, speed), 5)

        # SUMMARY METRICS ---------------------------------------------------------------------------------------------

        print("\n" + "Treadmill regression")

        print("-Walk speeds (m/s):", self.tm_object.walk_speeds)
        print("-Walk indexes: ", self.walk_indexes)

        print("\n" + "Linear regression:")
        print("-Equation: y = {}x + {}".format(coefficient, y_intercept))
        print("-Rounded equation: y = {}x + {}".format(round(coefficient, 5), round(y_intercept, 5)))
        print("-r^2 = {}".format(round(lm.score(counts, speed), 5)))

        # Calculates count and speed limits for different intensity levels
        light_speed = ((1.5 * self.rvo2 - self.rvo2) / 0.1) / 60  # m/s

        # TEMP VALUE
        light_counts = round((light_speed - y_intercept)/coefficient, 1)
        mod_speed = ((3 * self.rvo2 - self.rvo2) / 0.1) / 60
        mod_counts = round((mod_speed - y_intercept)/coefficient, 1)
        vig_speed = ((6 * self.rvo2 - self.rvo2) / 0.1) / 60
        vig_counts = round((vig_speed - y_intercept)/coefficient, 1)

        # ESTIMATING SPEED --------------------------------------------------------------------------------------------

        # Predicts speed using linear regression
        linear_predicted_speed = [svm * coefficient + y_intercept for svm in self.epoch_data]

        # Creates a list of predicted speeds where any speed below the sedentary threshold is set to 0 m/s
        above_sed_thresh = []

        # Threshold corresponding to a 5-second walk at preferred speed
        meaningful_threshold = round(self.tm_object.avg_walk_counts[2] / (self.epoch_len / 5), 2)

        # Sets threshold to either meaningful_threshold OR light_counts based on which is greater
        if meaningful_threshold >= light_counts:
            meaningful_threshold = meaningful_threshold
        if meaningful_threshold < light_counts:
            meaningful_threshold = light_counts

        for speed, counts in zip(linear_predicted_speed, self.epoch_data):
            if counts >= meaningful_threshold:
                above_sed_thresh.append(speed)
            if counts < meaningful_threshold:
                above_sed_thresh.append(0)

        linear_reg_dict = {"a": coefficient, "b": y_intercept,  "r2": self.r2,
                           "Light speed": round(light_speed, 3), "Light counts": light_counts,
                           "Moderate speed": round(mod_speed, 3), "Moderate counts": mod_counts,
                           "Vigorous speed": round(vig_speed, 3), "Vigorous counts": vig_counts,
                           "Meaningful threshold": meaningful_threshold}

        return linear_reg_dict, above_sed_thresh

    def counts_to_speed(self, count):

        speed = self.linear_dict["a"] * count + self.linear_dict["b"]

        print("-Predicted speed for {} counts is {} m/s.".format(count, round(speed, 3)))

    def calculate_quad_regression(self):

        # Sets up data in correct formatting
        counts = [i for i in self.tm_object.avg_walk_counts]
        counts.insert(0, 0)
        counts = np.asarray(counts).reshape(-1, 1)

        speed = [i for i in self.tm_object.walk_speeds]
        speed.insert(0, 0)
        speed = np.asarray(speed).reshape(-1, 1)

        # Performs polynomial regression
        poly_reg = PolynomialFeatures(degree=2)
        X_poly = poly_reg.fit_transform(counts)
        pol_reg = linear_model.LinearRegression()
        pol_reg.fit(X_poly, speed)
        r2 = round(pol_reg.score(X_poly, speed), 4)

        # Extracts coefficients
        x2_term = pol_reg.coef_[0][2]
        x_term = pol_reg.coef_[0][1]
        constant_term = pol_reg.coef_[0][0]

        equation = "{}x^2 + {}x + {}".format(x2_term, x_term, constant_term)

        # Predicting speed
        quad_speed = [i ** 2 * x2_term + i * x_term + constant_term for i in self.epoch_data]

        # Determines parabolic vertex and what activity intensity it falls into
        vertex = -x_term / (2 * x2_term)
        vertex_speed = vertex ** 2 * x2_term + vertex * x_term + constant_term

        # Calculates count and speed limits for different intensity levels
        # QUADRATIC EQUATIONS HAVE 2 SOLUTIONS; THESE ARE THE SMALLER VALUES
        light_speed = ((1.5 * self.rvo2 - self.rvo2) / 0.1) / 60  # m/s
        light_counts = round((-x_term + (x_term**2 - 4 * x2_term * -light_speed)**0.5) / (2 * x2_term), 1)
        mod_speed = ((3 * self.rvo2 - self.rvo2) / 0.1) / 60
        mod_counts = round((-x_term + (x_term**2 - 4 * x2_term * -mod_speed)**0.5) / (2 * x2_term), 1)
        vig_speed = ((6 * self.rvo2 - self.rvo2) / 0.1) / 60
        vig_counts = round((-x_term + (x_term ** 2 - 4 * x2_term * -vig_speed)**0.5) / (2 * x2_term), 1)

        if math.isnan(float(vig_counts)):
            print("\n" + "QUADRATIC REGRESSION ERROR: parabola's vertex does not reach the speed that elicits "
                         "vigorous intensity.")
            vig_counts = max(self.epoch_data)

        quad_reg_dict = {"a": x2_term, "b": x_term, "c": constant_term, "r2": r2,
                         "Light speed": light_speed, "Light counts": light_counts,
                         "Moderate speed": mod_speed, "Moderate counts": mod_counts,
                         "Vigorous speed": vig_speed, "Vigorous counts": vig_counts}

        print("\n" + "Quadratic regression:")
        print("-y ={}".format(equation))
        print("-r^2 = {}".format(r2))
        print("-Vertex at point ({} counts, {} m/s)".format(round(vertex, 0), round(vertex_speed, 2)))

        vertex_mets = round((vertex_speed * 60 * 0.1 + self.rvo2) / self.rvo2, 1)
        if vertex_mets < 1.5:
            print("     -Vertex falls into sedentary activity ({} METs).".format(vertex_mets))
        if 1.5 <= vertex_mets < 3.0:
            print("     -Vertex falls into light activity ({} METs).".format(vertex_mets))
        if 3.0 <= vertex_mets < 6:
            print("     -Vertex falls into moderate activity ({} METs).".format(vertex_mets))
        if vertex_mets >= 6:
            print("     -Vertex falls into vigorous activity ({} METs).".format(vertex_mets))

        return quad_reg_dict, quad_speed

    def plot_regression(self, regression_type="linear"):
        """Plots measured results and results predicted from regression."""

        # Variables from each regression type ------------------------------------------------------------------------
        min_value = np.floor(min(self.epoch_data))
        max_value = np.ceil(max(self.epoch_data))

        if regression_type == "linear" or regression_type == "l":
            regression_type = "linear"
            dict = self.linear_dict
            curve_data = [round(i * self.linear_dict["a"] + self.linear_dict["b"], 3)
                          for i in np.arange(0, max_value)]
            predicted_max = max_value * self.linear_dict["a"] + self.linear_dict["b"]

            # Threshold below which counts are considered noise (100% preferred speed / 3)
            meaningful_thresh = self.linear_dict["Meaningful threshold"]

            # Uses regression to calculate speed equivalent at meaningful threshold
            # No physiological meaning if it is derived from meaningful threshold instead of light counts
            light_speed = self.linear_dict["Meaningful threshold"] * self.linear_dict["a"] + self.linear_dict["b"]

            min_value = 0

        if regression_type == "quadratic" or regression_type == "q":
            regression_type = "quadratic"
            dict = self.quad_dict
            curve_data = [self.quad_dict["a"] * i ** 2 + self.quad_dict["b"] * i + self.quad_dict["c"] for i in
                          np.arange(min_value, max_value)]
            predicted_max = self.quad_dict["a"] * max_value ** 2 + self.quad_dict["b"] * max_value + self.quad_dict["c"]
            meaningful_thresh = 0

        if regression_type == "log":
            dict = self.log_dict

        # Plot --------------------------------------------------------------------------------------------------------

        plt.figure(figsize=(10, 7))

        # Measured (true) values
        plt.plot(self.tm_object.avg_walk_counts, self.tm_object.walk_speeds, label='Treadmill Protocol',
                 markerfacecolor='white', markeredgecolor='black', color='black', marker="o")

        # Predicted values: count range between min and max svm
        plt.plot(np.arange(min_value, max_value), curve_data,
                 label='Regression line (r^2 = {})'.format(dict["r2"]), color='#1993C5', linestyle='dashed')

        # Fills in regions for different intensities
        plt.fill_between(x=[0, meaningful_thresh], y1=0, y2=light_speed,
                         color='grey', alpha=0.5, label="Sedentary")

        plt.fill_between(x=[meaningful_thresh, dict["Moderate counts"]],
                         y1=light_speed, y2=dict["Moderate speed"],
                         color='green', alpha=0.5, label="Light")

        plt.fill_between(x=[dict["Moderate counts"], dict["Vigorous counts"]],
                         y1=dict["Moderate speed"], y2=dict["Vigorous speed"],
                         color='orange', alpha=0.5, label="Moderate")

        plt.fill_between(x=[dict["Vigorous counts"], max_value],
                         y1=dict["Vigorous speed"],
                         y2=predicted_max,
                         color='red', alpha=0.5, label="Vigorous")

        # Lines on axes
        plt.axhline(y=0, color='black')
        plt.axvline(x=0, color='black')

        plt.xlim(0, max(self.epoch_data))
        plt.ylim(0, max(self.linear_speed))

        plt.legend(loc='upper left')
        plt.ylabel("Gait speed (m/s)")
        plt.xlabel("Counts")
        plt.title("Participant #{}: Treadmill Protocols - " 
                  "Counts vs. Gait Speed ({} regression)".format(self.subjectID, regression_type))
        plt.show()

    def calculate_intensity(self, predicted_speed):
        """Calculates intensity category based on MET ranges.
           Sums values to determine total time spent in each category.

        :argument
        -predicted_speed: list containing epoch-by-epoch speed prediction from regression output

        :returns
        -intensity: epoch-by-epoch categorization by intensity. 0=sedentary, 1=light, 2=moderate, 3=vigorous
        -intensity_minutes: total minutes spent at each intensity, dictionary
        """

        # Converts m/s to m/min
        m_min = [i * 60 for i in predicted_speed]

        # Uses ACSM equation to predict METs from predicted gait speed
        mets = [((self.rvo2 + 0.1 * epoch_speed) / self.rvo2) for epoch_speed in m_min]

        # Calculates epoch-by-epoch intensity
        # <1.5 METs = sedentary, 1.5-2.99 METs = light, 3.00-5.99 METs = moderate, >= 6.0 METS = vigorous
        intensity = []

        for met in mets:
            if met < 1.5:
                intensity.append(0)
            if 1.5 <= met < 3.0:
                intensity.append(1)
            if 3.0 <= met < 6.0:
                intensity.append(2)
            if met >= 6.0:
                intensity.append(3)

        # Calculates time spent in each intensity category
        intensity_totals = {"Sedentary": intensity.count(0) / (60 / self.epoch_len),
                            "Sedentary%": round(intensity.count(0) / len(self.epoch_data), 3),
                            "Light": intensity.count(1) / (60 / self.epoch_len),
                            "Light%": round(intensity.count(1) / len(self.epoch_data), 3),
                            "Moderate": intensity.count(2) / (60 / self.epoch_len),
                            "Moderate%": round(intensity.count(2) / len(self.epoch_data), 3),
                            "Vigorous": intensity.count(3) / (60 / self.epoch_len),
                            "Vigorous%": round(intensity.count(3) / len(self.epoch_data), 3)}

        print("\n" + "ANKLE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        return mets, intensity, intensity_totals

    def plot_results(self):
        """Plots predicted speed, predicted METs, and predicted intensity categorization on 3 subplots"""

        print("\n" + "Plotting ankle model data...")

        # X-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col')
        ax1.set_title("Participant #{}: Ankle Model Data".format(self.subjectID))

        # Predicted speed (m/s)
        ax1.plot(self.epoch_timestamps[:len(self.linear_speed)], self.linear_speed, color='black')
        ax1.set_ylabel("Predicted Speed (m/s)")

        # Predicted METs
        ax2.plot(self.epoch_timestamps[:len(self.predicted_mets)], self.predicted_mets, color='black')
        ax2.axhline(y=1.5, linestyle='dashed', color='green')
        ax2.axhline(y=3.0, linestyle='dashed', color='orange')
        ax2.axhline(y=6.0, linestyle='dashed', color='red')
        ax2.set_ylabel("METs")

        # Intensity category
        ax3.plot(self.epoch_timestamps[:len(self.epoch_intensity)], self.epoch_intensity, color='black')
        ax3.set_ylabel("Intensity Category")

        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)
