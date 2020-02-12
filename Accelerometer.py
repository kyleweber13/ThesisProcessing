import ImportEDF
import EpochData

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import statistics as stats
from sklearn import linear_model


# ====================================================================================================================
# ================================================ WRIST ACCELEROMETER ===============================================
# ====================================================================================================================


class Wrist:

    def __init__(self, filepath, output_dir, epoch_len=15, start_offset=0, end_offset=0,
                 from_processed=True, processed_folder=None, write_results=False):

        print()
        print("======================================== WRIST ACCELEROMETER ========================================")

        self.filepath = filepath
        self.filename = self.filepath.split("/")[-1].split(".")[0]
        self.output_dir = output_dir

        self.epoch_len = epoch_len
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.from_processed = from_processed
        self.processed_folder = processed_folder
        self.write_results = write_results

        # Loads raw accelerometer data and generates timestamps
        self.raw = ImportEDF.GENEActiv(filepath=self.filepath,
                                       start_offset=self.start_offset, end_offset=self.end_offset,
                                       from_processed=self.from_processed)

        self.epoch = EpochData.EpochAccel(raw_data=self.raw,
                                          from_processed=self.from_processed, processed_folder=processed_folder)

        # Model
        self.model = WristModel(accel_object=self)

        # Write results
        if self.write_results:
            self.write_model()

    def write_model(self):
        """Writes csv of epoched timestamps, counts, and intensity categorization to working directory."""

        with open(self.output_dir + "Model Output/" + self.filename + "_IntensityData.csv", "w") as outfile:
            writer = csv.writer(outfile, delimiter=',', lineterminator="\n")

            writer.writerow(["Timestamp", "ActivityCount", "IntensityCategory"])
            writer.writerows(zip(self.epoch.timestamps, self.epoch.svm, self.model.epoch_intensity))

        print("\n" + "Complete. File {} saved.".format(self.output_dir + "Model Output/" +
                                                       self.filename + "_IntensityData.csv"))


class WristModel:

    def __init__(self, accel_object):

        self.accel_object = accel_object

        self.epoch_intensity = []
        self.intensity_totals = None

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

    def __init__(self, filepath=None, output_dir=None, rvo2=None, age=None, epoch_len=15,
                 start_offset=0, end_offset=0,
                 remove_baseline=False,
                 from_processed=True, treadmill_processed=False, treadmill_log_file=None,
                 processed_folder=None, write_results=False):

        print()
        print("======================================== ANKLE ACCELEROMETER ========================================")

        self.filepath = filepath
        self.filename = self.filepath.split("/")[-1].split(".")[0]
        self.output_dir = output_dir

        self.subjectID = self.filename.split("_")[2]
        self.rvo2 = rvo2
        self.age = age

        self.epoch_len = epoch_len
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.remove_baseline = remove_baseline

        self.from_processed = from_processed
        self.processed_folder = processed_folder
        self.treadmill_processed = treadmill_processed
        self.treadmill_log_file = treadmill_log_file
        self.write_results = write_results

        # Loads raw accelerometer data and generates timestamps
        self.raw = ImportEDF.GENEActiv(filepath=self.filepath,
                                       start_offset=self.start_offset, end_offset=self.end_offset,
                                       from_processed=self.from_processed)

        self.epoch = EpochData.EpochAccel(raw_data=self.raw, epoch_len=self.epoch_len,
                                          remove_baseline=self.remove_baseline,
                                          from_processed=self.from_processed, processed_folder=processed_folder)

        # Create Treadmill object
        self.treadmill = Treadmill(subjectID=self.subjectID, ankle_object=self, tm_log_file=self.treadmill_log_file)

        # Create AnkleModel object
        self.model = AnkleModel(ankle_object=self, treadmill_object=self.treadmill, write_results=self.write_results)


class Treadmill:

    def __init__(self, subjectID, ankle_object, tm_log_file, from_raw=False):
        """Class that stores treadmill protocol information from tracking spreadsheet.
           Imports protocol start time and each walks' speed. Stores this information in a dictionary.
           Calculates the index from the raw data which corresponds to the protocol start time.

        :arguments
        -subjectID: subject ID
        -ankle_object: AnkleAccel class instance
        -tm_logfile: spreadsheet that contains treadmill information (raw; not processed)
        -from_raw: if True, user will have to manually select treadmill walks. If False, indexes from processed data
                   will be read in

        :returns
        -treadmill_dict: information imported from treadmill protocol spreadsheet
        -walk_speeds: list of walk speeds (easier to use than values in treadmill_dict)
        """

        self.subjectID = subjectID
        self.log_file = tm_log_file
        self.epoch_data = ankle_object.epoch.svm
        self.walk_indexes = []
        self.from_raw = from_raw

        # Creates treadmill dictionary and walk speed data from spreadsheet data
        self.treadmill_dict, self.walk_speeds, self.walk_indexes = self.import_log(ankle_object=ankle_object)

        self.avg_walk_counts = self.calculate_average_counts()

        """if self.from_raw:
            # Manually selecting treadmill walks
            if len(self.walk_indexes) != 10:
                self.span = SpanSelector(ax=self.create_plot(accel_object=accel_object), onselect=self.select_walks,
                                         direction='horizontal', useblit=True,
                                         rectprops=dict(alpha=0.5, facecolor='grey'))"""

    def import_log(self, ankle_object):
        """Retrieves treadmill protocol information from spreadsheet for correct subject:
           -Protocol start time, walking speeds in m/s, data index that corresponds to start of protocol"""

        # Reads in relevant treadmill protocol details
        log = np.loadtxt(fname=self.log_file, delimiter=",", dtype="str", usecols=(0, 3, 6, 9, 11, 13, 15, 17,
                                                                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
                         skiprows=1)

        valid_data = False

        for row in log:
            # Only retrieves information for correct subject since all participants in one spreadsheet
            if str(self.subjectID) in row[0]:
                valid_data = True  # Data was found
                date = row[1][0:4] + "/" + str(row[1][4:7]).title() + "/" + row[1][7:] + " " + row[2]
                date_formatted = (datetime.strptime(date, "%Y/%b/%d %H:%M"))

                # Stores data and treadmill speeds (m/s) as dictionary
                treadmill_dict = {"File": row[0], "ProtocolTime": date_formatted,
                                  "StartIndex": "None",
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

                # Retrieves data index that corresponds to 10 minutes before treadmill protocol start
                # Finds correct timestamp and breaks loop
                if self.from_raw:
                    start_index = 0
                    for i, stamp in enumerate(ankle_object.raw.timestamps):
                        if stamp >= treadmill_dict["ProtocolTime"]:
                            start_index = i - 10*60*ankle_object.raw.sample_rate
                            break

                # Start index not needed if reading from processed data
                if not self.from_raw:
                    # start_index = "N/A"
                    start_index = 0

                # Updates value in dictionary
                treadmill_dict.update({"StartIndex": start_index})

        # Sets treadmill_dict, walk_indexes and walk_speeds to empty objects if no treadmill data found in log
        if not valid_data:
            treadmill_dict = {"File": "N/A", "ProtocolTime": "N/A",
                              "StartIndex": "N/A",
                              "60%": "N/A", "80%": "N/A",
                              "100%": "N/A", "120%": "N/A",
                              "140%": "N/A"}
            walk_indexes = []
            walk_speeds = []

            print("\n" + "No processed treadmill data found. Please try again using 'processed_treadmill=False'.")

        return treadmill_dict, walk_speeds, walk_indexes

    def create_plot(self, ankle_object):
        """Creates a plot of epoched data for manual walking bout selection."""

        print("\n" + "Highlight each walk on the graph.")

        # Indexes for start of protocol for raw and epoched data
        raw_start = self.treadmill_dict["StartIndex"]
        epoch_start = int(raw_start / (ankle_object.raw.sample_rate * ankle_object.epoch_len))

        fig = plt.figure(figsize=(11, 7))
        graph = plt.gca()

        plt.plot(np.arange(epoch_start, epoch_start + int((60*40)/ankle_object.epoch_len), 1),
                 ankle_object.epoched.epoch[epoch_start:epoch_start + int((60*40)/ankle_object.epoch_len)],
                 color='black', marker="o", markeredgecolor='black', markerfacecolor='red', markersize=4)

        plt.title('Highlight Individual Walks')
        plt.ylabel("Counts")
        plt.xlabel("Epoch Index")

        return graph

    def select_walks(self, xmin, xmax):
        """Function that is called by SpanSelector to retrieve x-coordinates from graph of treadmill walks."""

        min_index, max_index = np.searchsorted(np.arange(0, len(self.epoch_data)), (xmin, xmax))
        max_index = min(len(self.epoch_data) - 1, max_index)

        highlighted_range = np.arange(0, len(self.epoch_data))[min_index:max_index]

        # Saves x-coordinates to self.walk_indexes
        start = np.c_[highlighted_range][0]
        end = np.c_[highlighted_range][-1]

        self.walk_indexes.append(start[0])
        self.walk_indexes.append(end[0])

        # Adds shaded areas to plot as walks are selected
        plt.fill_betweenx(y=np.arange(0, max(self.epoch_data)), x1=start, x2=end, color='#29D114')

        return start, end

    @staticmethod
    def plot_treadmill_protocol(ankle_object):
        """Plots raw and epoched data during treadmill protocol on subplots."""

        raw_start = ankle_object.treadmill.treadmill_dict["StartIndex"]

        # If StartIndex is N/A...
        try:
            epoch_start = int(raw_start / (ankle_object.raw.sample_rate * ankle_object.epoch_len))
        except TypeError:
            epoch_start = ankle_object.raw.sample_rate * ankle_object.epoch_len

        # X-axis coordinates that correspond to epoch number
        index_list = np.arange(0, 3600 * ankle_object.raw.sample_rate) / \
                     ankle_object.raw.sample_rate / ankle_object.epoch_len

        fig, (ax1, ax2) = plt.subplots(2, sharex="col", figsize=(10, 7))

        ax1.set_title("{}: Treadmill Protocol".format(ankle_object.filename))

        ax1.plot(index_list, ankle_object.raw.x[raw_start:raw_start + ankle_object.raw.sample_rate * 3600],
                 color="black")
        ax1.set_ylabel("G's")

        # Epoched data
        """ax2.plot(index_list[::ankle_object.epoch_len*ankle_object.raw.sample_rate],
                 ankle_object.epoch.svm[epoch_start:epoch_start + int(3600/ankle_object.epoch_len)],
                 color='black', marker="o", markeredgecolor='black', markerfacecolor='red', markersize=4)"""
        ax2.bar(index_list[::ankle_object.epoch_len * ankle_object.raw.sample_rate],
                ankle_object.epoch.svm[epoch_start:epoch_start + int(3600 / ankle_object.epoch_len)],
                width=1.0, edgecolor='black', color='grey', alpha=0.75, align="edge")
        ax2.set_ylabel("Counts")

        plt.show()

    def calculate_average_counts(self):
        """Calculates average counts per epoch from the ankle accelerometer.

        :returns
        -avg_walk_count: name says what it is
        """

        try:
            avg_walk_count = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:self.walk_indexes[index+1]]), 2)
                              for index in np.arange(0, len(self.walk_indexes), 2)]

        except IndexError:
            avg_walk_count = [0, 0, 0, 0, 0]

        return avg_walk_count


class AnkleModel:

    def __init__(self, ankle_object, treadmill_object, write_results=False):
        """Class that stores ankle model data. Performs regression analysis on activity counts vs. gait speed.
        Predicts gait speed and METs from activity counts using ACSM equation that predicts VO2 from gait speed.

        :arguments
        -ankle_object: AnkleAccel class instance
        -treadmill_object: Treadmill class instance
        -output_dir: pathway to folder where data is to be saved
        """

        self.epoch_data = ankle_object.epoch.svm
        self.epoch_len = ankle_object.epoch_len
        self.epoch_scale = 1
        self.epoch_timestamps = ankle_object.epoch.timestamps
        self.subjectID = ankle_object.subjectID
        self.file_id = ankle_object.filepath.split("/")[-1].split(".")[0]
        self.rvo2 = ankle_object.rvo2
        self.tm_object = treadmill_object
        self.walk_indexes = None
        self.write_results = write_results

        self.output_dir = ankle_object.output_dir
        self.anklemodel_outfile = self.output_dir + "Model Output/" + "{}_IntensityData.csv".format(self.file_id)

        # Index multiplier for different epoch lengths since treadmill data processed with 15-second epochs
        self.walk_indexes = self.scale_epoch_indexes()

        # Adds average count data to self.tm_object since it has to be run in a weird order
        self.calculate_average_counts()

        # Values from regression equation
        self.r2 = None
        self.y_int, self.coef, self.threshold_dict = self.calculate_regression()
        self.equation = str(round(self.coef, 5)) + "x + " + str(round(self.y_int, 5))

        # Predicted outcome measures from regression
        self.predicted_speed = self.predict_speed(input_data=self.epoch_data)
        self.predicted_mets = self.predict_mets()
        self.epoch_intensity, self.intensity_totals = self.calculate_intensity()

        if self.write_results:
            self.write_anklemodel()

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

    def calculate_regression(self):
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

        # Prints treadmill protocol statistics
        print("\n" + "Treadmill regression")
        print("-Walk speeds (m/s):", self.tm_object.walk_speeds)
        print("-Walk indexes: ", self.walk_indexes)
        print("-Equation: y = {}x + {}".format(coefficient, y_intercept))
        print("-Rounded equation: y = {}x + {}".format(round(coefficient, 5), round(y_intercept, 5)))
        print("-r^2 = {}".format(round(lm.score(counts, speed), 5)))

        # Calculates count and speed limits for different intensity levels
        light_speed = ((1.5 * self.rvo2 - self.rvo2) / 0.1) / 60  # m/s
        light_counts = (light_speed - y_intercept)/coefficient
        mod_speed = ((3 * self.rvo2 - self.rvo2) / 0.1) / 60
        mod_counts = (mod_speed - y_intercept)/coefficient
        vig_speed = ((6 * self.rvo2 - self.rvo2) / 0.1) / 60
        vig_counts = (vig_speed - y_intercept)/coefficient

        threshold_dict = {"Light speed": light_speed, "Light counts": light_counts,
                          "Moderate speed": mod_speed, "Moderate counts": mod_counts,
                          "Vigorous speed": vig_speed, "Vigorous counts": vig_counts}

        return y_intercept, coefficient, threshold_dict

    def plot_regression(self):
        """Plots measured results and results predicted from regression."""

        plt.figure(figsize=(10, 7))

        # Measured (true) values
        plt.plot(self.tm_object.avg_walk_counts, self.tm_object.walk_speeds, label='Treadmill Protocol',
                 markerfacecolor='white', markeredgecolor='black', color='black', marker="o")

        # Predicted values: count range between min and max svm
        plt.plot(np.arange(min(self.epoch_data), (max(self.epoch_data))),
                 [round(i * self.coef + self.y_int, 3) for i in np.arange(min(self.epoch_data), max(self.epoch_data))],
                 label='Regression line (r^2 = {})'.format(self.r2), color='#1993C5', linestyle='dashed')

        # Fills in regions for different intensities
        plt.fill_between(x=[0, self.threshold_dict["Light counts"]], y1=0, y2=self.threshold_dict["Light speed"],
                         color='grey', alpha=0.5, label="Sedentary")

        plt.fill_between(x=[self.threshold_dict["Light counts"], self.threshold_dict["Moderate counts"]],
                         y1=self.threshold_dict["Light speed"], y2=self.threshold_dict["Moderate speed"],
                         color='green', alpha=0.5, label="Light")

        plt.fill_between(x=[self.threshold_dict["Moderate counts"], self.threshold_dict["Vigorous counts"]],
                         y1=self.threshold_dict["Moderate speed"], y2=self.threshold_dict["Vigorous speed"],
                         color='orange', alpha=0.5, label="Moderate")

        plt.fill_between(x=[self.threshold_dict["Vigorous counts"], max(self.epoch_data)],
                         y1=self.threshold_dict["Vigorous speed"],
                         y2=(max(self.epoch_data) * self.coef + self.y_int),
                         color='red', alpha=0.5, label="Vigorous")

        # Lines on axes
        plt.axhline(y=0, color='black')
        plt.axvline(x=0, color='black')

        plt.xlim(0, max(self.epoch_data))
        plt.ylim(0, max(self.predicted_speed))

        plt.legend(loc='upper left')
        plt.ylabel("Gait speed (m/s)")
        plt.xlabel("Counts")
        plt.title("Participant #{}: Treadmill Protocols - Counts vs. Gait Speed".format(self.subjectID))
        plt.show()

    def predict_speed(self, input_data):
        """Predicts gait speed (m/s) using the regression equation.

        :returns:
        -predicted: list of predicted speeds rounded to 5 decimals
        """

        predicted = [svm * self.coef + self.y_int for svm in input_data]

        return predicted

    def predict_mets(self):
        """Calculates METs using ACSM equation based on predicted gait speed and resting VO2 value.

        :returns
        -mets: list of predicted MET levels
        """

        # Creates a list of predicted speeds where any speed below the regression's y-intercept is set to 0 m/s
        speed_above_yint = []

        for speed in self.predicted_speed:
            if speed >= self.y_int:
                speed_above_yint.append(speed)
            if speed < self.y_int:
                speed_above_yint.append(0)

        # Converts m/s to m/min
        m_min = [i * 60 for i in speed_above_yint]

        # Uses ACSM equation to predict METs from predicted gait speed
        mets = [((self.rvo2 + 0.1 * epoch_speed) / self.rvo2) for epoch_speed in m_min]

        return mets

    def calculate_intensity(self):
        """Calculates intensity category based on MET ranges.
           Sums values to determine total time spent in each category.

        :returns
        -intensity: epoch-by-epoch categorization by intensity. 0=sedentary, 1=light, 2=moderate, 3=vigorous
        -intensity_minutes: total minutes spent at each intensity, dictionary
        """

        # Calculates epoch-by-epoch intensity
        # <1.5 METs = sedentary, 1.5-2.99 METs = light, 3.00-5.99 METs = moderate, >= 6.0 METS = vigorous

        intensity = []

        for met in self.predicted_mets:
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
                            "Vigorous%": round(intensity.count(3) / len(self.epoch_data), 3)
                            }

        print("\n" + "ANKLE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        return intensity, intensity_totals

    def count_to_outcome(self, count):
        """Prints predicted speed, VO2 and METs for a single input count value."""

        speed = count * self.coef + self.y_int
        vo2 = speed * 60 * .1 + self.rvo2
        mets = vo2 / self.rvo2

        print("Speed: {}, VO2: {}, METs: {}".format(round(speed, 5), round(vo2, 3), round(mets, 3)))

    def plot_results(self):
        """Plots predicted speed, predicted METs, and predicted intensity categorization on 3 subplots"""

        print("\n" + "Plotting ankle model data...")

        # X-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col')
        ax1.set_title("Participant #{}: Ankle Model Data".format(self.subjectID))

        # Predicted speed (m/s)
        ax1.plot(self.epoch_timestamps[:len(self.predicted_speed)], self.predicted_speed, color='black')
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

    def write_anklemodel(self):

        # Writes epoch-by-epoch data to .csv
        with open(self.anklemodel_outfile, "w") as output:
            writer = csv.writer(output, delimiter=",", lineterminator="\n")

            writer.writerow(
                ["Timestamp", "ActivityCount", "PredictedSpeed", "PredictedMETs", "IntensityCategory"])
            writer.writerows(zip(self.epoch_timestamps, self.epoch_data,
                                 self.predicted_speed, self.predicted_mets, self.epoch_intensity))
