import numpy as np
from datetime import datetime
import statistics as stats
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import matplotlib.dates as mdates
import csv
import os


class Treadmill:

    def __init__(self, subjectID, accel_object, tm_log_file, from_raw=False):
        """Class that stores treadmill protocol information from tracking spreadsheet.
           Imports protocol start time and each walks' speed. Stores this information in a dictionary.
           Calculates the index from the raw data which corresponds to the protocol start time.

        :arguments
        -subjectID: subject ID
        -accel_object: AnkleAccel class instance
        -tm_logfile: spreadsheet that contains treadmill information (raw; not processed)
        -from_raw: if True, user will have to manually select treadmill walks. If False, indexes from processed data
                   will be read in

        :returns
        -treadmill_dict: information imported from treadmill protocol spreadsheet
        -walk_speeds: list of walk speeds (easier to use than values in treadmill_dict)
        """

        self.subjectID = subjectID
        self.file = tm_log_file
        self.epoch_data = accel_object.epoched.epoch
        self.walk_indexes = []
        self.from_raw = from_raw

        # Creates treadmill dictionary and walk speed data from spreadsheet data
        self.treadmill_dict, self.walk_speeds, self.walk_indexes = self.import_log(accel_object=accel_object)

        if self.from_raw:
            # Manually selecting treadmill walks
            if len(self.walk_indexes) != 10:
                self.span = SpanSelector(ax=self.create_plot(accel_object=accel_object), onselect=self.select_walks,
                                         direction='horizontal', useblit=True,
                                         rectprops=dict(alpha=0.5, facecolor='grey'))

    def import_log(self, accel_object):
        """Retrieves treadmill protocol information from spreadsheet for correct subject:
           -Protocol start time, walking speeds in m/s, data index that corresponds to start of protocol"""

        # Reads in relevant treadmill protocol details
        log = np.loadtxt(fname=self.file, delimiter=",", dtype="str", usecols=(0, 3, 6, 9, 11, 13, 15, 17,
                                                                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
                         skiprows=1)

        for row in log:
            # Only retrieves information for correct subject since all participants in one spreadsheet
            if str(self.subjectID) in row[0]:
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
                # Assumes 75Hz sample rate
                if self.from_raw:
                    start_index = 0
                    for i, stamp in enumerate(accel_object.raw.timestamps):
                        if stamp >= treadmill_dict["ProtocolTime"]:
                            start_index = i - 10*60*75
                            break

                # Start index not needed if reading from processed data
                if not self.from_raw:
                    start_index = "N/A"

                # Updates value in dictionary
                treadmill_dict.update({"StartIndex": start_index})

        return treadmill_dict, walk_speeds, walk_indexes

    def create_plot(self, accel_object):
        """Creates a plot of epoched data for manual walking bout selection."""

        print("\n" + "Highlight each walk on the graph.")

        # Indexes for start of protocol for raw and epoched data
        raw_start = self.treadmill_dict["StartIndex"]
        epoch_start = int(raw_start / (75 * accel_object.epoch_len))

        fig = plt.figure(figsize=(11, 7))
        graph = plt.gca()

        plt.plot(np.arange(epoch_start, epoch_start + int((60*40)/accel_object.epoch_len), 1),
                 accel_object.epoched.epoch[epoch_start:epoch_start + int((60*40)/accel_object.epoch_len)],
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
    def plot_treadmill_protocol(accel_object):
        """Plots raw and epoched data during treadmill protocol on subplots. DOES NOT WORK YET."""

        raw_start = accel_object.tm.treadmill_dict["StartIndex"]
        epoch_start = int(raw_start / (75 * accel_object.epoch_len))

        # X-axis coordinates that correspond to
        # Sample rate is 75Hz by default; no check for this at present
        index_list = np.arange(0, 3600 * 75) / 75 / accel_object.epoch_len

        fig, (ax1, ax2) = plt.subplots(2, sharex="col")

        ax1.set_title("{}: Treadmill Protocol".format(accel_object.accel_file))

        ax1.plot(index_list, accel_object.raw.x[raw_start:raw_start + 75 * 3600],
                 color="black")
        ax1.set_ylabel("G's")

        ax2.plot(index_list[::accel_object.epoch_len*75],
                 accel_object.epoched.epoch[epoch_start:epoch_start + int(3600/accel_object.epoch_len)],
                 color='black', marker="o", markeredgecolor='black', markerfacecolor='red')
        ax2.set_ylabel("Counts")

        plt.show()

    def calculate_average_counts(self):
        """Calculates average counts per epoch from the ankle accelerometer.

        :returns
        -avg_walk_count: name says what it is
        """

        avg_walk_count = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:self.walk_indexes[index+1]]), 2)
                          for index in np.arange(0, len(self.walk_indexes), 2)]

        return avg_walk_count


class Model:

    def __init__(self, accel_object, tm_object, out_folder,
                 write_results=False, overwrite_existing=False):
        """Class that stores ankle model data. Performs regression analysis on activity counts vs. gait speed.
        Predicts gait speed and METs from activity counts using ACSM equation that predicts VO2 from gait speed.

        :arguments
        -accel_object: AnkleAccel class instance
        -tm_object: Treadmill class instance
        -out_folder: pathway to folder where data is to be saved
        -write_results: writes epoch timestamps, counts, predicted gait speed, predicted METs, and intensity category
                        if True. False by default.
        """

        self.epoch_data = accel_object.epoched.epoch
        self.epoch_len = accel_object.epoch_len
        self.epoch_scale = 1
        self.epoch_timestamps = accel_object.epoched.epoch_timestamps
        self.subjectID = accel_object.accel_file.split(".")[0]
        self.rvo2 = accel_object.rvo2
        self.tm_object = tm_object
        self.walk_indexes = None
        self.write_results = write_results
        self.overwrite_existing = overwrite_existing
        self.working_dir = accel_object.working_dir
        self.modeloutput_folder = accel_object.modeloutput_folder
        self.anklemodel_outfile = out_folder + "{}_IntensityData.csv".format(self.subjectID)

        # Index multiplier for different epoch lengths since treadmill data processed with 15-second epochs
        self.walk_indexes = self.scale_epoch_indexes()

        # Adds average count data to self.tm_object since it has to be run in a weird order
        self.calculate_average_counts()

        # Values from regression equation
        self.r2 = None
        self.y_int, self.coef, self.pred_light_speed = self.calculate_regression()
        self.equation = str(round(self.coef, 5)) + "x + " + str(round(self.y_int, 5))

        # Predicted outcome measures from regression
        self.predicted_speed = self.pred_speed(input_data=self.epoch_data)
        self.predicted_mets = self.pred_mets()
        self.epoch_intensity, self.intensity_minutes = self.calculate_intensity()

        if self.write_results:
            self.write_anklemodel()

    def scale_epoch_indexes(self):
        """Scales treadmill walk indexes if epoch length is not 15 seconds. Returns new list."""

        if self.epoch_len != 15:
            self.epoch_scale = int(np.floor(15 / self.epoch_len))

            walk_indexes = [i*self.epoch_scale for i in self.tm_object.walk_indexes]

        if self.epoch_len == 15:
            walk_indexes = self.tm_object.walk_indexes

        return walk_indexes

    def calculate_average_counts(self):
        """Calculates average count total for each treadmill walk."""

        self.tm_object.avg_walk_counts = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:
                                                                           self.walk_indexes[index + 1]]), 2)
                                          for index in np.arange(0, 10, 2)]

    def calculate_regression(self):
        """Performs linear regression to predict gait speed from activity counts.
           Calculates predicted speed that would attain 1.5 METs (sedentary -> light).

        :returns
        -y_intercept: y-intercept from gait speed vs. counts regression
        -coefficient: slope from gait speed vs. counts regression
        -pred_light_speed:
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
        print("-r^2 = {}".format(round(lm.score(counts, speed), 5)))

        # Calculates walk speed that corresponds to 1.5 METs (i.e. light activity)
        pred_light_speed = ((1.5 * self.rvo2 - self.rvo2) / 0.1) / 60  # in m/s

        return y_intercept, coefficient, round(pred_light_speed, 5)

    def plot_regression(self):
        """Plots measured results and results predicted from regression."""

        # Measured (true) values
        plt.plot(self.tm_object.avg_walk_counts, self.tm_object.walk_speeds,
                 markerfacecolor='white', markeredgecolor='black', color='black', marker="o", label='Measured')

        # Predicted values: count range between slowest and fastest walks
        plt.plot(np.arange(self.tm_object.avg_walk_counts[0], self.tm_object.avg_walk_counts[-1], 5),
                 [round(i * self.coef + self.y_int, 3) for i in np.arange(self.tm_object.avg_walk_counts[0],
                                                                          self.tm_object.avg_walk_counts[-1], 5)],
                 label='Predicted (r^2 = {})'.format(self.r2), color='red')

        plt.legend(loc='upper left')
        plt.ylabel("Gait speed (m/s)")
        plt.xlabel("Counts")
        plt.title("Participant #{}: Treadmill Protocols - Counts vs. Gait Speed".format(self.tm_object.subjectID))
        plt.show()

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

    def pred_speed(self, input_data):
        """Predicts gait speed (m/s) using the regression equation.

        :returns:
        -predicted: list of predicted speeds rounded to 5 decimals
        """

        predicted = [round(i * self.coef + self.y_int, 5) for i in input_data]

        return predicted

    def pred_mets(self):
        """Calculates METs using ACSM equation based on predicted gait speed and resting VO2 value.

        :returns
        -mets: list of predicted MET levels
        """

        # Changes epochs that correspond to gait speed below 60% preferred as no movement
        movement_counts = self.epoch_data

        """movement_counts = []
        for i in self.predicted_speed:
            if i < self.tm_object.walk_speeds[0]:
                movement_counts.append(0)
            if i >= self.tm_object.walk_speeds[0]:
                movement_counts.append(i)"""

        # Converts m/s to m/min for use in ACSM equation
        m_min = [round(i * 60, 4) for i in movement_counts]

        # Uses ACSM equation to predict METs from predicted gait speed
        mets = [(self.rvo2 + 0.1 * i) / self.rvo2 for i in m_min]

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
        intensity_minutes = {"Sedentary": intensity.count(0) / (60/self.epoch_len),
                             "Light": intensity.count(1) / (60/self.epoch_len),
                             "Moderate": intensity.count(2) / (60/self.epoch_len),
                             "Vigorous": intensity.count(3) / (60/self.epoch_len)}

        return intensity, intensity_minutes

    def write_anklemodel(self):

        # Checks if model output file already exists
        check_folder = self.working_dir + self.modeloutput_folder + "Model Output/"  # Folder to check
        existing_files = os.listdir(check_folder)  # Files in check_folder

        # Boolean: whether file exists
        file_exists = self.anklemodel_outfile.split(".")[0].split("/")[-1] + ".csv" in existing_files

        if not file_exists:
            print("\n" + "Creating new file...")
        if file_exists and self.overwrite_existing:
            print("\n" + "Overwriting epoched intensity data file...")

        # Creates file if overwrite is set to True or if file does not exist already
        if self.overwrite_existing or not file_exists:
            with open(self.anklemodel_outfile, "w") as output:
                writer = csv.writer(output, delimiter=",", lineterminator="\n")

                writer.writerow(["Timestamp", "ActivityCount", "PredictedSpeed", "PredictedMETs", "IntensityCategory"])
                writer.writerows(zip(self.epoch_timestamps, self.epoch_data,
                                 self.predicted_speed, self.predicted_mets, self.epoch_intensity))

        if not self.overwrite_existing and file_exists:
            print("\n" + "File {} already exists and will not be overwritten.".format(self.anklemodel_outfile.split(".")
                                                                                      [0].split("/")[-1] + ".csv"))