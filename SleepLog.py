import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class SleepLog:
    """Imports a participant's sleep log data and stores it as a dictionary.

    :argument
    -subjectID: integer
    -file_loc: folder that contains sleep logs
    """

    def __init__(self, subjectID, accel_object, sleeplog_file, plot=False):

        self.subjectID = subjectID
        self.file_loc = sleeplog_file
        self.accel_object = accel_object
        self.plot = plot

        # Imports sleep log data
        self.sleep_log = self.import_sleeplog()
        self.sleep_data = self.format_sleeplog()

        # Determines which epochs were asleep
        self.sleep_status = self.mark_sleep_epochs()

        # Plots result
        if self.plot:
            self.plot_sleeplog()

    def import_sleeplog(self):
        """Imports sleep log from .csv. Returns as ndarray.
        Column names: Date, TIME_OUT_BED, NAP_START, NAP_END, TIME_IN_BED
        """

        sleep_log = np.loadtxt(fname="{}{}_SleepLog.csv".format(self.file_loc, self.subjectID), delimiter=",",
                               dtype="str", skiprows=1, usecols=(3, 5, 7, 8, 9))

        return sleep_log

    def format_sleeplog(self):
        """Formats timestamps for sleep data. Returns a list of lists where each list is one day's worth of data.
           Values correspond to time awake, nap time, nap wake up time, bed time, respectively."""

        all_data = []

        # Loops through each day of data
        for day in self.sleep_log:
            day_data = []

            # Formats date stamp
            date = datetime.strptime(day[0], "%Y%b%d").date()

            # Loops through other values for each day
            for value in range(1, len(day)):

                # Formatting
                try:
                    # Combines date + time
                    datestamp = datetime.strptime(str(date) + " " + day[value], "%Y-%m-%d %H:%M")

                    # Extracts hour of day (integer)
                    hour_of_day = datetime.strptime(day[value], "%H:%M").hour

                    # Changes date to next day if went to bed after midnight
                    if value == len(day) - 1 and hour_of_day < 6:
                        datestamp += timedelta(days=1)

                # Error handling if no data in cell
                except ValueError:
                    datestamp = "N/A"

                day_data.append(datestamp)

            all_data.append(day_data)

        return all_data

    def mark_sleep_epochs(self):

        epoch_index = 0

        # Creates list of 0s corresponding to each epoch
        epoch_list = np.zeros(len(self.accel_object.epoched.epoch))

        for i, epoch_stamp in enumerate(self.accel_object.epoched.epoch_timestamps):
            for asleep, awake in zip(self.sleep_data[:], self.sleep_data[1:]):
                if asleep[3] != "N/A" and awake[0] != "N/A":
                    if asleep[3] <= epoch_stamp <= awake[0]:
                        epoch_list[i] = 500

        return epoch_list

    def plot_sleeplog(self):
        """Plots epoched accelerometer data with vertical lines marking sleep log data"""

        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, ax1 = plt.subplots(1, figsize=(12, 7))

        plt.title("Subject {}: Sleep Log Data (green = sleep; red = nap)".format(self.subjectID))

        ax1.plot(self.accel_object.epoched.epoch_timestamps[0:len(self.accel_object.epoched.epoch)],
                 self.accel_object.epoched.epoch[0:len(self.accel_object.epoched.epoch_timestamps)], color='black')

        plt.ylabel("Counts")
        ax1.xaxis.set_major_formatter(xfmt)
        ax1.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

        for day in self.sleep_data:
            for index, value in enumerate(day):
                if value != "N/A":
                    if index == 0:
                        plt.axvline(x=value, color="green", label="Woke up")
                    if index == 3:
                        plt.axvline(x=value, color="green", label="To bed")

                    if index == 1:
                        plt.axvline(x=value, color="red", label="Nap")
                    if index == 2:
                        plt.axvline(x=value, color="red", label="Wake up from nap")

        # Fills in region where participant was asleep
        for day1, day2 in zip(self.sleep_data[:], self.sleep_data[1:]):
            try:
                # Overnight --> green
                plt.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, max(self.accel_object.epoched.epoch)),
                                  color='green', alpha=0.35)

                # Naps --> red
                plt.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, max(self.accel_object.epoched.epoch)),
                                  color='red', alpha=0.35)
            except AttributeError:
                pass
