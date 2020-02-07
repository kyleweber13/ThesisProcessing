import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class NonWear:
    """Class that stores data from participant's non-wear/sensor removal log.

    :argument
    -subjectID: integer
    -accel_object: class instance for accel object (wrist or ankle)
    -file_loc: pathway to location of removal logs (folder)
    -plot: plots non-wear data if True (default)
    """

    def __init__(self, subjectID, accel_object, nonwearlog_file, plot=True):

        self.subjectID = subjectID
        self.file_loc = nonwearlog_file
        self.accel_object = accel_object
        self.plot = plot

        self.removal_log = self.import_nonwearlog()

        if self.removal_log is not None:
            self.removal_data = self.format_removallog()
            self.wear_status = self.mark_removal_epochs()

            if self.plot:
                self.plot_sleeplog()

    def import_nonwearlog(self):
        """Imports removal log from .csv. Returns as ndarray.
           COLUMN NAMES: DATE, TIME_REMOVED, TIME_REATTACHED, SENSOR_LA, SENSOR_RA, SENSOR_LW, SENSOR_RW, SENSOR_HR"""

        # Imports log if file is found
        try:
            removal_log = np.loadtxt(fname="{}/{}_SensRemLog.csv".format(self.file_loc, self.subjectID),
                                     delimiter=",", dtype="str", skiprows=1, usecols=(3, 4, 5, 6, 7, 8, 9))

        # Returns None if no log found
        except OSError:
            print("\n" + "No removal log found.")
            return None

        return removal_log

    def format_removallog(self):
        """Formats timestamps for removal data. Returns a list of lists where each list is one day's worth of data.
           Values correspond to removal and reattachment times."""

        all_data = []

        # Loops through each day of data
        for day in self.removal_log:
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

                # Error handling if no data in cell
                except ValueError:
                    datestamp = "N/A"

                day_data.append(datestamp)

            all_data.append(day_data)

        return all_data

    def mark_removal_epochs(self):
        """Generates binary list for each epoch of whether device was worn on not. 1 = not worn."""
        epoch_index = 0

        # Creates list of 0s corresponding to each epoch
        epoch_list = np.zeros(len(self.accel_object.epoched.epoch))

        for i, epoch_stamp in enumerate(self.accel_object.epoched.epoch_timestamps):
            for removal in self.removal_data:
                if removal[0] != "N/A" and removal[1] != "N/A":
                    if removal[0] <= epoch_stamp <= removal[1]:
                        epoch_list[i] = 1

        return epoch_list

    def plot_sleeplog(self):
        """Plots epoched accelerometer data with vertical lines marking removal log data"""

        print("\n" + "Plotting device removal data...")

        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, ax1 = plt.subplots(1, figsize=(12, 7))

        plt.title("Subject {}: Removal Log Data".format(self.subjectID))

        ax1.plot(self.accel_object.epoched.epoch_timestamps[0:len(self.accel_object.epoched.epoch)],
                 self.accel_object.epoched.epoch[0:len(self.accel_object.epoched.epoch_timestamps)], color='black')

        plt.ylabel("Counts")
        ax1.xaxis.set_major_formatter(xfmt)
        ax1.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

        for removal in self.removal_data:
            if removal[0] != "N/A":
                plt.axvline(x=removal[0], color="red", label="Removed")
            if removal[1] != "N/A":
                plt.axvline(x=removal[1], color="green", label="Re-attached")

        # Fills in region where participant was asleep
        for removal in self.removal_data:
            try:
                plt.fill_betweenx(x1=removal[0], x2=removal[1], y=np.arange(0, max(self.accel_object.epoched.epoch)),
                                  color='red', alpha=0.35)

            except AttributeError:
                pass
