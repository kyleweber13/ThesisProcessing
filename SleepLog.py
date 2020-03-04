import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class SleepLog:
    """Imports a participant's sleep log data from .csv. Creates a list corresponding to epochs where participant was
       awake vs. napping vs. asleep (overnight). Calculates total time spent in each state. Plots sleep periods
       on ankle/wrist/HR data, if available.

    :argument
    -subject_object: object of class Subject
    -file_loc: folder that contains sleep logs
    """

    def __init__(self, subject_object, sleeplog_file=None, plot=False):

        print()
        print("=========================================== SLEEP LOG DATA ===========================================")

        self.file_loc = sleeplog_file
        self.subject_object = subject_object
        self.subjectID = self.subject_object.subjectID
        self.plot = plot

        # Sets length of data (number of epochs) and timestamps based on any data that is available
        try:
            self.data_len = len(self.subject_object.ankle.epoch.svm)
            self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps
        except AttributeError:
            try:
                self.data_len = len(self.subject_object.wrist.epoch.svm)
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except AttributeError:
                self.data_len = len(self.subject_object.ecg.epoch_timestamps)
                self.epoch_timestamps = self.subject_object.ecg.epoch_timestamps

        # Imports epoched data if available
        try:
            self.ankle_svm = self.subject_object.ankle.epoch.svm
        except AttributeError:
            self.ankle_svm = None

        try:
            self.wrist_svm = self.subject_object.wrist.epoch.svm
        except AttributeError:
            self.wrist_svm = None

        try:
            self.hr = self.subject_object.ecg.valid_hr
        except AttributeError:
            self.hr = None

        if self.file_loc is not None:
            try:
                # Imports sleep log data
                self.sleep_log = self.import_sleeplog()
                self.sleep_data = self.format_sleeplog()

                # Determines which epochs were asleep
                self.sleep_status = self.mark_sleep_epochs()

                # Sleep report
                self.sleep_report = self.generate_sleep_report()

            except OSError:
                # Handles error if file not found
                self.sleep_log = None
                self.sleep_data = None
                self.sleep_status = np.zeros(self.data_len)  # Pretends participant did not sleep
                self.sleep_report = None

        if self.file_loc is None:
            self.sleep_log = None
            self.sleep_data = None
            self.sleep_status = np.zeros(self.data_len)  # Pretends participant did not sleep
            self.sleep_report = None

        # Plots result
        if self.plot:
            self.plot_sleeplog()

    def import_sleeplog(self):
        """Imports sleep log from .csv. Only keeps values associated with Subject. Returns as ndarray.
           Column names: SUBJECT, DATE, TIME_OUT_BED, NAP_START, NAP_END, TIME_IN_BED
        """

        sleep_log = np.loadtxt(fname="{}SleepLogs_All.csv".format(self.file_loc), delimiter=",",
                               dtype="str", skiprows=1, usecols=(0, 3, 5, 7, 8, 9))
        subj_sleep_log = [i for i in sleep_log if self.subjectID in i[0]]

        return subj_sleep_log

    def format_sleeplog(self):
        """Formats timestamps for sleep data. Returns a list of lists where each list is one day's worth of data.
           Values correspond to time awake, nap time, nap wake up time, bed time, respectively."""

        all_data = []

        # Loops through each day of data
        for day in self.sleep_log:
            day_data = []

            # Formats date stamp
            date = datetime.strptime(day[1], "%Y%b%d").date()

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

            all_data.append(day_data[1:])

        return all_data

    def mark_sleep_epochs(self):
        """Creates a list of len(epoch_timestamps) where awake is coded as 0, naps coded as 1, and
           overnight sleep coded as 2"""

        # Creates list of 0s corresponding to each epoch
        epoch_list = np.zeros(self.data_len + 1)

        for i, epoch_stamp in enumerate(self.epoch_timestamps):
            try:
                epoch_stamp = datetime.strptime(str(epoch_stamp)[:-3], "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                epoch_stamp = datetime.strptime(str(epoch_stamp).split(".")[0], "%Y-%m-%d %H:%M:%S")

            for asleep, awake in zip(self.sleep_data[:], self.sleep_data[1:]):
                # Overnight sleep
                if asleep[3] != "N/A" and awake[0] != "N/A":
                    if asleep[3] <= epoch_stamp <= awake[0]:
                        epoch_list[i] = 2
                # Naps
                if asleep[1] != "N/A" and asleep[2] != "N/A":
                    if asleep[1] <= epoch_stamp <= asleep[2]:
                        epoch_list[i] = 1

        return epoch_list

    def generate_sleep_report(self):
        """Generates summary sleep measures in minutes.
        Column names: SUBJECT, DATE, TIME_OUT_BED, NAP_START, NAP_END, TIME_IN_BED
        """

        epoch_to_mins = 60 / self.subject_object.epoch_len

        sleep_durations = []
        for asleep, awake in zip(self.sleep_data[:], self.sleep_data[1:]):
            sleep_durations.append(round((awake[0] - asleep[3]).seconds / 60, 2))

        nap_durations = []
        for data in self.sleep_data:
            if data[1] != "N/A" and data[2] != "N/A":
                nap_durations.append(round((data[2] - data[1]).seconds / 60, 2))

        sleep_report = {"SleepDuration": np.sum(self.sleep_status > 0) / epoch_to_mins,
                        "Sleep%": round(100 * np.sum(self.sleep_status > 0) / len(self.epoch_timestamps), 1),

                        "OvernightSleepDuration": np.sum(self.sleep_status == 2) / epoch_to_mins,
                        "OvernightSleepDurations": sleep_durations,
                        "OvernightSleep%": round(100 * np.sum(self.sleep_status == 2) / len(self.epoch_timestamps), 1),
                        "AvgSleepDuration": round(sum(sleep_durations) / len(sleep_durations), 1),

                        "NapDuration": np.sum(self.sleep_status == 1) / epoch_to_mins,
                        "NapDurations": nap_durations,
                        "Nap%": round(100 * np.sum(self.sleep_status == 1) / len(self.epoch_timestamps), 1),
                        "AvgNapDuration": round(sum(nap_durations) / len(nap_durations), 1)
                        if len(nap_durations) != 0 else 0
                        }

        print("\n" + "SLEEP REPORT")

        print("-Total time asleep: {} minutes ({}%)".format(sleep_report["SleepDuration"], sleep_report["Sleep%"]))

        print("\n" + "-Total overnight sleep: {} minutes ({}%)".format(sleep_report["OvernightSleepDuration"],
                                                                       sleep_report["OvernightSleep%"]))
        print("-Overnight sleep durations: {} minutes".format(sleep_report["OvernightSleepDurations"]))
        print("-Average overnight sleep duration: {} minutes".format(sleep_report["AvgSleepDuration"]))

        print("\n" + "-Total napping time: {} minutes ({}%)".format(sleep_report["NapDuration"], sleep_report["Nap%"]))
        print("-Nap durations: {} minutes".format(sleep_report["NapDurations"]))
        print("-Average nap duration: {} minutes".format(sleep_report["AvgNapDuration"]))

        return sleep_report

    def plot_sleeplog(self):
        """Plots epoched accelerometer data with shaded regions marking sleep log data.
           Plots ankle/wrist/HR data if available"""

        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(12, 7))

        ax1.set_title("Subject {}: Sleep Log Data (green = sleep; red = nap)".format(self.subjectID))

        # WRIST ACCELEROMETER ----------------------------------------------------------------------------------------
        try:
            ax1.plot(self.epoch_timestamps[:len(self.wrist_svm)], self.wrist_svm[:len(self.epoch_timestamps)],
                     label='Wrist', color='black')
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Counts")

            for day in self.sleep_data:
                for index, value in enumerate(day):
                    if value != "N/A":
                        if index == 0:
                            ax1.axvline(x=value, color="green", label="Woke up")

                        if index == 3:
                            ax1.axvline(x=value, color="green", label="To bed")

                        if index == 1:
                            ax1.axvline(x=value, color="red", label="Nap")

                        if index == 2:
                            ax1.axvline(x=value, color="red", label="Wake up from nap")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep_data[:], self.sleep_data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, max(self.wrist_svm)),
                                      color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":
                    # Naps --> red
                    ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, max(self.wrist_svm)),
                                      color='red', alpha=0.35)

        except (AttributeError, TypeError):
            pass

        # ANKLE ACCELEROMETER ----------------------------------------------------------------------------------------
        try:
            ax2.plot(self.epoch_timestamps[:len(self.ankle_svm)], self.ankle_svm[:len(self.epoch_timestamps)],
                     label='Ankle', color='black')
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Counts")

            for day in self.sleep_data:
                for index, value in enumerate(day):
                    if value != "N/A":
                        if index == 0:
                            ax2.axvline(x=value, color="green", label="Woke up")

                        if index == 3:
                            ax2.axvline(x=value, color="green", label="To bed")

                        if index == 1:
                            ax2.axvline(x=value, color="red", label="Nap")

                        if index == 2:
                            ax2.axvline(x=value, color="red", label="Wake up from nap")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep_data[:], self.sleep_data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, max(self.ankle_svm)),
                                      color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":
                    # Naps --> red
                    ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, max(self.ankle_svm)),
                                      color='red', alpha=0.35)

        except (AttributeError, TypeError):
            pass

        # HEART RATE -------------------------------------------------------------------------------------------------

        try:
            ax3.plot(self.epoch_timestamps[:len(self.hr)], self.hr[:len(self.epoch_timestamps)],
                     label='HR', color='black')
            ax3.legend(loc='upper left')
            ax3.set_ylabel("HR (bpm)")

            for day in self.sleep_data:
                for index, value in enumerate(day):
                    if value != "N/A":
                        if index == 0:
                            ax3.axvline(x=value, color="green", label="Woke up")

                        if index == 3:
                            ax3.axvline(x=value, color="green", label="To bed")

                        if index == 1:
                            ax3.axvline(x=value, color="red", label="Nap")

                        if index == 2:
                            ax3.axvline(x=value, color="red", label="Wake up from nap")

            # Fills in region where participant was asleep
            for day1, day2 in zip(self.sleep_data[:], self.sleep_data[1:]):
                if day1[3] != "N/A" and day2[0] != "N/A":
                    # Overnight --> green
                    ax3.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(min([i for i in self.hr if i is not None]),
                                                                          max([i for i in self.hr if i is not None])),
                                      color='green', alpha=0.35)

                if day1[2] != "N/A" and day1[1] != "N/A":

                    # Naps --> red
                    ax3.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(min([i for i in self.hr if i is not None]),
                                                                          max([i for i in self.hr if i is not None])),
                                      color='red', alpha=0.35)

        except (AttributeError, TypeError, ValueError):
            pass

        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)
