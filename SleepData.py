import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


class Sleep:
    """Imports a participant's sleep log data from .csv. Creates a list corresponding to epochs where participant was
       awake vs. napping vs. asleep (overnight). Calculates total time spent in each state. Plots sleep periods
       on ankle/wrist/HR data, if available.

    :argument
    -subject_object: object of class Subject
    -file_loc: folder that contains sleep logs
    """

    def __init__(self, subject_object):

        print()
        print("=========================================== SLEEP LOG DATA ===========================================")

        self.file_loc = subject_object.sleeplog_file
        self.subject_object = subject_object
        self.subjectID = self.subject_object.subjectID

        self.data_len = 0
        self.epoch_timestamps = None

        self.log = None
        self.data = None
        self.status = None
        self.report = {"SleepDuration": 0, "Sleep%": 0,
                       "OvernightSleepDuration": 0, "OvernightSleepDurations": 0,
                       "OvernightSleep%": 0, "AvgSleepDuration": 0,
                       "NapDuration": 0, "NapDurations": 0, "Nap%": 0, "AvgNapDuration": 0}

        # RUNS METHODS ===============================================================================================
        self.import_data()
        self.log = self.import_sleeplog()

        if self.file_loc is not None and os.path.exists(self.file_loc):
            self.data = self.format_sleeplog()

            # Determines which epochs were asleep
            self.status = self.mark_sleep_epochs()

            # Sleep report
            self.report = self.generate_sleep_report()

    def import_data(self):

        # MODEL DATA =================================================================================================
        # Sets length of data (number of epochs) and timestamps based on any data that is available
        try:
            self.data_len = len(self.subject_object.ankle.epoch.timestamps)
            self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps
        except AttributeError:
            try:
                self.data_len = len(self.subject_object.wrist.epoch.timestamps)
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except AttributeError:
                self.data_len = len(self.subject_object.ecg.epoch_timestamps)
                self.epoch_timestamps = self.subject_object.ecg.epoch_timestamps

    def import_sleeplog(self):
        """Imports sleep log from .csv. Only keeps values associated with Subject. Returns as ndarray.
           Column names: SUBJECT, DATE, TIME_OUT_BED, NAP_START, NAP_END, TIME_IN_BED
        """

        if self.file_loc is not None and os.path.exists(self.file_loc):

            # Imports sleep log data from CSV
            sleep_log = np.loadtxt(fname="{}".format(self.file_loc), delimiter=",",
                                   dtype="str", skiprows=1, usecols=(0, 3, 5, 7, 8, 9))

            subj_log = [i for i in sleep_log if str(self.subjectID) in i[0]]

            return subj_log

        if self.file_loc is None or not os.path.exists(self.file_loc):

            self.status = np.zeros(self.data_len)  # Pretends participant did not sleep

    def format_sleeplog(self):
        """Formats timestamps for sleep data. Returns a list of lists where each list is one day's worth of data.
           Values correspond to time awake, nap time, nap wake up time, bed time, respectively."""

        all_data = []

        # Loops through each day of data
        for day in self.log:
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

            for asleep, awake in zip(self.data[:], self.data[1:]):
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
        for asleep, awake in zip(self.data[:], self.data[1:]):
            sleep_durations.append(round((awake[0] - asleep[3]).seconds / 60, 2))

        nap_durations = []
        for data in self.data:
            if data[1] != "N/A" and data[2] != "N/A":
                nap_durations.append(round((data[2] - data[1]).seconds / 60, 2))

        report = {"SleepDuration": np.sum(self.status > 0) / epoch_to_mins,
                  "Sleep%": round(100 * np.sum(self.status > 0) / len(self.epoch_timestamps), 1),
                  "OvernightSleepDuration": np.sum(self.status == 2) / epoch_to_mins,
                  "OvernightSleepDurations": sleep_durations,
                  "OvernightSleep%": round(100 * np.sum(self.status == 2) / len(self.epoch_timestamps), 1),
                  "AvgSleepDuration": round(sum(sleep_durations) / len(sleep_durations), 1),
                  "NapDuration": np.sum(self.status == 1) / epoch_to_mins,
                  "NapDurations": nap_durations,
                  "Nap%": round(100 * np.sum(self.status == 1) / len(self.epoch_timestamps), 1),
                  "AvgNapDuration": round(sum(nap_durations) / len(nap_durations), 1)
                  if len(nap_durations) != 0 else 0}

        print("\n" + "SLEEP REPORT")

        print("-Total time asleep: {} minutes ({}%)".format(report["SleepDuration"], report["Sleep%"]))

        print("\n" + "-Total overnight sleep: {} minutes ({}%)".format(report["OvernightSleepDuration"],
                                                                       report["OvernightSleep%"]))
        print("-Overnight sleep durations: {} minutes".format(report["OvernightSleepDurations"]))
        print("-Average overnight sleep duration: {} minutes".format(report["AvgSleepDuration"]))

        print("\n" + "-Total napping time: {} minutes ({}%)".format(report["NapDuration"], report["Nap%"]))
        print("-Nap durations: {} minutes".format(report["NapDurations"]))
        print("-Average nap duration: {} minutes".format(report["AvgNapDuration"]))

        return report

