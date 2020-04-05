# Adapted from Adam Vert

import pyedflib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import datetime as dt
from scipy import signal
import ImportEDF


class Nonwear:

    def __init__(self, accel_object=None):
        """
        Note that we are using a forward moving window of 4s since that is how often we get temperature results. The
        original study by Zhou used 1s
        """

        self.load_raw = accel_object.load_raw
        self.accel = accel_object.raw
        self.temperature = accel_object.temperature

        if not self.load_raw:
            print("\n" + "No raw data available. Skipping non-wear processing.")
            self.status = None
        if self.load_raw:
            print("\n" + "Running Zhou non-wear algorithm...")
            self.data = self.zhou_nonwear()
            self.status = self.data["Wear Status"]
            self.timestamps = self.data["End Time"]

    def zhou_nonwear(self):

        temp_thresh = 26
        window_size = 60  # in seconds

        pd.set_option('mode.chained_assignment', None)

        zhou_df = pd.DataFrame({"Temperature Timestamps": self.temperature.timestamps,
                                "Raw Temperature Values": self.temperature.temperature})

        # Temperature data --------------------------------------------------------------------------------------------

        # Sliding window: average temperature over 60 * self.temperature.sample_rate seconds
        temperature_moving_average = pd.Series(self.temperature.temperature).rolling(
            window=int(60 * self.temperature.sample_rate)).mean()

        t0 = 26

        # Accelerometer data -----------------------------------------------------------------------------------------

        # Raw accel dataframe
        zhou_accelerometer_df = pd.DataFrame({"X": self.accel.x, "Y": self.accel.y, "Z": self.accel.z},
                                             index=self.accel.timestamps)

        # Rolling average accelerometer standard deviation
        zhou_accelerometer_rolling_std = zhou_accelerometer_df.rolling(window=int(60 * self.accel.sample_rate)).std()

        # Takes one row from data every 4 seconds
        binned_4s_df = zhou_accelerometer_rolling_std.iloc[::int(4 * self.accel.sample_rate), :]

        # Combined data ----------------------------------------------------------------------------------------------
        temp_moving_average_list = list(temperature_moving_average.values)

        if len(temp_moving_average_list) - len(binned_4s_df) > 0:
            temp_moving_average_list = temp_moving_average_list[:len(binned_4s_df) - len(temp_moving_average_list)]

        binned_4s_df["Temperature Moving Average"] = temp_moving_average_list

        # Algorithm --------------------------------------------------------------------------------------------------

        worn = []
        end_times = []
        for index, row in binned_4s_df.iterrows():

            # Timestamp generation
            end_times.append(index + dt.timedelta(seconds=4))

            if (row["Temperature Moving Average"] < t0) and (((row["X"] + row["Y"] + row["Z"]) / 3) < 0.013):
                worn.append(False)
            elif row["Temperature Moving Average"] >= t0:
                worn.append(True)
            else:
                earlier_window_temp = binned_4s_df["Temperature Moving Average"].shift(15).loc[index]
                if row["Temperature Moving Average"] > earlier_window_temp:
                    worn.append(True)
                elif row["Temperature Moving Average"] < earlier_window_temp:
                    worn.append(False)
                elif row["Temperature Moving Average"] == earlier_window_temp:
                    worn.append(worn[-1])
                else:
                    worn.append(True)

        binned_4s_df["Wear Status"] = worn
        binned_4s_df["End Time"] = end_times

        final_df = binned_4s_df[['End Time', 'Wear Status']].copy()

        return final_df


class NonwearLog:

    def __init__(self, subject_object):

        print("")
        print("=================================== ACCELEROMETER NONWEAR DATA ======================================")

        self.subject_object = subject_object
        self.subjectID = subject_object.subjectID
        self.epoch_timestamps = None
        self.data_len = 0

        self.file_loc = subject_object.nonwear_file
        self.status = []
        self.nonwear_log = None

        self.nonwear_dict = {"Minutes": 0, "Average Duration (Mins)": 0, "Percent of Time": 0}
        self.nonwear_minutes = 0
        self.avg_nonwear_duration = 0
        self.nonwear_percent = 0

        self.prep_data()
        self.import_nonwearlog()
        self.mark_nonwear_epochs()

    def prep_data(self):

        if self.subject_object.load_wrist:
            try:
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except (AttributeError, TypeError):
                self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps

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

    def import_nonwearlog(self):

        if self.file_loc is not None and os.path.exists(self.file_loc):

            # Imports sleep log data from CSV
            nonwear_log = pd.read_excel(io=self.file_loc, columns=["ID", "DEVICE OFF", "DEVICE ON"])

            nonwear_log = nonwear_log.loc[nonwear_log["ID"] == self.subjectID]

            nonwear_log["DEVICE OFF"] = pd.to_datetime(nonwear_log["DEVICE OFF"], format="%Y%b%d %H:%M")
            nonwear_log["DEVICE ON"] = pd.to_datetime(nonwear_log["DEVICE ON"], format="%Y%b%d %H:%M")

            nonwear_log = nonwear_log.fillna(self.epoch_timestamps[-1])
            nonwear_durs = nonwear_log["DEVICE ON"] - nonwear_log["DEVICE OFF"]

            self.nonwear_log = nonwear_log

            self.nonwear_log["PERIOD DURATION"] = nonwear_durs

            self.nonwear_dict["Average Duration (Mins)"] = round(self.nonwear_log.describe()
                                                                 ["PERIOD DURATION"]['mean'].total_seconds() / 60, 1)

            print("\nNon-wear log data imported. Found {} removals.".format(self.nonwear_log.shape[0]))

        if self.file_loc is None or not os.path.exists(self.file_loc):

            self.status = np.zeros(self.data_len)  # Pretends participant did not remove device

    def mark_nonwear_epochs(self):
        """Creates a list of len(epoch_timestamps) where worn is coded as 0 and non-wear coded as 1"""

        if self.file_loc is None or not os.path.exists(self.file_loc):
            print("\nNo log found. Skipping non-wear epoch marking...")
            return None

        print("\nMarking non-wear epochs...")

        # Puts pd.df data into list --> mucho faster
        off_stamps = [i for i in self.nonwear_log["DEVICE OFF"]]
        on_stamps = [i for i in self.nonwear_log["DEVICE ON"]]

        # Creates list of 0s corresponding to each epoch
        epoch_list = np.zeros(self.data_len)

        for i, epoch_stamp in enumerate(self.epoch_timestamps):
            for off, on in zip(off_stamps, on_stamps):
                if off <= epoch_stamp <= on:
                    epoch_list[i] = 1

        self.status = epoch_list

        self.nonwear_dict["Minutes"] = (np.count_nonzero(self.status)) / (60 / self.subject_object.epoch_len)
        self.nonwear_dict["Percent"] = round(self.nonwear_dict["Minutes"] * (60 / self.subject_object.epoch_len) / \
                                             len(self.status), 4)

        print("Complete. Found {} hours, {} minutes of "
              "non-wear time.".format(np.floor(self.nonwear_dict["Minutes"]/60), self.nonwear_dict["Minutes"] % 60))


class ZhouNonwear:
    """Coded by Adam Vert"""

    def __init__(self, subject_object=None):

        print("\nRunning Zhou algorithm to find non-wear periods in wrist accelerometer data...")

        self.accelerometer = None
        self.x_values = None
        self.y_values = None
        self.z_values = None
        self.accelerometer_frequency = None
        self.accelerometer_start_datetime = None
        self.accelerometer_duration = None
        self.accelerometer_endtime = None
        self.accelerometer_timestamps = None
        self.temperature = None
        self.temperature_values = None
        self.temperature_frequency = None
        self.temperature_start_datetime = None
        self.temperature_duration = None
        self.temperature_endtime = None
        self.epoch_timestamps = None
        self.nw_start_times = None
        self.nw_end_times = None
        self.status = []

        self.subject_id = None

        """RUNS METHODS"""
        if subject_object.wrist_filepath is not None and subject_object.wrist_temp_filepath is not None:
            self.read_accelerometer(subject_object.wrist_filepath)
            self.read_temperature(subject_object.wrist_temp_filepath)

            zhou_nw_df = self.run_algorithm(minimum_window_size=60, t0=26)

            self.nw_start_times = list(zhou_nw_df.index)
            self.nw_end_times = list(zhou_nw_df["End Time"])
            self.status = list(zhou_nw_df["Device Worn"])
            self.status = [int(not i) for i in self.status]  # Flips values - how I do it

    def read_accelerometer(self, path_to_accelerometer):
        """
        Read in accelerometer values into the SensorScripts class
        Args:
            path_to_accelerometer: full path to accelerometer EDF
        """
        if not os.path.exists(path_to_accelerometer):
            return

        self.accelerometer = pyedflib.EdfReader(path_to_accelerometer)
        self.x_values = self.accelerometer.readSignal(0)
        self.y_values = self.accelerometer.readSignal(1)
        self.z_values = self.accelerometer.readSignal(2)
        self.accelerometer_frequency = self.accelerometer.samplefrequency(0)
        self.accelerometer_start_datetime = self.accelerometer.getStartdatetime()
        self.accelerometer_duration = self.accelerometer.getFileDuration()
        self.accelerometer_endtime = self.accelerometer_start_datetime + dt.timedelta(
            seconds=len(self.x_values) / self.accelerometer_frequency)  # Currently using X values
        self.accelerometer_timestamps = np.asarray(
            pd.date_range(self.accelerometer_start_datetime, self.accelerometer_endtime,
                          periods=len(self.x_values)))  # Currently using X values

        if self.subject_id is None or self.subject_id == os.path.basename(path_to_accelerometer).split("_")[-1][:-4]:
            self.subject_id = os.path.basename(path_to_accelerometer).split("_")[-1][:-4]

        self.accelerometer.close()

    def read_temperature(self, path_to_temperature):
        """
        Read in temperature values into the SensorScripts class
        Args:
            path_to_temperature: full path to temperature EDF
        """
        if not os.path.exists(path_to_temperature):
            return

        self.temperature = pyedflib.EdfReader(path_to_temperature)
        self.temperature_values = self.temperature.readSignal(0)
        self.temperature_frequency = 0.25  # self.temperature.samplefrequency(0)
        self.temperature_start_datetime = self.temperature.getStartdatetime()
        self.temperature_duration = self.temperature.getFileDuration()

        self.temperature_endtime = self.temperature_start_datetime + dt.timedelta(
            seconds=len(self.temperature_values) / self.temperature_frequency)
        self.epoch_timestamps = np.asarray(
            pd.date_range(self.temperature_start_datetime, self.temperature_endtime,
                          periods=len(self.temperature_values)))

        if self.subject_id is None or self.subject_id == os.path.basename(path_to_temperature).split("_")[-1][:-4]:
            self.subject_id = os.path.basename(path_to_temperature).split("_")[-1][:-4]

        self.temperature.close()

    def run_algorithm(self, minimum_window_size=15, t0=26):

        temp_thresh = 26
        window_size = 60  # seconds

        # Temperature ------------------------------------------------------------------------------------------------
        pd.set_option('mode.chained_assignment', None)
        zhou_df = pd.DataFrame({"Temperature Timestamps": self.epoch_timestamps,
                                "Raw Temperature Values": self.temperature_values})

        temperature_moving_average = pd.Series(self.temperature_values).rolling(
            int(60 * self.temperature_frequency)).mean()

        # Accelerometer ----------------------------------------------------------------------------------------------

        zhou_accelerometer_df = pd.DataFrame({"X": self.x_values, "Y": self.y_values, "Z": self.z_values},
                                             index=self.accelerometer_timestamps)
        zhou_accelerometer_rolling_std = zhou_accelerometer_df.rolling(int(60 * self.accelerometer_frequency)).std()
        binned_4s_df = zhou_accelerometer_rolling_std.iloc[::int(4 * self.accelerometer_frequency), :]

        # Combined
        temp_moving_average_list = list(temperature_moving_average.values)
        if len(temp_moving_average_list) - len(binned_4s_df) > 0:
            temp_moving_average_list = temp_moving_average_list[:len(binned_4s_df) - len(temp_moving_average_list)]

        binned_4s_df["Temperature Moving Average"] = temp_moving_average_list

        # Zhou Algorithm ---------------------------------------------------------------------------------------------

        not_worn = []
        end_times = []
        for index, row in binned_4s_df.iterrows():
            end_times.append(index + dt.timedelta(seconds=4))
            if (row["Temperature Moving Average"] < t0) and (((row["X"] + row["Y"] + row["Z"]) / 3) < 0.013):
                not_worn.append(True)
            elif row["Temperature Moving Average"] >= t0:
                not_worn.append(False)
            else:
                earlier_window_temp = binned_4s_df["Temperature Moving Average"].shift(15).loc[index]
                if row["Temperature Moving Average"] > earlier_window_temp:
                    not_worn.append(False)
                elif row["Temperature Moving Average"] < earlier_window_temp:
                    not_worn.append(True)
                elif row["Temperature Moving Average"] == earlier_window_temp:
                    not_worn.append(not_worn[-1])
                else:
                    not_worn.append(False)

        binned_4s_df["Bin Not Worn?"] = not_worn
        binned_4s_df["Bin Worn Consecutive Count"] = \
            binned_4s_df["Bin Not Worn?"] * \
            (binned_4s_df["Bin Not Worn?"].groupby((binned_4s_df["Bin Not Worn?"]
                                                    != binned_4s_df["Bin Not Worn?"].shift()).cumsum()).cumcount() + 1)

        binned_4s_df["Device Worn"] = True
        binned_4s_df["Device Worn"].loc[binned_4s_df["Bin Worn Consecutive Count"] >=
                                         minimum_window_size / (4/60)] = False
        binned_4s_df["End Time"] = end_times

        final_df = binned_4s_df[['End Time', 'Device Worn']].copy()

        return final_df
