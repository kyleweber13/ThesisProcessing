# from owcurate.Files import Converters
import ImportEDF
import EpochData
import ConvertFile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import pandas as pd


# ConvertFile.bin_to_edf(file_in="", out_path="/Users/kyleweber/Desktop/Data/Conversion Folder/")


class Data:

    def __init__(self, accel_filepath, temperature_filepath, log_filepath=None):
        """Class that stores accelerometer and temperature data from a GENEActiv, epochs the data,
           reads in timestamps that mark device removal/reattachment from .csv, and plots results.

        :arguments
        -accel_filepath: full pathway to accelerometer file
        -temperature_pathway: full pathway to temperature file
        -log_filepath: full pathway to removal/reattachment file.
                -File should contain 2 columns for off and on timestamps, respectively."""

        self.accel_filepath = accel_filepath
        self.temperature_filepath = temperature_filepath
        self.log_filepath = log_filepath

        # Reads in accelerometer data, epochs
        self.accel_raw = ImportEDF.GENEActiv(filepath=self.accel_filepath, from_processed=False)
        self.accel_epoch = EpochData.EpochAccel(raw_data=self.accel_raw, from_processed=False, processed_folder="")

        # Reads in temperature data
        self.temperature = ImportEDF.GENEActivTemperature(filepath=self.temperature_filepath, from_processed=False)

        # Reads in removal/reattachment log
        self.off_stamps, self.on_stamps = self.read_log()

        self.plot_nonwear()

    def read_log(self):
        """Reads in removal/reattachment log and formats timestamps.

        :returns
        -off/on: lists of timestamps
        """

        file = pd.read_excel(io=self.log_filepath, header=0, usecols=(0, 1, 2))

        return file["Off"], file["On"]

    def plot_nonwear(self):
        """Plots raw triaxial accelerometer, epoched accelerometer, and temperature data on 3 subplots with shaded
           regions corresponding to worn/not worn periods."""

        # Timestamp x-axis formatting
        xfmt = mdates.DateFormatter("%m %d, %I:%M:%S %p")
        locator = mdates.HourLocator(byhour=[0, 6, 12, 18, 24], interval=1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))

        # Raw accelerometer
        ax1.plot(self.accel_raw.timestamps[::3], self.accel_raw.x[::3], color='purple',
                 label="X ({}Hz)".format(int(self.accel_raw.sample_rate / 3)))
        ax1.plot(self.accel_raw.timestamps[::3], self.accel_raw.y[::3], color='blue',
                 label="Y ({}Hz)".format(int(self.accel_raw.sample_rate / 3)))
        ax1.plot(self.accel_raw.timestamps[::3], self.accel_raw.z[::3], color='black',
                 label="Z ({}Hz)".format(int(self.accel_raw.sample_rate / 3)))
        ax1.legend(loc='upper left')
        ax1.set_ylabel("G")

        # Epoched accelerometer
        ax2.plot(self.accel_epoch.timestamps[0:len(self.accel_epoch.svm)], self.accel_epoch.svm,
                 color='black', label="Epoched accel")
        ax2.set_ylabel("Counts")
        ax2.legend(loc='upper left')

        # Temperature
        ax3.plot(self.temperature.timestamps[0:len(self.temperature.temp)], self.temperature.temp,
                 color='black', label="Temperature")
        ax3.set_ylabel("ºC")
        ax3.legend(loc='upper left')

        # Timestamp x-axis formatting
        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

        # fill_between use to shade regions
        try:
            # Non-wear periods
            for off, on in zip(self.off_stamps, self.on_stamps):
                ax1.fill_between(x=[off, on], y1=-8, y2=8, color='red', alpha=0.15)
                ax2.fill_between(x=[off, on], y1=0, y2=max(self.accel_epoch.svm), color='red', alpha=0.15)
                ax3.fill_between(x=[off, on], y1=min(self.temperature.temp), y2=max(self.temperature.temp),
                                 color='red', alpha=0.15)

            # Wear periods
            for off, on in zip(self.off_stamps[1:], self.on_stamps):
                ax1.fill_between(x=[on, off], y1=-8, y2=8, color='green', alpha=0.15)
                ax2.fill_between(x=[on, off], y1=0, y2=max(self.accel_epoch.svm), color='green', alpha=0.15)
                ax3.fill_between(x=[on, off], y1=min(self.temperature.temp), y2=max(self.temperature.temp),
                                 color='green', alpha=0.15)
        except:
            pass
