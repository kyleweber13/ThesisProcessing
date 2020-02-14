from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class ValidData:

    def __init__(self, subject_object=None, plot=False):
        """Generates a class instance that creates and stores data where all devices/models generated valid data.
           Removes periods of invalid ECG data, sleep, and non-wear (NOT CURRENTLY IMPLEMENTED)"""

        print()
        print("====================================== REMOVING INVALID DATA ========================================")

        self.subject_object = subject_object
        self.plot = plot

        # Data sets ---------------------------------------------------------------------------------------------------
        self.data_len = 1
        self.epoch_timestamps = None

        # Data from other objects; invalid data not removed
        self.ankle_intensity = None
        self.wrist_intensity = None
        self.hr_intensity = None
        self.hracc_intensity = None

        # Data used to determine what epochs are valid
        self.hr_validity = None
        self.sleep_validity = None

        # Data that only contains valid epochs
        self.ankle = None
        self.wrist = None
        self.hr = None
        self.hr_acc = None

        self.percent_valid = None

        # =============================================== RUNS METHODS ================================================

        # Organizing data depending on what data are available
        self.organize_data()

        # Data used to determine which epochs are valid ---------------------------------------------------------------
        # Removal based on HR data
        if self.hr_validity is not None:
            self.remove_invalid_hr()

        # Removal based on sleep data
        if self.sleep_validity is not None:
            self.remove_invalid_sleep()

        self.generate_validity_report()

        if self.plot:
            self.plot_validity_data()

    def organize_data(self):

        # Uses timestamps from an available device (should all be equivalent)
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

        # Ankle intensity data
        self.ankle_intensity = self.subject_object.ankle.model.epoch_intensity if \
            self.subject_object.ankle.model.epoch_intensity is not None else None

        # Wrist intensity data
        self.wrist_intensity = self.subject_object.wrist.model.epoch_intensity if \
            self.subject_object.wrist.model.epoch_intensity is not None else None

        # HR intensity data
        self.hr_intensity = self.subject_object.ecg.epoch_intensity if \
            self.subject_object.ecg.epoch_intensity is not None else None

        # HR validity status
        self.hr_validity = self.subject_object.ecg.epoch_validity if \
            self.subject_object.ecg.epoch_validity is not None else None

        # HR-Acc intensity data
        self.hracc_intensity = self.subject_object.hracc.model.epoch_intensity if \
            self.subject_object.hracc is not None else None  # UPDATE THIS ONCE IT EXISTS

        # Sleep validity data
        self.sleep_validity = self.subject_object.sleep.sleep_status if \
            self.subject_object.sleep.sleep_status is not None else None

    def remove_invalid_hr(self):
        """Removes invalid epochs from Ankle and Wrist data based on HR validity."""

        print("\n" + "Removing invalid HR epochs...")

        self.hr = self.hr_intensity

        if self.ankle_intensity is not None:
            self.ankle = [self.ankle_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]

        if self.wrist_intensity is not None:
            self.wrist = [self.wrist_intensity[i] if self.hr_validity[i] == 0 else None for i in range(self.data_len)]

        print("Complete.")

    def remove_invalid_sleep(self):
        """Removes invalid epochs from Ankle, Wrist, and HR data based on sleep validity.
           If invalid data has been removed using HR data, this method further removes invalid periods due to sleep.
           If invalid data has not been removed using HR data, this method removes invalid periods due to sleep from
           the "raw" data.
        """

        print("\n" + "Removing epochs during sleep...")

        # Ankle -------------------------------------------------------------------------------------------------------
        # If ankle data available and invalid data was not removed using HR
        if self.ankle_intensity is not None and self.ankle is None:
            self.ankle = [self.ankle_intensity[i] if self.sleep_validity[i] == 0 else None
                          for i in range(self.data_len)]

        # If ankle data available and invalid data was removed
        if self.ankle_intensity is not None and self.ankle is not None:
            self.ankle = [self.ankle[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        # Wrist ------------------------------------------------------------------------------------------------------
        # If wrist data available and invalid data was not removed using HR
        if self.wrist_intensity is not None and self.wrist is None:
            self.wrist = [self.wrist_intensity[i] if self.sleep_validity[i] == 0 else None
                          for i in range(self.data_len)]

        # If wrist data available and invalid data was removed
        if self.wrist_intensity is not None and self.wrist is not None:
            self.wrist = [self.wrist[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        # Heart Rate -------------------------------------------------------------------------------------------------
        if self.hr_intensity is not None and self.hr is None:
            self.hr = [self.hr_intensity[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        if self.hr_intensity is not None and self.hr is not None:
            self.hr = [self.hr[i] if self.sleep_validity[i] == 0 else None for i in range(self.data_len)]

        print("Complete.")

    def generate_validity_report(self):

        try:
            self.percent_valid = round(100 * (self.data_len - self.ankle.count(None)) / self.data_len, 1)
        except TypeError:
            try:
                self.percent_valid = round(100 * (self.data_len - self.wrist.count(None)) / self.data_len, 1)
            except TypeError:
                self.percent_valid = round(100 * (self.data_len - self.hr.count(None)) / self.data_len, 1)

        print("\n" + "Validity check complete. {}% of the original data is valid.".format(self.percent_valid))

    def plot_validity_data(self):
        """Generates 4 subplots for each activity model with invalid data removed."""

        # x-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 7))
        ax1.set_title("Participant {}: Valid Data ({}% valid)".format(self.subject_object.subjectID,
                                                                      self.percent_valid))

        # Fills in region where participant was asleep
        for day1, day2 in zip(self.subject_object.sleep.sleep_data[:], self.subject_object.sleep.sleep_data[1:]):
            try:
                # Overnight sleep
                ax1.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax2.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax3.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)
                ax4.fill_betweenx(x1=day1[3], x2=day2[0], y=np.arange(0, 4), color='black', alpha=0.35)

                # Daytime naps
                ax1.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax2.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax3.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)
                ax4.fill_betweenx(x1=day1[2], x2=day1[1], y=np.arange(0, 4), color='black', alpha=0.35)

            except AttributeError:
                pass

        if self.ankle_intensity is not None:
            ax1.plot(self.epoch_timestamps, self.ankle[:self.data_len], color='#606060', label='Ankle')
            ax1.set_ylim(-0.1, 3)
            ax1.legend(loc='upper left')
            ax1.set_ylabel("Intensity Cat.")

        if self.wrist_intensity is not None:
            ax2.plot(self.epoch_timestamps, self.wrist[:self.data_len], color='#606060', label='Wrist')
            ax2.set_ylim(-0.1, 3)
            ax2.legend(loc='upper left')
            ax2.set_ylabel("Intensity Cat.")

        if self.hr_intensity is not None:
            ax3.plot(self.epoch_timestamps, self.hr[:self.data_len], color='red', label='HR')
            ax3.set_ylim(-0.1, 3)
            ax3.legend(loc='upper left')
            ax3.set_ylabel("Intensity Cat.")

        if self.hracc_intensity is not None:
            ax4.plot(self.epoch_timestamps, self.hr_acc[:self.data_len], color='black', label='HR-Acc')
            ax4.set_ylim(-0.1, 3)
            ax4.legend(loc='upper left')
            ax4.set_ylabel("Intensity Cat.")

        ax4.xaxis.set_major_formatter(xfmt)
        ax4.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)

        plt.show()
